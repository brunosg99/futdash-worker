# core/core_logic.py
# VERSÃO OTIMIZADA (GPU + BATCH + FP16) — pronta para usar no worker

import argparse, json, cv2, torch, numpy as np
from pathlib import Path
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn
import os
import logging
import pytesseract
from PIL import Image, ImageOps, ImageFilter
import re
from contextlib import nullcontext

logging.basicConfig(level=logging.INFO)

# === Constantes ===
ROIS_JSON = "data/rois.json"
TIPOS_PERCENT = [
    "circulo_time1_dribles","circulo_time1_precisao_fin","circulo_time1_precisao_pas",
    "circulo_time2_dribles","circulo_time2_precisao_fin","circulo_time2_precisao_pas"
]
TIPOS_DEC = ["tabela_time1_gols_esperados","tabela_time2_gols_esperados"]
CLASS_NAMES = [str(i) for i in range(10)] + ["comma"]
ID2CHAR = {i:n for i,n in enumerate(CLASS_NAMES)}
CKPT_CHAR = "data/checkpoints/char_cnn_best.pt"

# ===================================================================
# Helpers gerais
# ===================================================================
def crop_xywh(img, box, inset=2):
    x,y,w,h = box
    x=max(0,int(x+inset)); y=max(0,int(y+inset)); w=int(w-2*inset); h=int(h-2*inset)
    H,W = img.shape[:2]; x2=min(W,x+w); y2=min(H,y+h)
    if x>=x2 or y>=y2: return img[0:1,0:1].copy()
    return img[y:y2, x:x2].copy()

def resize_with_scale(img, scale):
    if scale == 1.0: return img, 1.0
    h, w = img.shape[:2]; nh, nw = int(h*scale), int(w*scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC), scale

def filter_decimal_boxes(dets, roi_h, roi_w):
    out = []
    for (x,y,w,h,cls,cf) in dets:
        if not (0.25 <= h/max(1.0,roi_h) <= 1.05): continue
        ar = w/max(1.0,h); ch = ID2CHAR.get(cls,"")
        if (ch in "0123456789" and ar < 1.35) or (ch == "comma" and ar < 0.80):
            out.append((x,y,w,h,cls,cf))
    out.sort(key=lambda t: (t[2], -t[5]))
    keep = out[:3]
    keep.sort(key=lambda t: t[0])
    return keep

def filter_int_boxes(dets, roi_h, roi_w, max_digits=4):
    keep = []
    for (x,y,w,h,cls,cf) in dets:
        if ID2CHAR.get(cls,"") in "0123456789" and 0.30 <= h/max(1.0,roi_h) <= 0.95 and w/max(1.0,h) < 1.05:
            keep.append((x,y,w,h,cls,cf))
    keep.sort(key=lambda t: t[5], reverse=True)
    out = keep[:max_digits]
    out.sort(key=lambda t: t[0])
    return out

def looks_like_one_patch(g):
    gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ys, xs = np.where(bw>0); H,W = bw.shape
    if len(xs)==0: return False
    ww, hh = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    return (hh >= H*0.5) and (ww/max(1,hh) <= 0.40) and (bw.sum()/255.0/max(1,ww*hh) <= 0.70)

def _to_square_64(bgr):
    h, w = bgr.shape[:2]; s = max(h, w); canvas = np.zeros((s, s, 3), dtype=bgr.dtype)
    y0, x0 = (s-h)//2, (s-w)//2; canvas[y0:y0+h, x0:x0+w] = bgr
    return cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_AREA)

def assemble_int(dets, roi_bgr, max_digits=4):
    chars=[]
    for (x,y,w,h,cls,cf) in dets[:max_digits]:
        ch = ID2CHAR.get(cls,"")
        if ch not in "0123456789":
            ch = "1" if looks_like_one_patch(roi_bgr[int(y):int(y+h), int(x):int(x+w)]) else ""
        if ch: chars.append(ch)
    txt = "".join(chars)
    return txt.lstrip("0") or "0" if txt else ""

def save_debug_boxes(out_dir, name, roi, dets, color=(0,255,0)):
    vis = roi.copy()
    for (x,y,w,h,cls,cf) in dets:
        p1=(int(x),int(y)); p2=(int(x+w),int(y+h)); cv2.rectangle(vis,p1,p2,color,2)
        cv2.putText(vis,f"{ID2CHAR.get(cls,'?')} {cf:.2f}",(p1[0],max(0,p1[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1,cv2.LINE_AA)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(Path(out_dir, f"{name}.png")), vis)

def _pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ===================================================================
# Classe principal
# ===================================================================
class HudProcessor:
    def __init__(self, ckpt_path="weights/best.pt"):
        logging.info("HudProcessor: inicializando...")
        self.device = _pick_device()
        self.fp16 = (self.device == "cuda")
        self.channels_last = (self.device == "cuda")
        torch.backends.cudnn.benchmark = True

        # YOLO
        self.yolo_model = YOLO(ckpt_path)
        try:
            self.yolo_model.fuse()
        except Exception:
            pass

        # Char-CNN
        ck_char = torch.load(CKPT_CHAR, map_location="cpu")
        self.CLASS_NAMES_CHAR = ck_char["class_names"]
        NUM_CLASSES_CHAR = len(self.CLASS_NAMES_CHAR)
        w0 = ck_char["state_dict"]["classifier.0.weight"]
        hidden, in_features = w0.shape[0], w0.shape[1]
        base_char = models.mobilenet_v3_small(weights=None)
        base_char.classifier = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
            nn.Dropout(p=0.2), nn.Linear(hidden, NUM_CLASSES_CHAR),
        )
        self.char_cnn_model = base_char.to(self.device).eval()
        if self.channels_last:
            self.char_cnn_model = self.char_cnn_model.to(memory_format=torch.channels_last)
        self.char_cnn_model.load_state_dict(ck_char["state_dict"])

        self.to_tensor_char = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

        logging.info(f"HudProcessor: device={self.device}, fp16={self.fp16}")

        # Warm-up leve do YOLO (não bloqueia health da API; é no worker)
        try:
            _dummy = np.zeros((64,64,3), dtype=np.uint8)
            self.yolo_model.predict(_dummy, imgsz=256, device=self.device, half=self.fp16, verbose=False)
        except Exception:
            pass

    # ===========================
    # OCR de porcentagens (CPU)
    # ===========================
    def read_percent_tesseract(self, roi_bgr) -> str:
        def _clean(text):
            if not text: return ""
            return re.sub(r'[^\d%]', '', text.strip())
        try:
            all_texts = []
            rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            cfg = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789%'

            try:
                cleaned = _clean(pytesseract.image_to_string(pil, config=cfg))
                if cleaned: all_texts.append(cleaned)
            except Exception as e:
                logging.warning(f"Tesseract original: {e}")

            try:
                pil_inv = ImageOps.invert(pil.convert("L")).convert("RGB")
                cleaned = _clean(pytesseract.image_to_string(pil_inv, config=cfg))
                if cleaned: all_texts.append(cleaned)
            except Exception as e:
                logging.warning(f"Tesseract invertida: {e}")

            try:
                pil_sharp = pil.filter(ImageFilter.SHARPEN)
                cleaned = _clean(pytesseract.image_to_string(pil_sharp, config=cfg))
                if cleaned: all_texts.append(cleaned)
            except Exception as e:
                logging.warning(f"Tesseract sharpen: {e}")

            for text in all_texts:
                m = re.search(r'(\d{1,3})%?', text)
                if m:
                    try:
                        num = int(m.group(1))
                        if 0 <= num <= 100:
                            return str(num)
                    except ValueError:
                        continue
            return ""
        except Exception as e:
            logging.warning(f"Erro geral no Tesseract: {e}")
            return ""

    # ===========================
    # YOLO helpers
    # ===========================
    def yolo_detect_batch(self, images, conf=0.25, iou=0.45, imgsz=256, classes=None, max_det=20):
        """
        images: list[np.ndarray BGR] ou single np.ndarray
        retorna: list[list[(x,y,w,h,cls,conf)]]
        """
        if not isinstance(images, list):
            images = [images]
        results = self.yolo_model.predict(
            images, imgsz=imgsz, conf=conf, iou=iou,
            classes=classes, max_det=max_det,
            device=self.device, half=self.fp16, verbose=False
        )
        out_all = []
        for res in results:
            boxes = []
            if res.boxes is not None:
                for b in res.boxes:
                    x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].detach().cpu().numpy()]
                    boxes.append((x1, y1, x2-x1, y2-y1, int(b.cls.item()), float(b.conf.item())))
                boxes.sort(key=lambda t: t[0])
            out_all.append(boxes)
        return out_all

    @torch.inference_mode()
    def cnn_predict_digit(self, bgr_patch):
        g64 = _to_square_64(bgr_patch)
        x = self.to_tensor_char(g64).unsqueeze(0)
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        x = x.to(self.device, non_blocking=True)
        autocast_ctx = torch.autocast(self.device, dtype=torch.float16) if self.fp16 else nullcontext()
        with autocast_ctx:
            logits = self.char_cnn_model(x)[0].detach().float().cpu().numpy()
        logits = logits - logits.max()
        p = np.exp(logits); p = p / (p.sum() + 1e-9)
        idx = int(np.argmax(p))
        return self.CLASS_NAMES_CHAR[idx], float(p[idx])

    def assemble_dec(self, dets, roi_bgr, args):
        if not dets: return ""
        dets = [d for d in dets if ID2CHAR.get(d[4], "") in "0123456789,"]  # mantém dígitos e vírgula
        if not dets: return ""
        dets.sort(key=lambda t: t[0])
        digits = [d for d in dets if ID2CHAR[d[4]] in "0123456789"]
        commas = [d for d in dets if ID2CHAR[d[4]] == "comma"]

        def _refine_with_cnn(d):
            x,y,w,h,cls,cf = d
            patch = roi_bgr[int(y):int(y+h), int(x):int(x+w)]
            yolo_ch = ID2CHAR.get(int(cls), "")
            if yolo_ch in "0123456789" and cf >= args.lock_yolo: return yolo_ch
            ch_cnn, p = self.cnn_predict_digit(patch)
            if ch_cnn in "0123456789" and p >= args.cnn_min:
                if yolo_ch in "0123456789" and yolo_ch != ch_cnn and cf >= args.keep_yolo: return yolo_ch
                return ch_cnn
            if yolo_ch in "0123456789": return yolo_ch
            return "1" if looks_like_one_patch(patch) else ""

        if commas and digits:
            sep = max(commas, key=lambda t: t[5])
            sx = sep[0] + sep[2]/2.0
            left_candidates  = [d for d in digits if (d[0]+d[2]/2.0) <  sx]
            right_candidates = [d for d in digits if (d[0]+d[2]/2.0) >= sx]
            if left_candidates and right_candidates:
                left  = max(left_candidates,  key=lambda d: d[0]+d[2]/2.0)
                right = min(right_candidates, key=lambda d: d[0]+d[2]/2.0)
                a, b = _refine_with_cnn(left), _refine_with_cnn(right)
                if a in "0123456789" and b in "0123456789": return f"{a},{b}"
            return ""
        if len(digits) >= 2:
            centers = [d[0] + d[2]/2.0 for d in digits]
            best_pair, best_gap = None, 1e9
            for i in range(len(digits)-1):
                gap = centers[i+1] - centers[i]
                if gap < best_gap: best_gap, best_pair = gap, (digits[i], digits[i+1])
            if best_pair:
                a, b = _refine_with_cnn(best_pair[0]), _refine_with_cnn(best_pair[1])
                if a in "0123456789" and b in "0123456789": return f"{a},{b}"
        return ""

    # ===========================
    # Pipeline por frame
    # ===========================
    def process_frame(self, frame, args_dict=None):
        class Args: pass
        args = Args()
        default_args = {
            'conf_int': 0.27, 'conf_dec': 0.05, 'iou': 0.50, 'imgsz': 640,
            'lock_yolo': 0.55, 'keep_yolo': 0.45, 'cnn_min': 0.80, 'viz': False
        }
        if args_dict: default_args.update(args_dict)
        for k, v in default_args.items(): setattr(args, k, v)

        H, W = frame.shape[:2]
        BASE_W, BASE_H = 1920, 1080
        sx, sy = W / BASE_W, H / BASE_H
        def _scale_box(box, sx, sy): return [box[0]*sx, box[1]*sy, box[2]*sx, box[3]*sy]
        rois_base = json.load(open(ROIS_JSON,"r",encoding="utf-8"))
        rois = {k: _scale_box(v, sx, sy) for k, v in rois_base.items()}
        DEC_PAD_X = max(1, int(round(8 * sx)))
        tipos = {k:"int" for k in rois}
        tipos.update({k:"dec" for k in TIPOS_DEC})
        tipos.update({k:"percent" for k in TIPOS_PERCENT})

        results = {}
        # coletores para batch
        int_names, int_rois = [], []
        dec_names, dec_rois_up, dec_scales, dec_rois_orig = [], [], [], []

        for name, box in rois.items():
            t = tipos[name]
            if t == "dec":
                roi_src = crop_xywh(frame, (box[0]-DEC_PAD_X, box[1], box[2]+2*DEC_PAD_X, box[3]), inset=0)
            else:
                roi_src = crop_xywh(frame, box, inset=2)

            if t == "percent":
                val = self.read_percent_tesseract(roi_src)
                results[name] = int(val) if val != "" else ""
            elif t == "int":
                int_names.append(name)
                int_rois.append(roi_src)
            elif t == "dec":
                roi_up, s = resize_with_scale(roi_src, 2.0)
                dec_names.append(name); dec_rois_up.append(roi_up); dec_scales.append(s); dec_rois_orig.append(roi_src)

        # Inteiros em batch
        if int_rois:
            dets_list = self.yolo_detect_batch(
                int_rois, conf=args.conf_int, iou=args.iou,
                imgsz=args.imgsz, classes=list(range(10)), max_det=6
            )
            for name, roi, dets in zip(int_names, int_rois, dets_list):
                dets = filter_int_boxes(dets, roi.shape[0], roi.shape[1], max_digits=4)
                results[name] = assemble_int(dets, roi, max_digits=4)

        # Decimais em batch (dígitos e vírgula separados)
        if dec_rois_up:
            imgsz_dec = max(768, args.imgsz)
            dets_digits_list = self.yolo_detect_batch(
                dec_rois_up, conf=args.conf_int, iou=args.iou,
                imgsz=imgsz_dec, classes=list(range(10)), max_det=6
            )
            dets_comma_list  = self.yolo_detect_batch(
                dec_rois_up, conf=args.conf_dec, iou=args.iou,
                imgsz=imgsz_dec, classes=[10], max_det=3
            )
            for name, roi_up, s, dets_digits, dets_comma, roi_orig in zip(
                dec_names, dec_rois_up, dec_scales, dets_digits_list, dets_comma_list, dec_rois_orig
            ):
                dets = [(d[0]/s, d[1]/s, d[2]/s, d[3]/s, d[4], d[5]) for d in (dets_digits + dets_comma)]
                dets = filter_decimal_boxes(dets, roi_orig.shape[0], roi_orig.shape[1])
                results[name] = self.assemble_dec(dets, roi_orig, args)

        return results

# ===================================================================
# CLI para teste manual
# ===================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",  type=str, required=True, help="caminho da screenshot")
    ap.add_argument("--ckpt", type=str, default="weights/best.pt")
    ap.add_argument("--conf_int", type=float, default=0.27)
    ap.add_argument("--conf_dec", type=float, default=0.05)
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--lock_yolo", type=float, default=0.55)
    ap.add_argument("--keep_yolo", type=float, default=0.45)
    ap.add_argument("--cnn_min",   type=float, default=0.80)
    ap.add_argument("--out_json",  type=str, default="hud_resultados.json")
    args = ap.parse_args()

    print("--- TESTE: PROCESS_FRAME ---")
    processor = HudProcessor(ckpt_path=args.ckpt)

    frame = cv2.imread(args.img)
    if frame is None:
        print(f"Erro: não foi possível ler a imagem em {args.img}")
        raise SystemExit(1)

    results = processor.process_frame(frame, args_dict=vars(args))
    for k, v in results.items():
        print(f"{k:>32}: {v}")

    if args.out_json:
        out = Path(args.out_json)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nOK. Resultados salvos em {out.resolve()}")
