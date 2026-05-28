# -*- coding: utf-8 -*-
"""
test_models.py
==============
Comprehensive test for the MobiKD 2-stage pipeline.

Tests:
  1. Stage 1 - Leaf Validator   (leaf vs not_leaf)
  2. Stage 2 - Potato Disease   (early_blight / healthy / late_blight / not_potato_leaf)
  3. End-to-End 2-Stage Pipeline

Run from MobiKD root:
    py test_models.py
"""

import os, sys, time, random
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
from pathlib import Path
from PIL import Image

# -- Colour codes --------------------------------------------------------------
R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
B  = "\033[94m"
M  = "\033[95m"
C  = "\033[96m"
W  = "\033[97m"
DIM= "\033[2m"
RST= "\033[0m"
BOLD="\033[1m"

def banner(title, color=B):
    w = 66
    print(f"\n{color}{'='*w}{RST}")
    print(f"{color}  {BOLD}{title}{RST}")
    print(f"{color}{'='*w}{RST}")

def section(title):
    pad = '-' * (60 - len(title))
    print(f"\n{C}-- {title} {pad}{RST}")

def ok(msg):   print(f"  {G}[OK]{RST}  {msg}")
def fail(msg): print(f"  {R}[FAIL]{RST}  {msg}")
def warn(msg): print(f"  {Y}[WARN]{RST}  {msg}")
def info(msg): print(f"  {DIM}{msg}{RST}")

# -- Paths ---------------------------------------------------------------------
ROOT         = Path(__file__).parent
MODELS_DIR   = ROOT / "models"
LEAF_MODEL   = MODELS_DIR / "leaf_validator"  / "kd_mobilenetv2_leaf_validator_float16.tflite"
DISEASE_MODEL= MODELS_DIR / "potato_disease"  / "kd_mobilenetv2_potato_disease_float16.tflite"
DATA_LEAF    = ROOT / "data" / "lead and not leaf"
DATA_POTATO  = ROOT / "data" / "potato"

LEAF_CLASSES    = ["not_leaf", "leaf"]
DISEASE_CLASSES = ["early_blight", "healthy", "late_blight", "not_potato_leaf"]
LEAF_SIZE       = 32
DISEASE_SIZE    = 96
LEAF_THRESHOLD  = 0.70
N_SAMPLES       = 20   # images per class to sample

# -- Import TFLite -------------------------------------------------------------
def load_tflite():
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except ImportError:
        pass
    print(f"{R}ERROR: Neither tflite_runtime nor tensorflow found.{RST}")
    print("  Install with:  pip install tflite-runtime")
    sys.exit(1)

# -- Preprocessing -------------------------------------------------------------
def preprocess(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    # centre crop
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)   # (1, size, size, 3)

# -- Single-model inference ----------------------------------------------------
def run_model(interpreter, inp_details, out_details, data):
    interpreter.set_tensor(inp_details[0]["index"], data)
    interpreter.invoke()
    return interpreter.get_tensor(out_details[0]["index"])[0]

# -- Sample images from a folder -----------------------------------------------
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def sample_images(folder: Path, n: int) -> list[Path]:
    if not folder.exists():
        return []
    files = [f for f in folder.iterdir() if f.suffix.lower() in EXTS]
    random.shuffle(files)
    return files[:n]

# -----------------------------------------------------------------------------
# 1. Environment check
# -----------------------------------------------------------------------------
banner("MobiKD — Model Test Suite", B)

section("Environment Check")
Interpreter = load_tflite()
ok("TFLite runtime loaded")

for label, path in [("Leaf Validator model", LEAF_MODEL),
                    ("Disease Detector model", DISEASE_MODEL)]:
    if path.exists():
        ok(f"{label}  ({path.stat().st_size/1e6:.2f} MB)")
    else:
        fail(f"{label} NOT found at: {path}")
        sys.exit(1)

for label, path in [("Leaf dataset", DATA_LEAF),
                    ("Potato dataset", DATA_POTATO)]:
    if path.exists():
        ok(f"{label} found at: {path}")
    else:
        warn(f"{label} not found — some tests will be skipped")

# -----------------------------------------------------------------------------
# 2. Load models
# -----------------------------------------------------------------------------
section("Loading Models")

t0 = time.time()
leaf_interp = Interpreter(model_path=str(LEAF_MODEL))
leaf_interp.allocate_tensors()
leaf_inp = leaf_interp.get_input_details()
leaf_out = leaf_interp.get_output_details()
info(f"Leaf Validator  — input: {leaf_inp[0]['shape']}  output: {leaf_out[0]['shape']}")
ok(f"Leaf Validator loaded in {(time.time()-t0)*1000:.0f} ms")

t0 = time.time()
dis_interp = Interpreter(model_path=str(DISEASE_MODEL))
dis_interp.allocate_tensors()
dis_inp = dis_interp.get_input_details()
dis_out = dis_interp.get_output_details()
info(f"Disease Detector — input: {dis_inp[0]['shape']}  output: {dis_out[0]['shape']}")
ok(f"Disease Detector loaded in {(time.time()-t0)*1000:.0f} ms")

# -----------------------------------------------------------------------------
# 3. Stage 1 — Leaf Validator test
# -----------------------------------------------------------------------------
banner("Stage 1 — Leaf Validator", G)

def test_leaf_validator(class_name, folder, expected_idx, n=N_SAMPLES):
    images = sample_images(folder, n)
    if not images:
        warn(f"No images found in {folder}")
        return 0, 0, 0.0

    correct = 0
    confidences = []
    latencies   = []
    for img_path in images:
        try:
            data  = preprocess(img_path, LEAF_SIZE)
            t0    = time.time()
            probs = run_model(leaf_interp, leaf_inp, leaf_out, data)
            latencies.append((time.time()-t0)*1000)
            pred_idx = int(np.argmax(probs))
            conf     = float(probs[pred_idx])
            confidences.append(conf)
            if pred_idx == expected_idx:
                correct += 1
        except Exception as e:
            warn(f"  Error on {img_path.name}: {e}")

    acc      = correct / len(images) * 100
    avg_conf = np.mean(confidences) * 100
    avg_lat  = np.mean(latencies)
    color    = G if acc >= 85 else Y if acc >= 70 else R
    print(f"\n  {BOLD}Class: {class_name}{RST}  ({len(images)} images)")
    print(f"    Accuracy   : {color}{acc:.1f}%{RST}  ({correct}/{len(images)})")
    print(f"    Avg Conf   : {avg_conf:.1f}%")
    print(f"    Avg Latency: {avg_lat:.1f} ms / image")
    return correct, len(images), acc

leaf_results = {}
if (DATA_LEAF / "leaf").exists():
    c, t, a = test_leaf_validator("leaf", DATA_LEAF / "leaf", expected_idx=1)
    leaf_results["leaf"] = (c, t, a)
if (DATA_LEAF / "non_leaf").exists():
    c, t, a = test_leaf_validator("not_leaf", DATA_LEAF / "non_leaf", expected_idx=0)
    leaf_results["not_leaf"] = (c, t, a)

if leaf_results:
    total_c = sum(v[0] for v in leaf_results.values())
    total_t = sum(v[1] for v in leaf_results.values())
    overall = total_c / total_t * 100 if total_t > 0 else 0
    color   = G if overall >= 90 else Y if overall >= 75 else R
    section("Leaf Validator Summary")
    print(f"  Overall Accuracy: {color}{BOLD}{overall:.1f}%{RST}  ({total_c}/{total_t})")

# -----------------------------------------------------------------------------
# 4. Stage 2 — Potato Disease test
# -----------------------------------------------------------------------------
banner("Stage 2 — Potato Disease Detector", M)

def test_disease_detector(class_name, folder, expected_idx, n=N_SAMPLES):
    images = sample_images(folder, n)
    if not images:
        warn(f"No images found in {folder}")
        return 0, 0, 0.0

    correct     = 0
    confidences = []
    latencies   = []
    per_class   = {c: 0 for c in DISEASE_CLASSES}

    for img_path in images:
        try:
            data  = preprocess(img_path, DISEASE_SIZE)
            t0    = time.time()
            probs = run_model(dis_interp, dis_inp, dis_out, data)
            latencies.append((time.time()-t0)*1000)
            pred_idx  = int(np.argmax(probs))
            conf      = float(probs[pred_idx])
            confidences.append(conf)
            per_class[DISEASE_CLASSES[pred_idx]] += 1
            if pred_idx == expected_idx:
                correct += 1
        except Exception as e:
            warn(f"  Error on {img_path.name}: {e}")

    acc      = correct / len(images) * 100
    avg_conf = np.mean(confidences) * 100
    avg_lat  = np.mean(latencies)
    color    = G if acc >= 80 else Y if acc >= 60 else R

    print(f"\n  {BOLD}Class: {class_name}{RST}  (expected: {DISEASE_CLASSES[expected_idx]})  [{len(images)} images]")
    print(f"    Accuracy    : {color}{acc:.1f}%{RST}  ({correct}/{len(images)})")
    print(f"    Avg Conf    : {avg_conf:.1f}%")
    print(f"    Avg Latency : {avg_lat:.1f} ms / image")
    print(f"    Predictions : ", end="")
    for cls, cnt in per_class.items():
        if cnt > 0:
            mark = G if cls == DISEASE_CLASSES[expected_idx] else R
            print(f"{mark}{cls}={cnt}{RST} ", end="")
    print()
    return correct, len(images), acc

dis_map = {
    "early_blight"   : (DATA_POTATO / "early_blight",    0),
    "healthy"        : (DATA_POTATO / "healthy",          1),
    "late_blight"    : (DATA_POTATO / "late_blight",      2),
    "not_potato_leaf": (DATA_POTATO / "not_potato_leaf",  3),
}

dis_results = {}
for cls_name, (folder, exp_idx) in dis_map.items():
    if folder.exists():
        c, t, a = test_disease_detector(cls_name, folder, exp_idx)
        dis_results[cls_name] = (c, t, a)

if dis_results:
    total_c = sum(v[0] for v in dis_results.values())
    total_t = sum(v[1] for v in dis_results.values())
    overall = total_c / total_t * 100 if total_t > 0 else 0
    color   = G if overall >= 85 else Y if overall >= 70 else R
    section("Disease Detector Summary")
    print(f"  Overall Accuracy: {color}{BOLD}{overall:.1f}%{RST}  ({total_c}/{total_t})")

# -----------------------------------------------------------------------------
# 5. End-to-End Pipeline test
# -----------------------------------------------------------------------------
banner("End-to-End 2-Stage Pipeline", C)

def run_pipeline(img_path: Path):
    # Stage 1
    data1  = preprocess(img_path, LEAF_SIZE)
    t0     = time.time()
    probs1 = run_model(leaf_interp, leaf_inp, leaf_out, data1)
    t1     = (time.time()-t0)*1000
    idx1   = int(np.argmax(probs1))
    conf1  = float(probs1[idx1])
    label1 = LEAF_CLASSES[idx1]

    if label1 == "not_leaf" or conf1 < LEAF_THRESHOLD:
        return {
            "stage1": label1, "conf1": conf1,
            "stage2": None,   "conf2": None,
            "verdict": "REJECTED — Not a leaf",
            "lat_ms": t1, "passed_gate": False,
        }

    # Stage 2
    data2  = preprocess(img_path, DISEASE_SIZE)
    t0     = time.time()
    probs2 = run_model(dis_interp, dis_inp, dis_out, data2)
    t2     = (time.time()-t0)*1000
    idx2   = int(np.argmax(probs2))
    conf2  = float(probs2[idx2])
    label2 = DISEASE_CLASSES[idx2]

    verdicts = {
        "early_blight"   : "DISEASED — Early Blight",
        "late_blight"    : "DISEASED — Late Blight",
        "healthy"        : "HEALTHY",
        "not_potato_leaf": "REJECTED — Not a Potato Leaf",
    }
    return {
        "stage1": label1, "conf1": conf1,
        "stage2": label2, "conf2": conf2,
        "verdict": verdicts.get(label2, label2),
        "lat_ms": t1+t2,  "passed_gate": True,
    }

# Pick one image from each class for end-to-end demo
e2e_tests = [
    ("Potato leaf — Early Blight", DATA_POTATO / "early_blight"),
    ("Potato leaf — Healthy",      DATA_POTATO / "healthy"),
    ("Potato leaf — Late Blight",  DATA_POTATO / "late_blight"),
    ("Non-potato leaf",            DATA_POTATO / "not_potato_leaf"),
    ("Non-leaf image",             DATA_LEAF   / "non_leaf"),
]

print()
all_e2e_ok = True
for desc, folder in e2e_tests:
    imgs = sample_images(folder, 3)
    if not imgs:
        warn(f"Skipping '{desc}' — no images")
        continue

    for img_path in imgs[:1]:   # one per class
        r = run_pipeline(img_path)
        s1c = f"{r['conf1']*100:.0f}%"
        s2c = f"{r['conf2']*100:.0f}%" if r['conf2'] else "—"
        lat = f"{r['lat_ms']:.0f} ms"

        # colour verdict
        v = r["verdict"]
        if "HEALTHY" in v:
            vcolor = G
        elif "DISEASED" in v:
            vcolor = R
        elif "REJECTED" in v:
            vcolor = Y
        else:
            vcolor = W

        print(f"  {BOLD}{desc}{RST}")
        print(f"    File    : {DIM}{img_path.name[:60]}{RST}")
        print(f"    Stage 1 : {r['stage1']}  ({s1c})")
        if r["stage2"]:
            print(f"    Stage 2 : {r['stage2']}  ({s2c})")
        print(f"    Verdict : {vcolor}{BOLD}{v}{RST}")
        print(f"    Latency : {lat}")
        print()

# -----------------------------------------------------------------------------
# 6. Robustness spot check (blur, noise, low-light)
# -----------------------------------------------------------------------------
banner("Robustness Spot Check", Y)

import cv2

def apply_blur(arr):
    u = (arr * 255).astype(np.uint8)
    u = cv2.GaussianBlur(u, (9, 9), 0)
    return u.astype(np.float32) / 255.0

def apply_noise(arr, sigma=0.12):
    n = arr + np.random.normal(0, sigma, arr.shape).astype(np.float32)
    return np.clip(n, 0, 1)

def apply_low_light(arr, factor=0.3):
    return np.clip(arr * factor, 0, 1)

augments = [
    ("Clean",      lambda x: x),
    ("Blur",       apply_blur),
    ("Noise",      apply_noise),
    ("Low Light",  apply_low_light),
]

# Use 5 healthy potato images for robustness check
healthy_imgs = sample_images(DATA_POTATO / "healthy", 5) if (DATA_POTATO / "healthy").exists() else []

if healthy_imgs:
    section("Healthy potato leaf under degradations (Disease Detector)")
    for aug_name, aug_fn in augments:
        correct = 0
        for img_path in healthy_imgs:
            try:
                arr  = preprocess(img_path, DISEASE_SIZE)[0]   # (96,96,3)
                arr  = aug_fn(arr)
                data = np.expand_dims(arr, 0)
                probs= run_model(dis_interp, dis_inp, dis_out, data)
                if int(np.argmax(probs)) == 1:   # healthy = index 1
                    correct += 1
            except Exception:
                pass
        acc   = correct / len(healthy_imgs) * 100
        color = G if acc >= 75 else Y if acc >= 55 else R
        bar   = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
        print(f"    {aug_name:<12}  {color}{bar}  {acc:.0f}%{RST}  ({correct}/{len(healthy_imgs)})")
else:
    warn("No healthy images found for robustness check")

# -----------------------------------------------------------------------------
# 7. Final report
# -----------------------------------------------------------------------------
banner("Final Report", B)

# Leaf validator
if leaf_results:
    lv_total = sum(v[0] for v in leaf_results.values())
    lv_n     = sum(v[1] for v in leaf_results.values())
    lv_acc   = lv_total / lv_n * 100 if lv_n else 0
    lv_color = G if lv_acc >= 90 else Y if lv_acc >= 75 else R
    print(f"  Leaf Validator    : {lv_color}{BOLD}{lv_acc:.1f}%{RST}  ({lv_total}/{lv_n})", end="")
    print(f"  {'[OK] PASS' if lv_acc >= 85 else '[FAIL] FAIL'}")

# Disease detector
if dis_results:
    dv_total = sum(v[0] for v in dis_results.values())
    dv_n     = sum(v[1] for v in dis_results.values())
    dv_acc   = dv_total / dv_n * 100 if dv_n else 0
    dv_color = G if dv_acc >= 80 else Y if lv_acc >= 65 else R
    print(f"  Disease Detector  : {dv_color}{BOLD}{dv_acc:.1f}%{RST}  ({dv_total}/{dv_n})", end="")
    print(f"  {'[OK] PASS' if dv_acc >= 75 else '[FAIL] FAIL'}")

print(f"\n  {DIM}Models tested: Float16 TFLite (edge-ready){RST}")
print(f"  {DIM}Inference:     CPU (no GPU required){RST}")
print(f"\n{B}{'='*66}{RST}\n")
