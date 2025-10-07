#!/usr/bin/env python3
import os
import time
import queue
import threading
import numpy as np
import cv2
import csv

USE_LAPTOP_CAMERA = True

# Try TFLite runtimes (prefer tflite_runtime)
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter

# Dummy gpiozero-like classes for laptop testing
if USE_LAPTOP_CAMERA:
    class PWMOutputDevice:
        def __init__(self, pin, frequency=1000):
            self.pin = pin
            self.frequency = frequency
            self.value = 0.0
    class DigitalOutputDevice:
        def __init__(self, pin):
            self.pin = pin
        def on(self): pass
        def off(self): pass

# ================= CONFIG =================
MOVEMENT_MODEL = "tflite_model/model.tflite"
MOVEMENT_LABELS = "tflite_model/labels.txt"
FISH_MODEL = "tflite_model/fish_model.tflite"
FISH_LABELS = "tflite_model/fish_labels.txt"

NUM_THREADS = 2
SSD_SCORE_THRESHOLD = 0.45
FISH_SCORE_THRESHOLD = 0.95     # only count when classifier >= 95%
CLASSIFY_EVERY_N_FRAMES = 2

ROTATE_FRAME = False

# motor pins (kept for later Pi)
L_IN1, L_IN2, L_EN = 17, 27, 18
R_IN1, R_IN2, R_EN = 22, 23, 13

# speeds used for dummy motor prints (0..1)
FORWARD_SPEED = 0.6
TURN_SPEED = 0.6

TRACK_DIST_PX = 60
TRACK_EXPIRE_S = 1.2
LOG_CSV = "fish_count_log.csv"

# Keywords used to decide whether a classifier label is a fish.
FISH_LABEL_KEYWORDS = [
    "fish", "trout", "salmon", "carp", "tilapia", "bass", "mackerel",
    "cod", "perch", "herring", "tuna", "shark", "pike", "catfish",
    "goldfish", "guppy", "anchovy", "sardine", "halibut", "flounder",
    "bream", "sole", "seabass", "snapper", "grouper", "clownfish"
]

# ================= Motors (dummy on laptop) =================
class MotorDummy:
    def __init__(self, name):
        self.name = name
        self.speed = 0.0
    def set_speed(self, s):
        s = float(max(-1.0, min(1.0, s)))
        self.speed = s
        # Print summary per call — kept lightweight
        print(f"[MOTOR] {self.name} speed {s:.2f}")
    def set_stop(self):
        self.speed = 0.0
        print(f"[MOTOR] {self.name} STOP")

left_motor = MotorDummy("LEFT")
right_motor = MotorDummy("RIGHT")

# ================= Labels loader =================
def read_labels(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

if not os.path.exists(MOVEMENT_MODEL) or not os.path.exists(MOVEMENT_LABELS):
    raise FileNotFoundError("Place movement model and labels in tflite_model/ (model.tflite and labels.txt)")

movement_labels = read_labels(MOVEMENT_LABELS)
use_fish_classifier = os.path.exists(FISH_MODEL) and os.path.exists(FISH_LABELS)
fish_labels = read_labels(FISH_LABELS) if use_fish_classifier else []

if use_fish_classifier:
    print("Fish classifier found. Fish counting enabled.")
else:
    print("Fish classifier NOT found. Fish counting disabled.")

# ================= Interpreters =================
movement_interpreter = Interpreter(MOVEMENT_MODEL, num_threads=NUM_THREADS)
movement_interpreter.allocate_tensors()
mov_in = movement_interpreter.get_input_details()[0]
mov_out = movement_interpreter.get_output_details()
try:
    MOV_H = int(mov_in["shape"][1]); MOV_W = int(mov_in["shape"][2])
except Exception:
    MOV_H, MOV_W = 320, 320

fish_interpreter = None
if use_fish_classifier:
    fish_interpreter = Interpreter(FISH_MODEL, num_threads=NUM_THREADS)
    fish_interpreter.allocate_tensors()
    fish_in = fish_interpreter.get_input_details()[0]
    fish_out = fish_interpreter.get_output_details()
    try:
        FISH_H = int(fish_in["shape"][1]); FISH_W = int(fish_in["shape"][2])
    except Exception:
        FISH_H, FISH_W = 224, 224

# ================= Helper utilities =================
def _get_detection_tensors(interpreter, output_details):
    outs = []
    for o in output_details:
        try:
            outs.append(interpreter.get_tensor(o["index"]))
        except Exception:
            outs.append(None)
    boxes = None; classes = None; scores = None; num = None
    for o in outs:
        if o is None: continue
        if isinstance(o, np.ndarray):
            if o.ndim == 3 and o.shape[2] == 4:
                boxes = o[0]; continue
            if o.ndim == 2 and o.shape[1] == 4:
                boxes = o; continue
    for o in outs:
        if o is None: continue
        if isinstance(o, np.ndarray):
            if scores is None and o.dtype == np.float32:
                if o.ndim == 2 and o.shape[0] == 1 and o.shape[1] <= 200:
                    if o.max() <= 1.0:
                        scores = o[0]; continue
                if o.ndim == 1 and o.max() <= 1.0:
                    scores = o; continue
            if classes is None and (o.dtype == np.int32 or o.dtype == np.uint8 or o.dtype == np.float32):
                if o.ndim == 2 and o.shape[0] == 1 and o.shape[1] <= 200:
                    maybe = o[0]
                    if maybe.max() > 1:
                        classes = maybe.astype(np.int32); continue
                if o.ndim == 1 and o.size <= 200:
                    classes = o.astype(np.int32); continue
            if num is None and o.size == 1:
                try:
                    num = int(o.ravel()[0])
                except Exception:
                    pass
    if boxes is None:
        boxes = np.zeros((0,4), dtype=np.float32)
    else:
        boxes = np.asarray(boxes, dtype=np.float32)
    if scores is None:
        scores = np.zeros((boxes.shape[0],), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32).ravel()
    if classes is None:
        classes = np.zeros((scores.shape[0],), dtype=np.int32)
    else:
        classes = np.asarray(classes, dtype=np.int32).ravel()
    if num is None:
        num = int(scores.shape[0]) if scores.size > 0 else 0
    return boxes, classes, scores, num

def prepare_movement_input(frame):
    img = cv2.resize(frame, (MOV_W, MOV_H))
    if mov_in["dtype"] == np.uint8:
        return np.expand_dims(img.astype(np.uint8), axis=0)
    else:
        arr = img.astype(np.float32)
        arr = (arr - 127.5) / 127.5
        return np.expand_dims(arr, axis=0)

def prepare_fish_input(crop):
    img = cv2.resize(crop, (FISH_W, FISH_H))
    if fish_in["dtype"] == np.uint8:
        return np.expand_dims(img.astype(np.uint8), axis=0)
    else:
        arr = img.astype(np.float32)
        arr = (arr - 127.5) / 127.5
        return np.expand_dims(arr, axis=0)

def is_label_a_fish(label):
    if not label:
        return False
    low = label.lower()
    for kw in FISH_LABEL_KEYWORDS:
        if kw.lower() in low:
            return True
    return False

# ============== Camera setup ==============
if USE_LAPTOP_CAMERA:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
else:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    cam_conf = picam2.create_preview_configuration({"format":"BGR888","size":(640,480),"preserve_ar":True})
    picam2.configure(cam_conf)
    picam2.start()
    time.sleep(0.4)

# ============== Tracking & counting ==============
tracks = {}
next_tid = 1
total_count = 0
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp","event","tid","total","note"])

def create_track(cx, cy):
    global next_tid, tracks, total_count
    tid = next_tid; next_tid += 1
    tracks[tid] = {'centroid':(cx,cy), 'last_seen':time.time()}
    total_count += 1
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([ts, "counted", tid, total_count, "classifier"])
    print(f"🔢 New fish counted: total={total_count} (id={tid})")

def update_or_create(cx, cy):
    best = None; best_d = 1e9
    for tid, info in tracks.items():
        x,y = info['centroid']; d = ((x-cx)**2 + (y-cy)**2)**0.5
        if d < best_d:
            best_d = d; best = tid
    if best is not None and best_d <= TRACK_DIST_PX:
        tracks[best]['centroid'] = (cx,cy)
        tracks[best]['last_seen'] = time.time()
        return False
    else:
        create_track(cx, cy); return True

def expire_tracks():
    now = time.time()
    todel = []
    for tid, info in tracks.items():
        if now - info['last_seen'] > TRACK_EXPIRE_S:
            todel.append(tid)
    for tid in todel:
        del tracks[tid]

# ============== Main loop ==============
frame_idx = 0
last_action = None
last_print_time = 0.0
PRINT_COOLDOWN = 0.5

print("Starting. Press 'q' in the window to quit.")

try:
    while True:
        if USE_LAPTOP_CAMERA:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
        else:
            frame = picam2.capture_array()

        if ROTATE_FRAME:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        h, w = frame.shape[:2]

        # Movement detection
        mov_input = prepare_movement_input(frame)
        movement_interpreter.set_tensor(mov_in["index"], mov_input)
        movement_interpreter.invoke()
        boxes, classes, scores, num = _get_detection_tensors(movement_interpreter, mov_out)

        # Normalize boxes if they appear to be in pixel coords
        if boxes.size > 0:
            max_val = boxes.max()
            if max_val > 1.5:  # likely pixel coords (not normalized)
                boxes_norm = boxes.copy()
                # boxes format assumed ymin,xmin,ymax,xmax (pixel coords)
                boxes_norm[:, 0] = boxes[:, 0] / float(h)
                boxes_norm[:, 2] = boxes[:, 2] / float(h)
                boxes_norm[:, 1] = boxes[:, 1] / float(w)
                boxes_norm[:, 3] = boxes[:, 3] / float(w)
                boxes = boxes_norm

        # choose best detection for movement
        best_idx = -1; best_score = 0.0
        for i, s in enumerate(scores):
            if s >= SSD_SCORE_THRESHOLD and s > best_score:
                best_score = float(s); best_idx = i

        action = "FORWARD"
        left_cmd = FORWARD_SPEED
        right_cmd = FORWARD_SPEED

        if best_idx >= 0 and best_idx < boxes.shape[0]:
            ymin, xmin, ymax, xmax = boxes[best_idx]
            # center x in normalized coords
            cx_norm = (xmin + xmax) / 2.0
            # robust clamp
            cx_norm = max(0.0, min(1.0, float(cx_norm)))

            if cx_norm >= 0.5:
                action = "LEFT"
                left_cmd = 0.0
                right_cmd = TURN_SPEED
            else:
                action = "RIGHT"
                left_cmd = TURN_SPEED
                right_cmd = 0.0

            # draw box in frame (convert to pixel coords)
            x1 = int(xmin * w); y1 = int(ymin * h); x2 = int(xmax * w); y2 = int(ymax * h)
            cls = int(classes[best_idx]) if best_idx < len(classes) else -1
            label = movement_labels[cls] if (0 <= cls < len(movement_labels)) else str(cls)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {best_score:.2f}", (x1, max(12,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Apply motors (dummy prints)
        left_motor.set_speed(left_cmd)
        right_motor.set_speed(right_cmd)

        now = time.time()
        if action != last_action or (now - last_print_time) >= PRINT_COOLDOWN:
            last_print_time = now
            last_action = action
            if action == "FORWARD":
                print(f"[ACTION] FORWARD  (L={left_cmd:.2f}, R={right_cmd:.2f})  total_fish={total_count}")
            elif action == "LEFT":
                print(f"[ACTION] LEFT (object in RIGHT sector)  total_fish={total_count}")
            elif action == "RIGHT":
                print(f"[ACTION] RIGHT (object in LEFT sector)  total_fish={total_count}")
            else:
                print(f"[ACTION] {action}  total_fish={total_count}")

        # Fish classification & counting (run occasionally)
        if use_fish_classifier and (frame_idx % CLASSIFY_EVERY_N_FRAMES == 0):
            for i, s in enumerate(scores):
                if s < SSD_SCORE_THRESHOLD:
                    continue
                ymin, xmin, ymax, xmax = boxes[i]
                x1 = int(xmin * w); y1 = int(ymin * h); x2 = int(xmax * w); y2 = int(ymax * h)
                x1c = max(0, x1); y1c = max(0, y1); x2c = min(w-1, x2); y2c = min(h-1, y2)
                if x2c - x1c <= 4 or y2c - y1c <= 4:
                    continue
                crop = frame[y1c:y2c, x1c:x2c]
                try:
                    inp = prepare_fish_input(crop)
                    fish_interpreter.set_tensor(fish_in["index"], inp)
                    fish_interpreter.invoke()
                    out = fish_interpreter.get_tensor(fish_out[0]["index"])
                    out = np.squeeze(out)
                    preds = out.ravel() if out.ndim != 1 else out
                    fid = int(np.argmax(preds)); fscore = float(np.max(preds))
                    flabel = fish_labels[fid] if fid < len(fish_labels) else str(fid)

                    # ONLY count if classifier thinks it's a fish-like label (keyword match)
                    if fscore >= FISH_SCORE_THRESHOLD and is_label_a_fish(flabel):
                        cx_px = int((x1c + x2c) / 2.0); cy_px = int((y1c + y2c) / 2.0)
                        created = update_or_create(cx_px, cy_px)
                        if created:
                            print(f"[COUNT] Detected fish '{flabel}' score={fscore:.3f} -> total={total_count}")
                        cv2.rectangle(frame, (x1c,y1c), (x2c,y2c), (0,165,255), 2)
                        cv2.putText(frame, f"{flabel} {fscore:.2f}", (x1c, max(12,y1c-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 2)
                except Exception as e:
                    print("Fish classifier error:", e)

        expire_tracks()

        # overlays
        cv2.putText(frame, f"Fish Count: {total_count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        for tid, info in tracks.items():
            cx, cy = info['centroid']
            cv2.putText(frame, f"ID{tid}", (int(cx)-20, int(cy)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            cv2.circle(frame, (int(cx), int(cy)), 3, (255,0,0), -1)

        # central divider
        cv2.line(frame, (w//2,0), (w//2,h), (200,200,200), 1)

        cv2.imshow("Swish Debug", frame)

        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    try:
        if USE_LAPTOP_CAMERA:
            cap.release()
        else:
            picam2.stop()
    except Exception:
        pass
    left_motor.set_stop(); right_motor.set_stop()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
