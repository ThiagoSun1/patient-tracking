# =========================
# SILENCE OPENCV + C++ ERRORS
# =========================
import os, sys

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"

class SuppressErrors:
    def __enter__(self):
        self._python_stderr = sys.stderr
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        self._saved_fd2 = os.dup(2)
        os.dup2(self._devnull_fd, 2)
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, *_):
        try:
            sys.stderr.close()
        except:
            pass
        os.dup2(self._saved_fd2, 2)
        os.close(self._saved_fd2)
        os.close(self._devnull_fd)
        sys.stderr = self._python_stderr


# =========================
# IMPORTS
# =========================
import cv2
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit
import requests
import pyttsx3
import time
import threading
from queue import Queue


# =========================
# LOAD MODEL
# =========================
print("Loading YOLO model...")
with SuppressErrors():
    model = YOLO("yolov8n-pose.pt")
print("Model loaded.")


# =========================
# SERVO SETUP
# =========================
kit = ServoKit(channels=16)
PAN_CHANNEL, TILT_CHANNEL = 0, 1
pan_angle, tilt_angle = 90, 90

kit.servo[PAN_CHANNEL].angle = pan_angle
kit.servo[TILT_CHANNEL].angle = tilt_angle


# =========================
# CAMERA
# =========================
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

with SuppressErrors():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


# =========================
# TRACKING SETTINGS
# =========================
SMOOTHING = 0.2
MAX_STEP = 4
DEADZONE = 30
PAN_DIRECTION = 1
TILT_DIRECTION = -1


# =========================
# AI + VOICE (THREADED)
# =========================
engine = pyttsx3.init()

ai_queue = Queue()
voice_queue = Queue()

alert_triggered = False


def ask_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2:0.5b", "prompt": prompt, "stream": False},
            timeout=20
        )
        return response.json()["response"]
    except Exception as e:
        print("Ollama Error:", e)
        return "AI unavailable."


# 🔹 AI THREAD
def ai_worker():
    while True:
        prompt = ai_queue.get()
        if prompt is None:
            break
        reply = ask_ollama(prompt)
        print("AI:", reply)
        voice_queue.put(reply)
        ai_queue.task_done()


# 🔹 VOICE THREAD
def voice_worker():
    while True:
        text = voice_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()


# Start threads
threading.Thread(target=ai_worker, daemon=True).start()
threading.Thread(target=voice_worker, daemon=True).start()


# =========================
# TRACK HEAD
# =========================
def track_head(head_x, head_y):
    global pan_angle, tilt_angle

    error_x = head_x - FRAME_WIDTH // 2
    error_y = head_y - FRAME_HEIGHT // 3

    if abs(error_x) < DEADZONE: error_x = 0
    if abs(error_y) < DEADZONE: error_y = 0

    step_pan = np.clip(PAN_DIRECTION * (-error_x * 0.04), -MAX_STEP, MAX_STEP)
    step_tilt = np.clip(TILT_DIRECTION * (error_y * 0.04), -MAX_STEP, MAX_STEP)

    pan_angle = float(np.clip(pan_angle + SMOOTHING * step_pan, 10, 170))
    tilt_angle = float(np.clip(tilt_angle + SMOOTHING * step_tilt, 10, 170))

    kit.servo[PAN_CHANNEL].angle = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle


# =========================
# BEHAVIOR
# =========================
def classify_behavior(person):
    nose = person[0]
    l_shoulder, r_shoulder = person[5], person[6]
    l_wrist, r_wrist = person[9], person[10]
    l_hip, r_hip = person[11], person[12]

    hip_y = (l_hip[1] + r_hip[1]) / 2

    if abs(nose[1] - hip_y) < 50:
        return "FALL DETECTED"

    if (np.linalg.norm(l_wrist - nose) < 60 or
            np.linalg.norm(r_wrist - nose) < 60):
        return "HEAD PAIN"

    stomach = np.array([(l_hip[0] + r_hip[0]) / 2,
                        (l_hip[1] + r_hip[1]) / 2])
    if (np.linalg.norm(l_wrist - stomach) < 60 or
            np.linalg.norm(r_wrist - stomach) < 60):
        return "STOMACH PAIN"

    back = np.array([(l_shoulder[0] + r_shoulder[0]) / 2,
                     (l_hip[1] + r_hip[1]) / 2])
    if (np.linalg.norm(l_wrist - back) < 60 and
            np.linalg.norm(r_wrist - back) < 60):
        return "BACK PAIN"

    return "OK"


# =========================
# MAIN LOOP
# =========================
print("Running...")

while True:
    with SuppressErrors():
        ret, frame = cap.read()
    if not ret:
        continue

    with SuppressErrors():
        results = model(frame, verbose=False)

    status = "No Person"

    if len(results) > 0 and results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) > 0:
            person = keypoints[0]

            for x, y in person.astype(int):
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            track_head(int(person[0][0]), int(person[0][1]))
            status = classify_behavior(person)

    # 🔥 NON-BLOCKING ALERT
    if status not in ["OK", "No Person"] and not alert_triggered:
        print("ALERT:", status)
        prompt = "A camera detected a person. Calmly ask if they need help."
        ai_queue.put(prompt)  # ← goes to thread (no lag!)
        alert_triggered = True

    if status in ["OK", "No Person"]:
        alert_triggered = False

    cv2.putText(frame, status, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
