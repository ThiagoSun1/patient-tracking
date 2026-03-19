import cv2
import numpy as np
from ultralytics import YOLO
from adafruit_servokit import ServoKit
import requests
import pyttsx3
import time

# ----------------------------
# LOAD YOLOv8 POSE MODEL
# ----------------------------
model = YOLO("yolov8n-pose.pt")

# ----------------------------
# SERVO SETUP (Adafruit PCA9685)
# ----------------------------
kit = ServoKit(channels=16)

PAN_CHANNEL = 0
TILT_CHANNEL = 1

pan_angle = 90
tilt_angle = 90

kit.servo[PAN_CHANNEL].angle = pan_angle
kit.servo[TILT_CHANNEL].angle = tilt_angle

# ----------------------------
# CAMERA SETUP
# ----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
# ----------------------------
# TRACKING SETTINGS
# ----------------------------
SMOOTHING = 0.2
MAX_STEP = 4
DEADZONE = 30

PAN_DIRECTION = 1
TILT_DIRECTION = -1  # flip if tilt moves wrong way

# ----------------------------
# AI + VOICE SETUP
# ----------------------------
engine = pyttsx3.init()
alert_triggered = False

def ask_ollama(message):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2:0.5b",
                "prompt": message,
                "stream": False
            },
            timeout=20
        )
        return response.json()["response"]
    except Exception as e:
        print("Ollama Error:", e)
        return "AI unavailable."

# ----------------------------
# SERVO TRACKING FUNCTION
# ----------------------------
def track_head(head_x, head_y):
    global pan_angle, tilt_angle

    center_x = FRAME_WIDTH // 2
    center_y = FRAME_HEIGHT // 3

    error_x = head_x - center_x
    error_y = head_y - center_y

    if abs(error_x) < DEADZONE:
        error_x = 0
    if abs(error_y) < DEADZONE:
        error_y = 0

    step_pan = PAN_DIRECTION * (-error_x * 0.04)
    step_tilt = TILT_DIRECTION * (error_y * 0.04)

    step_pan = np.clip(step_pan, -MAX_STEP, MAX_STEP)
    step_tilt = np.clip(step_tilt, -MAX_STEP, MAX_STEP)

    pan_angle += SMOOTHING * step_pan
    tilt_angle += SMOOTHING * step_tilt

    pan_angle = np.clip(pan_angle, 10, 170)
    tilt_angle = np.clip(tilt_angle, 10, 170)

    kit.servo[PAN_CHANNEL].angle = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle

# ----------------------------
# BEHAVIOR CLASSIFICATION
# ----------------------------
def classify_behavior(person):

    nose = person[0]
    l_shoulder = person[5]
    r_shoulder = person[6]
    l_wrist = person[9]
    r_wrist = person[10]
    l_hip = person[11]
    r_hip = person[12]

    # FALL DETECTION (body horizontal)
    hip_y = (l_hip[1] + r_hip[1]) / 2
    if abs(nose[1] - hip_y) < 50:
        return "FALL DETECTED"

    # HEAD PAIN (hand near head)
    if (np.linalg.norm(l_wrist - nose) < 60 or
        np.linalg.norm(r_wrist - nose) < 60):
        return "HEAD PAIN"

    # STOMACH PAIN (hand near hips center)
    stomach_center = np.array([
        (l_hip[0] + r_hip[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    ])

    if (np.linalg.norm(l_wrist - stomach_center) < 60 or
        np.linalg.norm(r_wrist - stomach_center) < 60):
        return "STOMACH PAIN"

    # BACK PAIN (both hands near spine area)
    back_center = np.array([
        (l_shoulder[0] + r_shoulder[0]) / 2,
        (l_hip[1] + r_hip[1]) / 2
    ])

    if (np.linalg.norm(l_wrist - back_center) < 60 and
        np.linalg.norm(r_wrist - back_center) < 60):
        return "BACK PAIN"

    return "OK"

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    status = "No Person"

    if len(results) > 0 and results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy.cpu().numpy()

        if len(keypoints) > 0:

            person = keypoints[0]

            # Draw keypoints
            for x, y in person.astype(int):
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Draw bounding box
            x_min = int(np.min(person[:, 0]))
            y_min = int(np.min(person[:, 1]))
            x_max = int(np.max(person[:, 0]))
            y_max = int(np.max(person[:, 1]))

            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (255, 0, 0), 2)

            # Head tracking
            head_x = int(person[0][0])
            head_y = int(person[0][1])
            track_head(head_x, head_y)

            # Behavior detection
            status = classify_behavior(person)

    # ----------------------------
    # AI TRIGGER
    # ----------------------------
    if status not in ["OK", "No Person"] and not alert_triggered:

        name = "Thiago"
        disease = "stomach ache"
        age = "10"
        print("ALERT:", name, "has", status)

        prompt = f"Use {name} throughout the conversation so they know its them. A camera detected a person named {name} has {status}. Calmly ask if {name} need help. For information , {name} is {age} years old, and has {disease}, just talk to him based on the information I just gave you."

        reply = ask_ollama(prompt)
        print("AI:", reply)

        engine.say(reply)
        engine.runAndWait()

        alert_triggered = True

    if status in ["OK", "No Person"]:
        alert_triggered = False

    # Display status
    cv2.putText(frame, status, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 3)

    cv2.imshow("Tracking + Behavior Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()
