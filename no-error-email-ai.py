# =========================
# SILENCE OPENCV + C++ ERRORS (OS-level fd redirect)
# =========================
import os
import sys

os.environ["OPENCV_LOG_LEVEL"]     = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
os.environ["PYTHONWARNINGS"]       = "ignore"


class SuppressErrors:
    """
    Silences BOTH Python stderr AND the raw OS file-descriptor 2 so that
    C++ libraries (OpenCV, GStreamer, V4L2) cannot print error output.
    """
    def __enter__(self):
        self._python_stderr = sys.stderr
        self._devnull_fd    = os.open(os.devnull, os.O_WRONLY)
        self._saved_fd2     = os.dup(2)
        os.dup2(self._devnull_fd, 2)
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, *_):
        try:
            sys.stderr.close()
        except Exception:
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
import pyttsx3
import time
import smtplib
import logging
import requests
import threading
from email.message import EmailMessage
from pathlib import Path

# ──────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# CONFIG  ← edit only this block
# ──────────────────────────────────────────────
EMAIL_SENDER    = "sunthiago1@gmail.com"
EMAIL_PASSWORD  = "rmxh xdzf cgbm ndky"
EMAIL_RECEIVER  = "sunjian1949@gmail.com"

OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "qwen2:0.5b"

EVIDENCE_DIR    = Path("/mnt/nvme")
IMG_PATH        = EVIDENCE_DIR / "emergency.jpg"
VIDEO_PATH      = EVIDENCE_DIR / "emergency.mp4"
VIDEO_DURATION  = 5
VIDEO_FPS       = 10.0
VIDEO_SIZE      = (320, 240)

FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720

PAN_CHANNEL     = 0
TILT_CHANNEL    = 1
DEADZONE        = 20

SERVO_GAIN      = 0.015
SERVO_ALPHA     = 0.04
SERVO_HZ        = 50

TILT_REST_ANGLE = 100

KP_NOSE         = 0
KP_L_EAR        = 3
KP_R_EAR        = 4
KP_L_SHOULDER   = 5
KP_R_SHOULDER   = 6
KP_L_ELBOW      = 7
KP_R_ELBOW      = 8
KP_L_WRIST      = 9
KP_R_WRIST      = 10
KP_L_HIP        = 11
KP_R_HIP        = 12
KP_CONF_THRESH  = 0.3

STATUS_COLORS = {
    "OK":            (0,  200,   0),
    "No Person":     (180, 180, 180),
    "FALL DETECTED": (0,    0, 255),
    "HEAD PAIN":     (0,  140, 255),
    "STOMACH PAIN":  (0,  200, 255),
}

# ──────────────────────────────────────────────
# SHARED STATE
# ──────────────────────────────────────────────
_stop_event = threading.Event()

_frame_lock   = threading.Lock()
_latest_frame = None

_result_lock   = threading.Lock()
_latest_result = {"status": "No Person", "person": None, "kp_summary": ""}

_servo_lock  = threading.Lock()
_target_pan  = 90.0
_target_tilt = float(TILT_REST_ANGLE)

_alert_lock    = threading.Lock()
_last_llm_text = ""

_email_lock         = threading.Lock()
_email_banner_text  = ""
_email_banner_until = 0.0

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2


def _head_zone(nose, l_ear, r_ear, shoulder_centre):
    l_vis = l_ear[0] > 0 and l_ear[1] > 0
    r_vis = r_ear[0] > 0 and r_ear[1] > 0
    if l_vis and r_vis:
        cx = (l_ear[0] + r_ear[0]) / 2
        cy = (l_ear[1] + r_ear[1]) / 2
    else:
        cx, cy = nose[0], nose[1]
    head_height = abs(shoulder_centre[1] - nose[1]) * 0.9
    head_height = max(head_height, 40)
    radius      = head_height * 0.85
    return np.array([cx, cy]), radius


def classify_behavior(person: np.ndarray, conf) -> tuple:
    nose    = person[KP_NOSE]
    l_ear   = person[KP_L_EAR]
    r_ear   = person[KP_R_EAR]
    l_elbow = person[KP_L_ELBOW]
    r_elbow = person[KP_R_ELBOW]
    l_wrist = person[KP_L_WRIST]
    r_wrist = person[KP_R_WRIST]
    l_hip   = person[KP_L_HIP]
    r_hip   = person[KP_R_HIP]
    l_sh    = person[KP_L_SHOULDER]
    r_sh    = person[KP_R_SHOULDER]

    hip_centre      = midpoint(l_hip, r_hip)
    shoulder_centre = midpoint(l_sh, r_sh)
    stomach_centre  = midpoint(shoulder_centre, hip_centre)
    head_centre, head_radius = _head_zone(nose, l_ear, r_ear, shoulder_centre)

    l_wrist_to_head = np.linalg.norm(l_wrist - head_centre)
    r_wrist_to_head = np.linalg.norm(r_wrist - head_centre)
    l_elbow_to_head = np.linalg.norm(l_elbow - head_centre)
    r_elbow_to_head = np.linalg.norm(r_elbow - head_centre)

    summary = (
        f"nose=({nose[0]:.0f},{nose[1]:.0f}), "
        f"head_centre=({head_centre[0]:.0f},{head_centre[1]:.0f}), "
        f"head_radius={head_radius:.0f}, "
        f"left_wrist=({l_wrist[0]:.0f},{l_wrist[1]:.0f}) dist_head={l_wrist_to_head:.0f}, "
        f"right_wrist=({r_wrist[0]:.0f},{r_wrist[1]:.0f}) dist_head={r_wrist_to_head:.0f}, "
        f"stomach_est=({stomach_centre[0]:.0f},{stomach_centre[1]:.0f}), "
        f"hip_centre=({hip_centre[0]:.0f},{hip_centre[1]:.0f})"
    )

    if abs(nose[1] - hip_centre[1]) < 60:
        return "FALL DETECTED", summary

    wrist_in_head      = (l_wrist_to_head < head_radius or r_wrist_to_head < head_radius)
    both_wrists_near   = (l_wrist_to_head < head_radius * 1.4 and
                          r_wrist_to_head < head_radius * 1.4)
    l_elbow_raised     = (l_elbow[1] > 0 and l_elbow[1] < shoulder_centre[1])
    r_elbow_raised     = (r_elbow[1] > 0 and r_elbow[1] < shoulder_centre[1])
    elbow_raised_wrist = (
        (l_elbow_raised and l_wrist_to_head < head_radius * 1.6) or
        (r_elbow_raised and r_wrist_to_head < head_radius * 1.6)
    )
    wrist_above_nose   = (
        (l_wrist[1] > 0 and l_wrist[1] < nose[1] and
         abs(l_wrist[0] - head_centre[0]) < head_radius * 1.3) or
        (r_wrist[1] > 0 and r_wrist[1] < nose[1] and
         abs(r_wrist[0] - head_centre[0]) < head_radius * 1.3)
    )

    if wrist_in_head or both_wrists_near or elbow_raised_wrist or wrist_above_nose:
        return "HEAD PAIN", summary

    stomach_radius = max(abs(hip_centre[1] - shoulder_centre[1]) * 0.45, 55)
    if (np.linalg.norm(l_wrist - stomach_centre) < stomach_radius or
            np.linalg.norm(r_wrist - stomach_centre) < stomach_radius):
        return "STOMACH PAIN", summary

    return "OK", summary


def draw_overlay(frame: np.ndarray, status: str,
                 llm_text: str, person) -> np.ndarray:
    color = STATUS_COLORS.get(status, (0, 0, 255))
    with SuppressErrors():
        cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

        if llm_text:
            y, line = 110, ""
            for word in llm_text.split():
                test = f"{line} {word}".strip()
                if len(test) > 78:
                    cv2.putText(frame, line, (20, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y += 28
                    line = word
                else:
                    line = test
            if line:
                cv2.putText(frame, line, (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if person is not None:
            for idx in [KP_NOSE, KP_L_EAR, KP_R_EAR,
                        KP_L_WRIST, KP_R_WRIST,
                        KP_L_ELBOW, KP_R_ELBOW,
                        KP_L_HIP, KP_R_HIP,
                        KP_L_SHOULDER, KP_R_SHOULDER]:
                x, y = int(person[idx][0]), int(person[idx][1])
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 6, color, -1)

            shoulder_centre = midpoint(person[KP_L_SHOULDER], person[KP_R_SHOULDER])
            hc, hr = _head_zone(
                person[KP_NOSE], person[KP_L_EAR], person[KP_R_EAR], shoulder_centre)
            head_color = (0, 140, 255) if status == "HEAD PAIN" else (200, 200, 200)
            cv2.circle(frame, (int(hc[0]), int(hc[1])), int(hr), head_color, 2)
            cv2.putText(frame, "head zone",
                        (int(hc[0]) - 40, int(hc[1]) - int(hr) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, head_color, 1)

            hip_centre = midpoint(person[KP_L_HIP], person[KP_R_HIP])
            sc = midpoint(shoulder_centre, hip_centre)
            stomach_radius = max(abs(hip_centre[1] - shoulder_centre[1]) * 0.45, 55)
            stom_color = (0, 200, 255) if status == "STOMACH PAIN" else (200, 200, 200)
            cv2.circle(frame, (int(sc[0]), int(sc[1])), int(stomach_radius), stom_color, 2)
            cv2.putText(frame, "stomach zone",
                        (int(sc[0]) + int(stomach_radius) + 6, int(sc[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, stom_color, 1)

            for wrist_idx in [KP_L_WRIST, KP_R_WRIST]:
                wx, wy = int(person[wrist_idx][0]), int(person[wrist_idx][1])
                if wx > 0 and wy > 0:
                    lc = (0, 140, 255) if status == "HEAD PAIN" else (80, 80, 80)
                    cv2.line(frame, (wx, wy), (int(hc[0]), int(hc[1])), lc, 1, cv2.LINE_AA)

    return frame


def _draw_email_banner(frame: np.ndarray) -> np.ndarray:
    with _email_lock:
        text  = _email_banner_text
        until = _email_banner_until
    if not text or time.monotonic() > until:
        return frame
    bg_color = (0, 140, 0) if text.startswith("📧") else (0, 0, 180)
    banner_h = 40
    y0 = FRAME_HEIGHT - banner_h
    with SuppressErrors():
        cv2.rectangle(frame, (0, y0), (FRAME_WIDTH, FRAME_HEIGHT), bg_color, -1)
        safe_text = "Email Sent" if text.startswith("📧") else "Email Failed"
        cv2.putText(frame, safe_text, (16, y0 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return frame

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
def ask_qwen(status: str, keypoint_summary: str) -> str:
    prompt = (
        f"You are an AI medical assistant monitoring a patient via camera.\n"
        f"The computer-vision system has just detected: {status}\n"
        f"Body keypoint summary: {keypoint_summary}\n\n"
        f"In 2-3 short sentences:\n"
        f"1. Acknowledge what you observed.\n"
        f"2. Give a calm, reassuring message to the patient.\n"
        f"3. State what action is being taken (alert sent to caregiver).\n"
        f"Keep the language simple and caring. Do not use medical jargon."
    )
    payload = {
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"num_predict": 200, "temperature": 0.7},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        log.error("Cannot reach Ollama — run: ollama serve")
    except requests.exceptions.Timeout:
        log.error("Ollama timed out — model may still be loading")
    except Exception as exc:
        log.error("Qwen2 error: %s", exc)
    return f"I detected {status}. Please stay calm — help is on the way."

# ──────────────────────────────────────────────
# EMAIL / EVIDENCE
# ──────────────────────────────────────────────
def _capture_evidence() -> tuple:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with _frame_lock:
        snap = _latest_frame.copy() if _latest_frame is not None else None
    if snap is not None:
        with SuppressErrors():
            cv2.imwrite(str(IMG_PATH), snap)
        log.info("Snapshot saved (%d bytes)", IMG_PATH.stat().st_size)
    with SuppressErrors():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out    = cv2.VideoWriter(str(VIDEO_PATH), fourcc, VIDEO_FPS, VIDEO_SIZE)
    deadline = time.time() + VIDEO_DURATION
    while time.time() < deadline:
        with _frame_lock:
            f = _latest_frame.copy() if _latest_frame is not None else None
        if f is not None:
            with SuppressErrors():
                out.write(cv2.resize(f, VIDEO_SIZE))
        time.sleep(1 / VIDEO_FPS)
    out.release()
    time.sleep(0.5)
    log.info("Video saved (%d bytes)", VIDEO_PATH.stat().st_size)
    return IMG_PATH, VIDEO_PATH


def _attach(msg, path, maintype, subtype, fname):
    try:
        size = path.stat().st_size
        if size == 0:
            raise ValueError("empty file")
        with open(path, "rb") as fh:
            msg.add_attachment(fh.read(), maintype=maintype,
                               subtype=subtype, filename=fname)
        log.info("Attached %s (%d bytes)", fname, size)
    except Exception as exc:
        log.warning("Could not attach %s: %s", fname, exc)


def _set_email_banner(text: str, duration: float = 5.0) -> None:
    global _email_banner_text, _email_banner_until
    with _email_lock:
        _email_banner_text  = text
        _email_banner_until = time.monotonic() + duration


def _send_email(status: str, llm_response: str) -> None:
    img_path, video_path = _capture_evidence()
    msg = EmailMessage()
    msg["Subject"] = f"Patient Alert: {status}"
    msg["From"]    = EMAIL_SENDER
    msg["To"]      = EMAIL_RECEIVER
    msg.set_content(
        f"Emergency alert.\n\nStatus: {status}\n"
        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"AI Assessment:\n{llm_response}\n\n"
        "Please check on the patient immediately."
    )
    _attach(msg, img_path,   "image", "jpeg", "emergency.jpg")
    _attach(msg, video_path, "video", "mp4",  "emergency.mp4")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.ehlo(); s.starttls(); s.ehlo()
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.send_message(msg)
        log.info("Email sent to %s", EMAIL_RECEIVER)
        print(f"\nEmail sent successfully to {EMAIL_RECEIVER}\n")
        _set_email_banner("Email sent!", duration=6.0)
    except smtplib.SMTPAuthenticationError:
        log.error("SMTP auth failed — check Gmail App Password and 2FA.")
        print("\nEmail FAILED — check Gmail App Password\n")
        _set_email_banner("Email failed: auth error", duration=8.0)
    except Exception as exc:
        log.error("Email error: %s", exc)
        print(f"\nEmail FAILED: {exc}\n")
        _set_email_banner(f"Email failed: {exc}", duration=8.0)

# ──────────────────────────────────────────────
# THREAD 1 — CAMERA
# ──────────────────────────────────────────────
def camera_thread_fn(cap):
    global _latest_frame
    log.info("[CameraThread] started")
    while not _stop_event.is_set():
        with SuppressErrors():
            ret, frame = cap.read()
        if ret:
            with _frame_lock:
                _latest_frame = frame
        else:
            time.sleep(0.005)
    log.info("[CameraThread] stopped")

# ──────────────────────────────────────────────
# THREAD 2 — INFERENCE
# ──────────────────────────────────────────────
def inference_thread_fn(yolo_model):
    global _target_pan, _target_tilt
    log.info("[InferenceThread] started")
    while not _stop_event.is_set():
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None
        if frame is None:
            time.sleep(0.01)
            continue
        with SuppressErrors():
            results = yolo_model(frame, verbose=False)
        status  = "No Person"
        person  = None
        summary = ""
        if results and results[0].keypoints is not None:
            kp_xy   = results[0].keypoints.xy.cpu().numpy()
            kp_conf = (results[0].keypoints.conf.cpu().numpy()
                       if results[0].keypoints.conf is not None else None)
            if len(kp_xy) > 0:
                person  = kp_xy[0]
                conf    = kp_conf[0] if kp_conf is not None else None
                status, summary = classify_behavior(person, conf)
                nose  = person[KP_NOSE]
                l_hip = person[KP_L_HIP]
                r_hip = person[KP_R_HIP]
                mid_x = (nose[0] + (l_hip[0] + r_hip[0]) / 2) / 2
                mid_y = (nose[1] + (l_hip[1] + r_hip[1]) / 2) / 2
                err_x = mid_x - FRAME_WIDTH  // 2
                err_y = mid_y - FRAME_HEIGHT // 2
                if abs(err_x) < DEADZONE: err_x = 0
                if abs(err_y) < DEADZONE: err_y = 0
                with _servo_lock:
                    _target_pan  = float(np.clip(_target_pan  - err_x * SERVO_GAIN, 10, 170))
                    _target_tilt = float(np.clip(_target_tilt - err_y * SERVO_GAIN, 10, 170))
        with _result_lock:
            _latest_result["status"]     = status
            _latest_result["person"]     = person
            _latest_result["kp_summary"] = summary
    log.info("[InferenceThread] stopped")

# ──────────────────────────────────────────────
# THREAD 3 — SERVO
# ──────────────────────────────────────────────
def servo_thread_fn(kit):
    log.info("[ServoThread] started  (alpha=%.3f, %.0f Hz)", SERVO_ALPHA, SERVO_HZ)
    pan_actual  = 90.0
    tilt_actual = float(TILT_REST_ANGLE)
    interval    = 1.0 / SERVO_HZ
    while not _stop_event.is_set():
        t0 = time.monotonic()
        with _servo_lock:
            t_pan  = _target_pan
            t_tilt = _target_tilt
        pan_actual  += SERVO_ALPHA * (t_pan  - pan_actual)
        tilt_actual += SERVO_ALPHA * (t_tilt - tilt_actual)
        kit.servo[PAN_CHANNEL].angle  = pan_actual
        kit.servo[TILT_CHANNEL].angle = tilt_actual
        sleep_t = interval - (time.monotonic() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)
    log.info("[ServoThread] stopped")

# ──────────────────────────────────────────────
# THREAD 4 — ALERT
# ──────────────────────────────────────────────
def alert_thread_fn(status: str, kp_summary: str) -> None:
    global _last_llm_text
    log.info("[AlertThread] fired for: %s", status)
    llm_response = ask_qwen(status, kp_summary)
    print("\n" + "=" * 60)
    print(f"  QWEN2 RESPONSE [{status}]")
    print("=" * 60)
    print(llm_response)
    print("=" * 60 + "\n")
    with _alert_lock:
        _last_llm_text = llm_response
    try:
        local_tts = pyttsx3.init()
        local_tts.setProperty("rate", 155)
        local_tts.say(llm_response)
        local_tts.runAndWait()
    except Exception as exc:
        log.warning("TTS error: %s", exc)
    _send_email(status, llm_response)
    log.info("[AlertThread] done")

# ──────────────────────────────────────────────
# HARDWARE INIT
# ──────────────────────────────────────────────
log.info("Loading YOLOv8-pose model...")
with SuppressErrors():
    model = YOLO("yolov8n-pose.pt")

log.info("Initialising servos...")
kit = ServoKit(channels=16)
kit.servo[PAN_CHANNEL].angle  = 90.0
kit.servo[TILT_CHANNEL].angle = float(TILT_REST_ANGLE)

log.info("Opening camera...")
with SuppressErrors():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
log.info("Camera: %.0f x %.0f",
         cap.get(cv2.CAP_PROP_FRAME_WIDTH),
         cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ──────────────────────────────────────────────
# START BACKGROUND THREADS
# ──────────────────────────────────────────────
bg_threads = [
    threading.Thread(target=camera_thread_fn,    args=(cap,),
                     daemon=True, name="CameraThread"),
    threading.Thread(target=inference_thread_fn, args=(model,),
                     daemon=True, name="InferenceThread"),
    threading.Thread(target=servo_thread_fn,     args=(kit,),
                     daemon=True, name="ServoThread"),
]
for t in bg_threads:
    t.start()

log.info("All threads running — press Q in the window to quit.")

# ──────────────────────────────────────────────
# MAIN THREAD — DISPLAY LOOP
# ──────────────────────────────────────────────
alert_dispatched = False

try:
    while True:
        with _result_lock:
            status  = _latest_result["status"]
            person  = _latest_result["person"]
            summary = _latest_result["kp_summary"]
        with _alert_lock:
            llm_text = _last_llm_text

        if status not in ("OK", "No Person") and not alert_dispatched:
            alert_dispatched = True
            threading.Thread(
                target=alert_thread_fn,
                args=(status, summary),
                daemon=True, name="AlertThread"
            ).start()

        if status in ("OK", "No Person"):
            if alert_dispatched:
                alert_dispatched = False
                with _alert_lock:
                    _last_llm_text = ""
            llm_text = ""

        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None

        if frame is not None:
            frame = draw_overlay(frame, status, llm_text, person)
            frame = _draw_email_banner(frame)
            with SuppressErrors():
                cv2.imshow("Patient Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    log.info("Shutting down — waiting for threads...")
    _stop_event.set()
    for t in bg_threads:
        t.join(timeout=3.0)
    with SuppressErrors():
        cap.release()
        cv2.destroyAllWindows()
    log.info("Shutdown complete.")
