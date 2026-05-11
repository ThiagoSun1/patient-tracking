# =========================
# 🔇 SILENCE OPENCV + C++ ERRORS
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
        # open /dev/null at the OS level
        self._devnull_fd = os.open(os.devnull, os.O_WRONLY)
        # save a dup of the real fd 2 so we can restore it later
        self._saved_fd2  = os.dup(2)
        # point fd 2 (what C++ writes to) at /dev/null
        os.dup2(self._devnull_fd, 2)
        # also silence the Python-level stderr object
        sys.stderr = open(os.devnull, 'w')
        return self

    def __exit__(self, *_):
        try:
            sys.stderr.close()
        except Exception:
            pass
        # restore fd 2 to the real stderr
        os.dup2(self._saved_fd2, 2)
        os.close(self._saved_fd2)
        os.close(self._devnull_fd)
        # restore Python stderr
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
import threading
from email.message import EmailMessage
from collections import deque
from flask import Flask, Response, render_template_string, jsonify

# ----------------------------
# CONFIG — edit these only
# ----------------------------
EMAIL_FROM     = "sunthiago1@gmail.com"
EMAIL_TO       = "sunjian1949@gmail.com"
EMAIL_PASSWORD = "rmxh xdzf cgbm ndky"

FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

PAN_CHANNEL    = 0
TILT_CHANNEL   = 1

CONF_THRESHOLD  = 0.4
DEADZONE        = 20
SERVO_SPEED     = 0.02
DEBOUNCE_FRAMES = 20

# Wave detection config
WAVE_FRAMES_NEEDED = 20
WAVE_MIN_DISTANCE  = 40

# ----------------------------
# FLASK APP
# ----------------------------
app = Flask(__name__)

latest_frame  = None
latest_status = "Waiting for wave..."
alert_log     = []
system_log    = []
frame_lock    = threading.Lock()
system_active = False

# ----------------------------
# SYSTEM EMAIL
# ----------------------------
def send_system_email(subject, body):
    def _send():
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"]    = EMAIL_FROM
            msg["To"]      = EMAIL_TO
            msg.set_content(body)
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"System email sent: {subject}")
        except Exception as e:
            print(f"System email failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

def log_system_event(event, send_email=True):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    system_log.append({'event': event, 'time': timestamp})
    print(f"[SYSTEM] {event}")
    if send_email:
        send_system_email(
            subject=f"System Event: {event}",
            body=f"System Event: {event}\nTime: {timestamp}"
        )

# ----------------------------
# WEB APP HTML
# ----------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black">
    <meta name="apple-mobile-web-app-title" content="Patient Monitor">
    <title>Patient Monitor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #111;
            color: white;
            font-family: -apple-system, sans-serif;
            min-height: 100vh;
        }
        header {
            background: #1a1a1a;
            padding: 16px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        #status-bar {
            padding: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            background: #222;
            transition: background 0.3s;
        }
        #status-bar.ok        { background: #1a4a1a; color: #4cff4c; }
        #status-bar.alert     { background: #4a1a1a; color: #ff4c4c; }
        #status-bar.no-person { background: #2a2a2a; color: #aaaaaa; }
        #status-bar.system    { background: #4a3a1a; color: #ffaa4c; }
        #status-bar.waiting   { background: #1a2a4a; color: #4caaff; }
        #feed {
            width: 100%;
            max-width: 640px;
            display: block;
            margin: 0 auto;
        }
        .section {
            padding: 16px;
            max-width: 640px;
            margin: 0 auto;
        }
        .section h2 {
            font-size: 16px;
            color: #aaa;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .alert-item {
            background: #1a1a1a;
            border-left: 4px solid #ff4c4c;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        .system-item {
            background: #1a1a1a;
            border-left: 4px solid #ffaa4c;
            padding: 10px 14px;
            margin-bottom: 8px;
            border-radius: 4px;
            font-size: 14px;
        }
        .item-time {
            color: #888;
            font-size: 12px;
            margin-top: 4px;
        }
        .no-items {
            color: #555;
            font-size: 14px;
            text-align: center;
            padding: 20px;
        }
        #system-status {
            padding: 10px 16px;
            text-align: center;
            font-size: 14px;
            background: #1a1a1a;
            color: #aaa;
        }
        #system-status.warning      { background: #2a2000; color: #ffaa4c; }
        #system-status.disconnected { background: #4a1a1a; color: #ff4c4c; }
        #system-status.waiting      { background: #1a2a4a; color: #4caaff; }
        #wave-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            z-index: 100;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        #wave-overlay.show { display: flex; }
        #wave-overlay h1 { font-size: 48px; margin-bottom: 16px; }
        #wave-overlay p  { font-size: 18px; color: #aaa; }
    </style>
</head>
<body>
    <header>🏥 Patient Monitor</header>
    <div id="status-bar" class="waiting">Waiting for wave...</div>
    <div id="system-status" class="waiting">System: Standby</div>
    <img id="feed" src="/video_feed" />

    <div class="section">
        <h2>🚨 Alert History</h2>
        <div id="alert-list"><p class="no-items">No alerts yet</p></div>
    </div>

    <div class="section">
        <h2>⚙️ System Events</h2>
        <div id="system-list"><p class="no-items">No system events</p></div>
    </div>

    <div id="wave-overlay" class="show">
        <h1>👋</h1>
        <p>Wave at the camera to start monitoring</p>
    </div>

    <script>
        let wasActive = false;

        function updateStatus() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    const bar     = document.getElementById('status-bar');
                    const sysBar  = document.getElementById('system-status');
                    const overlay = document.getElementById('wave-overlay');

                    if (data.system_active) {
                        overlay.classList.remove('show');
                        if (!wasActive) wasActive = true;
                    } else {
                        overlay.classList.add('show');
                    }

                    bar.textContent = data.status;
                    if (!data.system_active) {
                        bar.className = 'waiting';
                    } else if (data.status === 'OK') {
                        bar.className = 'ok';
                    } else if (data.status === 'No Person' || data.status === 'Starting...') {
                        bar.className = 'no-person';
                    } else if (data.status.includes('Camera') || data.status.includes('System')) {
                        bar.className = 'system';
                    } else {
                        bar.className = 'alert';
                    }

                    sysBar.textContent = 'System: ' + data.system_status;
                    if (!data.system_active) {
                        sysBar.className = 'waiting';
                    } else if (data.system_status === 'Online') {
                        sysBar.className = '';
                    } else if (data.system_status === 'Camera Disconnected') {
                        sysBar.className = 'disconnected';
                    } else {
                        sysBar.className = 'warning';
                    }

                    const alertList = document.getElementById('alert-list');
                    if (data.alerts.length === 0) {
                        alertList.innerHTML = '<p class="no-items">No alerts yet</p>';
                    } else {
                        alertList.innerHTML = data.alerts.slice().reverse().map(a =>
                            `<div class="alert-item">
                                <strong>${a.status}</strong>
                                <div class="item-time">${a.time}</div>
                            </div>`
                        ).join('');
                    }

                    const sysList = document.getElementById('system-list');
                    if (data.system_log.length === 0) {
                        sysList.innerHTML = '<p class="no-items">No system events</p>';
                    } else {
                        sysList.innerHTML = data.system_log.slice().reverse().map(s =>
                            `<div class="system-item">
                                <strong>${s.event}</strong>
                                <div class="item-time">${s.time}</div>
                            </div>`
                        ).join('');
                    }
                })
                .catch(() => {
                    document.getElementById('status-bar').textContent = 'System Offline';
                    document.getElementById('status-bar').className = 'alert';
                    document.getElementById('system-status').textContent = 'System: Disconnected';
                    document.getElementById('system-status').className = 'disconnected';
                });
        }

        setInterval(updateStatus, 1000);
        updateStatus();
    </script>
</body>
</html>
"""

# ----------------------------
# FLASK ROUTES
# ----------------------------
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.05)
                    continue
                with SuppressErrors():
                    _, jpeg = cv2.imencode('.jpg', latest_frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 70])
                frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.05)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    recent = [s['event'] for s in system_log[-3:]]
    if any('Camera lost' in e for e in recent):
        sys_status = 'Camera Disconnected'
    elif any('reconnect' in e.lower() for e in recent):
        sys_status = 'Reconnecting...'
    elif system_active:
        sys_status = 'Online'
    else:
        sys_status = 'Standby'

    return jsonify({
        'status':        latest_status,
        'system_status': sys_status,
        'system_active': system_active,
        'alerts':        alert_log[-20:],
        'system_log':    system_log[-20:]
    })

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# ----------------------------
# LOAD MODEL (SAFE)
# ----------------------------
print("Loading YOLO model...")
with SuppressErrors():
    model = YOLO("yolov8n-pose.pt")
print("Model loaded.")

# ----------------------------
# SERVO SETUP
# ----------------------------
kit = ServoKit(channels=16)

pan_angle  = 90.0
tilt_angle = 90.0

kit.servo[PAN_CHANNEL].angle  = pan_angle
kit.servo[TILT_CHANNEL].angle = tilt_angle

# ----------------------------
# CAMERA SETUP
# ----------------------------
def open_camera():
    with SuppressErrors():
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

cap = open_camera()

# ----------------------------
# TTS
# ----------------------------
tts_engine = pyttsx3.init()
tts_lock   = threading.Lock()

def speak(text):
    def _speak():
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ----------------------------
# WAVE DETECTION
# ----------------------------
wave_history     = deque(maxlen=30)
wave_frame_count = 0
wave_direction   = None
wave_switches    = 0

def detect_wave(person, conf):
    global wave_frame_count, wave_direction, wave_switches

    def visible(idx):
        return conf[idx] >= CONF_THRESHOLD

    if not (visible(5) and visible(6)):
        wave_history.clear()
        return False

    shoulder_mid = (person[5] + person[6]) / 2

    wrist = None
    for wrist_idx in [9, 10]:
        if visible(wrist_idx):
            w = person[wrist_idx]
            if w[1] < shoulder_mid[1]:
                wrist = w
                break

    if wrist is None:
        wave_history.clear()
        wave_frame_count = 0
        wave_switches    = 0
        wave_direction   = None
        return False

    wave_history.append(wrist[0])

    if len(wave_history) < 5:
        return False

    recent    = list(wave_history)
    current_x = recent[-1]
    prev_x    = recent[-3]
    diff      = current_x - prev_x

    if abs(diff) > WAVE_MIN_DISTANCE:
        new_direction = 'right' if diff > 0 else 'left'
        if wave_direction and new_direction != wave_direction:
            wave_switches += 1
        wave_direction = new_direction

    if wave_switches >= 3:
        wave_switches  = 0
        wave_direction = None
        wave_history.clear()
        return True

    return False

# ----------------------------
# BLUR ALL FACES
# ----------------------------
def blur_faces(frame, kp, confs):
    for person_idx in range(len(kp)):
        person = kp[person_idx]
        conf   = confs[person_idx]

        face_points = []
        for idx in [0, 1, 2, 3, 4]:
            if conf[idx] >= CONF_THRESHOLD:
                face_points.append(person[idx])

        if len(face_points) == 0:
            continue

        face_points = np.array(face_points)
        x_min = int(np.min(face_points[:, 0]))
        x_max = int(np.max(face_points[:, 0]))
        y_min = int(np.min(face_points[:, 1]))
        y_max = int(np.max(face_points[:, 1]))

        padding = max(30, int((x_max - x_min) * 0.6))
        x_min = max(0, x_min - padding)
        x_max = min(FRAME_WIDTH,  x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(FRAME_HEIGHT, y_max + padding)

        if x_max <= x_min or y_max <= y_min:
            continue

        with SuppressErrors():
            face_region = frame[y_min:y_max, x_min:x_max]
            blurred     = cv2.GaussianBlur(face_region, (99, 99), 30)
            frame[y_min:y_max, x_min:x_max] = blurred

# ----------------------------
# DRAW KEYPOINTS
# ----------------------------
def draw_keypoints(frame, person, conf):
    for i, (point, confidence) in enumerate(zip(person, conf)):
        if confidence < CONF_THRESHOLD:
            continue
        x, y = int(point[0]), int(point[1])
        with SuppressErrors():
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# ----------------------------
# SERVO TRACKING
# ----------------------------
def track_body(x, y):
    global pan_angle, tilt_angle

    cx = FRAME_WIDTH  // 2
    cy = FRAME_HEIGHT // 2

    error_x = x - cx
    error_y = y - cy

    if abs(error_x) < DEADZONE: error_x = 0
    if abs(error_y) < DEADZONE: error_y = 0

    pan_angle  += -error_x * SERVO_SPEED
    tilt_angle += -error_y * SERVO_SPEED

    pan_angle  = float(np.clip(pan_angle,  10, 170))
    tilt_angle = float(np.clip(tilt_angle, 10, 170))

    kit.servo[PAN_CHANNEL].angle  = pan_angle
    kit.servo[TILT_CHANNEL].angle = tilt_angle

# ----------------------------
# BEHAVIOR DETECTION
# ----------------------------
def classify_behavior(person, confs):
    def visible(idx):
        return confs[idx] >= CONF_THRESHOLD

    if not (visible(5) and visible(6) and visible(11) and visible(12)):
        return "OK"

    l_shoulder = person[5]
    r_shoulder = person[6]
    l_hip      = person[11]
    r_hip      = person[12]
    nose       = person[0]

    shoulder_mid = (l_shoulder + r_shoulder) / 2
    hip_mid      = (l_hip      + r_hip)      / 2

    torso_height = abs(shoulder_mid[1] - hip_mid[1])
    if torso_height < 20:
        return "OK"

    stomach = np.array([hip_mid[0], shoulder_mid[1] + 0.5 * torso_height])

    head_points = []
    for idx in [0, 1, 2, 3, 4]:
        if confs[idx] >= CONF_THRESHOLD:
            head_points.append(person[idx])

    # FALL DETECTED
    if visible(0):
        if nose[1] >= hip_mid[1] - (0.1 * torso_height):
            return "FALL DETECTED"

    # HEAD PAIN
    if len(head_points) > 0:
        head_points_arr = np.array(head_points)
        head_center     = head_points_arr.mean(axis=0)
        head_radius     = torso_height * 0.35

        hands_on_head = 0
        for wrist_idx in [9, 10]:
            if not visible(wrist_idx):
                continue
            wrist          = person[wrist_idx]
            dist_to_head   = np.linalg.norm(wrist - head_center)
            above_shoulder = wrist[1] < shoulder_mid[1] + (0.1 * torso_height)
            if dist_to_head < head_radius and above_shoulder:
                hands_on_head += 1

        if hands_on_head >= 1:
            return "HEAD PAIN"

    # STOMACH PAIN
    torso_width      = abs(l_shoulder[0] - r_shoulder[0])
    hands_on_stomach = 0

    for wrist_idx in [9, 10]:
        if not visible(wrist_idx):
            continue
        wrist = person[wrist_idx]

        in_vertical_zone   = shoulder_mid[1] < wrist[1] < hip_mid[1]
        in_horizontal_zone = abs(wrist[0] - stomach[0]) < 0.35 * max(torso_width, 40)
        close_to_stomach   = np.linalg.norm(wrist - stomach) < 0.22 * torso_height

        if in_vertical_zone and in_horizontal_zone and close_to_stomach:
            hands_on_stomach += 1

    if hands_on_stomach >= 1:
        return "STOMACH PAIN"

    return "OK"

# ----------------------------
# CAPTURE EVIDENCE
# ----------------------------
def capture_evidence():
    img_path   = "/mnt/nvme/emergency.jpg"
    video_path = "/mnt/nvme/emergency.mp4"

    for _ in range(3):
        with SuppressErrors():
            cap.grab()

    with SuppressErrors():
        ret, frame = cap.read()
    if ret:
        with SuppressErrors():
            cv2.imwrite(img_path, frame)
        print("Image saved.")
    else:
        print("Warning: could not capture image.")
        img_path = None

    with SuppressErrors():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
    start = time.time()

    while time.time() - start < 5:
        with SuppressErrors():
            ret, frame = cap.read()
        if ret:
            with SuppressErrors():
                out.write(cv2.resize(frame, (320, 240)))

    out.release()
    time.sleep(0.5)
    print("Video saved.")

    return img_path, video_path

# ----------------------------
# EMAIL ALERT
# ----------------------------
def send_email_alert(status):
    print("Sending email alert for:", status)
    try:
        img_path, video_path = capture_evidence()

        msg = EmailMessage()
        msg["Subject"] = f"Emergency Detected: {status}"
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO
        msg.set_content(
            f"Emergency detected: {status}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        if img_path and os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            with open(img_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image",
                                   subtype="jpeg", filename="emergency.jpg")

        if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            with open(video_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="video",
                                   subtype="mp4", filename="emergency.mp4")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)

        print("Email sent successfully.")

    except Exception as e:
        print("Email failed:", e)

def alert_in_background(status):
    def _run():
        send_email_alert(status)
    threading.Thread(target=_run, daemon=True).start()
    speak(f"I detected {status}. Are you okay?")

# ----------------------------
# DEBOUNCE
# ----------------------------
status_history = deque(maxlen=DEBOUNCE_FRAMES)

def debounced_status(new_status):
    status_history.append(new_status)
    if len(status_history) == DEBOUNCE_FRAMES and len(set(status_history)) == 1:
        return new_status
    return None

# ----------------------------
# START FLASK IN BACKGROUND
# ----------------------------
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
print("Flask server started on port 5000.")

log_system_event("System started — waiting for wave", send_email=True)

# ----------------------------
# MAIN LOOP
# ----------------------------
alert_triggered      = False
last_alert_status    = None
frame_fail_count     = 0
MAX_FAIL_FRAMES      = 30
disconnected_emailed = False

print("Starting patient monitor — wave at camera to begin. Press Q to quit.")

try:
    while True:

        with SuppressErrors():
            ret, frame = cap.read()

        if not ret or frame is None:
            frame_fail_count += 1
            print(f"Frame read failed ({frame_fail_count}/{MAX_FAIL_FRAMES})")

            if frame_fail_count == 1:
                log_system_event("Camera frame read failed", send_email=True)

            if frame_fail_count >= MAX_FAIL_FRAMES:
                if not disconnected_emailed:
                    log_system_event("System Disconnected — camera lost", send_email=True)
                    disconnected_emailed = True
                with frame_lock:
                    latest_status = "Camera Disconnected"
                with SuppressErrors():
                    cap.release()
                time.sleep(2)
                cap = open_camera()
                frame_fail_count     = 0
                disconnected_emailed = False
                log_system_event("Camera reconnected — system back online", send_email=True)
            continue

        frame_fail_count     = 0
        disconnected_emailed = False

        with SuppressErrors():
            results = model(frame, verbose=False)

        raw_status = "No Person"

        if (len(results) > 0
                and results[0].keypoints is not None
                and results[0].keypoints.conf is not None):

            kp    = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()

            if len(kp) > 0:
                person = kp[0]
                conf   = confs[0]

                # WAVE DETECTION — runs always
                if not system_active:
                    waved = detect_wave(person, conf)
                    if waved:
                        system_active = True
                        speak("System activated. I am now monitoring you.")
                        log_system_event("Wave detected — monitoring started", send_email=True)
                        print("Wave detected! System activated.")

                    with SuppressErrors():
                        cv2.putText(frame, "Wave to Start", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        cv2.putText(frame, "Raise hand above shoulder and wave", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    with frame_lock:
                        latest_frame  = frame.copy()
                        latest_status = "Waiting for wave..."

                    with SuppressErrors():
                        cv2.imshow("Patient Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # SYSTEM ACTIVE — normal monitoring
                draw_keypoints(frame, person, conf)
                blur_faces(frame, kp, confs)

                if (conf[5] >= CONF_THRESHOLD and conf[6] >= CONF_THRESHOLD and
                        conf[11] >= CONF_THRESHOLD and conf[12] >= CONF_THRESHOLD):
                    torso_x = int((person[5][0] + person[6][0] +
                                   person[11][0] + person[12][0]) / 4)
                    torso_y = int((person[5][1] + person[6][1] +
                                   person[11][1] + person[12][1]) / 4)
                    if torso_x > 0 and torso_y > 0:
                        track_body(torso_x, torso_y)

                raw_status = classify_behavior(person, conf)

        # Only run alert logic when system is active
        if system_active:
            stable = debounced_status(raw_status)

            if stable and stable not in ["OK", "No Person"]:
                if not alert_triggered or stable != last_alert_status:
                    print("ALERT:", stable)
                    alert_log.append({
                        'status': stable,
                        'time':   time.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    alert_in_background(stable)
                    alert_triggered   = True
                    last_alert_status = stable

            if raw_status in ["OK", "No Person"]:
                alert_triggered   = False
                last_alert_status = None

            color = (0, 255, 0) if raw_status == "OK" else (0, 0, 255)
            with SuppressErrors():
                cv2.putText(frame, raw_status, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                if stable and stable not in ["OK", "No Person"]:
                    cv2.putText(frame, "!! CONFIRMED !!", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        with frame_lock:
            latest_frame  = frame.copy()
            latest_status = raw_status

        with SuppressErrors():
            cv2.imshow("Patient Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    log_system_event(f"System error: {str(e)}", send_email=True)
    print("Fatal error:", e)

finally:
    log_system_event("System shut down", send_email=True)
    with SuppressErrors():
        cap.release()
        cv2.destroyAllWindows()
    print("Monitor shut down.")
