"""
================================================================================
ROBOTIC ARM CONTROL V3 — REAL-TIME GESTURE, GEMINI VOICE & AUTO-VELOCITY
================================================================================
1. Dependencies:
   pip install streamlit pyserial opencv-python mediapipe numpy pandas plotly scipy sounddevice google-generativeai SpeechRecognition pyaudio

2. Setup Gemini API Key (Optional for Voice Control):
   Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"

3. Run Application:
   streamlit run newrr.py
================================================================================
"""

import streamlit as st
import serial
import serial.tools.list_ports
import numpy as np
import os
import cv2
from datetime import datetime
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from scipy.linalg import pinv
import time
import base64
import pandas as pd
import threading
import queue
import json
import re
import urllib.request

# ==========================================
# OPTIONAL IMPORTS (GEMINI & SPEECH)
# ==========================================
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
st.set_page_config(
    page_title="🦾 6-DOF Robotic Arm Master Controller",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 3D GLB Model Path
GLB_PATH = r"C:\Users\hp\Downloads\Robotic Arm 3D Model.glb"
TASK_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# DH Parameters [theta_offset, d, a, alpha]
DH_PARAMS = np.array([
    [0, 0.15, 0,    -np.pi / 2],
    [0, 0,    0.3,   0         ],
    [0, 0,    0.25,  0         ],
    [0, 0,    0,    -np.pi / 2],
    [0, 0.1,  0,     np.pi / 2],
    [0, 0.05, 0,     0         ]
])

# Joint Limits (degrees) — [Min, Max]
JOINT_LIMITS = np.array([
    [0, 180],   # Waist (J0)
    [0, 180],   # Shoulder (J1)
    [0, 180],   # Elbow (J2)
    [0, 180],   # Wrist Pitch (J3)
    [0, 180],   # Wrist Roll (J4)
    [0,  90]    # Gripper (J5)
])

# Servo-to-kinematic offset (Servo 90° = Kinematic 0° for joints 0-4; Servo 45° = Kinematic 0° for gripper)
JOINT_OFFSETS = np.array([90, 90, 90, 90, 90, 45], dtype=np.float64)
JOINT_NAMES = ['Waist', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Gripper']
DEFAULT_ANGLES = np.array([90, 90, 90, 90, 90, 45], dtype=np.float64)

# Velocity & Filter Settings
MIN_DT = 1e-3
VELOCITY_ALPHA = 0.35
VELOCITY_CLAMP = 120.0  # Max realistic servo velocity (°/s)

# ==========================================
# CUSTOM STYLING (DARK CYBER THEME)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; 
        background-color: #0b0f19; 
        color: #e6edf3; 
    }
    
    #MainMenu {visibility: hidden;} 
    footer {visibility: hidden;}
    
    .bento-container { 
        background: #111827; 
        border: 1px solid #1f293d; 
        border-radius: 14px; 
        padding: 20px; 
        margin-bottom: 20px; 
        box-shadow: 0 8px 32px rgba(0,0,0,0.5); 
        transition: all 0.3s ease; 
    }
    .bento-container:hover { 
        border-color: #3b82f6; 
        box-shadow: 0 10px 35px rgba(59, 130, 246, 0.15); 
    }
    
    h1, h2, h3, h4 { color: #f8fafc !important; font-weight: 700; }
    
    div.stButton > button { 
        width: 100%; 
        border-radius: 10px; 
        height: 48px; 
        font-weight: 600; 
        border: 1px solid #1f293d; 
        background: #1e293b; 
        color: #f1f5f9; 
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); 
    }
    div.stButton > button:hover { 
        background: #2563eb; 
        color: #ffffff; 
        border-color: #3b82f6; 
        transform: translateY(-2px); 
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }
    
    [data-testid="stSlider"] > div > div > div > div { background: #3b82f6; }
    
    .metric-card { 
        background: #1e293b; 
        padding: 12px 14px; 
        border-radius: 8px; 
        border-left: 4px solid #3b82f6; 
        margin-bottom: 8px; 
        font-size: 0.9rem;
    }
    .metric-card.warning { border-left-color: #f59e0b; }
    .metric-card.danger  { border-left-color: #ef4444; }
    .metric-card.success { border-left-color: #10b981; }
    
    .vel-badge { 
        display: inline-block; 
        background: #0f172a; 
        border: 1px solid #334155; 
        border-radius: 6px;
        padding: 2px 8px; 
        font-family: 'JetBrains Mono', monospace; 
        font-size: 0.75rem; 
        color: #60a5fa; 
        margin: 2px 3px; 
    }
    
    .joint-label {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .joint-value {
        background: #0f172a;
        padding: 2px 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: #38bdf8;
        border: 1px solid #334155;
    }
    
    .voice-status {
        padding: 12px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .voice-listening { background: #064e3b; color: #34d399; border: 2px solid #10b981; }
    .voice-processing { background: #1e3a8a; color: #93c5fd; border: 2px solid #3b82f6; }
    .voice-error { background: #7f1d1d; color: #fca5a5; border: 2px solid #ef4444; }
    
    .gesture-hud {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 12px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3D GLB MODEL VIEWER
# ==========================================
def render_glb_model(file_path):
    if not os.path.exists(file_path):
        return (f"<div class='bento-container' style='text-align:center; color:#ef4444;'>"
                f"<h3>⚠️ Model Not Found</h3><p>Could not find: {file_path}</p></div>")
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>
    <model-viewer src="data:model/gltf-binary;base64,{b64}" alt="Robotic Arm"
        camera-controls auto-rotate shadow-intensity="1"
        environment-image="neutral"
        style="width:100%;height:490px;background-color:#0b0f19;border-radius:12px;--poster-color:transparent;">
    </model-viewer>"""


# ==========================================
# ROBOTICS KINEMATICS CLASS
# ==========================================
class RobotKinematics:
    def __init__(self, dh_params):
        self.dh_params    = dh_params
        self.n_joints     = len(dh_params)
        self.joint_limits = JOINT_LIMITS

    def dh_transform(self, theta, d, a, alpha):
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [ 0,     sa,     ca,    d],
            [ 0,      0,      0,    1]
        ])

    def forward_kinematics(self, joint_angles_deg):
        kinematic_angles = joint_angles_deg - JOINT_OFFSETS
        joint_angles_rad = np.radians(kinematic_angles)
        
        T = np.eye(4)
        transforms = []
        for i in range(self.n_joints):
            theta = joint_angles_rad[i] + self.dh_params[i, 0]
            d, a, alpha = self.dh_params[i, 1], self.dh_params[i, 2], self.dh_params[i, 3]
            Ti = self.dh_transform(theta, d, a, alpha)
            T  = T @ Ti
            transforms.append(T.copy())
        return T, transforms

    def get_end_effector_pose(self, joint_angles_deg):
        T, _ = self.forward_kinematics(joint_angles_deg)
        position        = T[:3, 3]
        rotation_matrix = T[:3, :3]
        r = R.from_matrix(rotation_matrix)
        return {
            'position':        position,
            'orientation':     r.as_euler('xyz', degrees=True),
            'rotation_matrix': rotation_matrix,
            'T':               T
        }

    def calculate_jacobian(self, joint_angles_deg, delta=1e-6):
        pose  = self.get_end_effector_pose(joint_angles_deg)
        pos0  = pose['position']
        rot0  = pose['rotation_matrix']
        jacobian = np.zeros((6, self.n_joints))
        for i in range(self.n_joints):
            angles_plus      = joint_angles_deg.copy()
            angles_plus[i]  += np.degrees(delta)
            pose_plus        = self.get_end_effector_pose(angles_plus)
            jacobian[:3, i]  = (pose_plus['position'] - pos0) / delta
            dR               = (pose_plus['rotation_matrix'] - rot0) / delta
            skew             = dR @ rot0.T
            jacobian[3:, i]  = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        return jacobian

    def calculate_end_effector_velocity(self, joint_angles_deg, joint_velocities_deg_s):
        J      = self.calculate_jacobian(joint_angles_deg)
        q_dot  = np.radians(joint_velocities_deg_s)
        v_ee   = J @ q_dot
        return {
            'velocity_vector': v_ee,
            'linear_speed':    np.linalg.norm(v_ee[:3]),
            'angular_speed':   np.linalg.norm(v_ee[3:]),
            'jacobian':        J
        }

    def calculate_manipulability(self, joint_angles_deg):
        J    = self.calculate_jacobian(joint_angles_deg)
        J_v  = J[:3, :]
        w    = np.sqrt(max(0.0, np.linalg.det(J_v @ J_v.T)))
        try:
            sigma = np.linalg.svd(J_v, compute_uv=False)
            condition_number = sigma[0] / sigma[-1] if sigma[-1] > 1e-10 else float('inf')
        except Exception:
            condition_number = float('inf')
        return {
            'manipulability':  w,
            'condition_number': condition_number,
            'is_singular':     condition_number > 100
        }

    def inverse_kinematics(self, target_pos, target_orientation_deg, current_angles,
                           max_iter=200, tolerance=1e-4):
        angles     = np.array(current_angles, dtype=np.float64)
        lam        = 0.1
        error_norm = float('inf')

        for iteration in range(max_iter):
            current_pose = self.get_end_effector_pose(angles)
            pos_error    = target_pos - current_pose['position']
            target_r     = R.from_euler('xyz', target_orientation_deg, degrees=True)
            rot_err_mat  = target_r.as_matrix() @ current_pose['rotation_matrix'].T
            rot_error    = R.from_matrix(rot_err_mat).as_rotvec()
            error        = np.concatenate([pos_error * 2.0, rot_error])
            error_norm   = np.linalg.norm(error)

            if error_norm < tolerance:
                break

            J     = self.calculate_jacobian(angles)
            manip = self.calculate_manipulability(angles)
            lam   = min(lam * 1.5, 1.0) if manip['is_singular'] else max(lam * 0.9, 0.001)

            try:
                JT      = J.T
                JTJ     = J @ JT
                delta_q = JT @ np.linalg.inv(JTJ + lam**2 * np.eye(6)) @ error
            except Exception:
                delta_q = pinv(J) @ error

            angles += np.degrees(delta_q)
            for i in range(self.n_joints):
                angles[i] = np.clip(angles[i], self.joint_limits[i][0], self.joint_limits[i][1])
            angles[5] = np.clip(angles[5], 0, 90)

        return {
            'angles':     angles,
            'iterations': iteration + 1,
            'converged':  error_norm < tolerance,
            'warnings':   []
        }


# ==========================================
# HARDWARE SERIAL CONTROLLER
# ==========================================
class RobotController:
    def __init__(self):
        self.ser        = None
        self.connected  = False
        self.port       = None
        self.kinematics = RobotKinematics(DH_PARAMS)
        self.sequence   = []
        self.last_sent  = np.zeros(6)
        self.last_send_time = 0.0

    def connect(self, port, baud=115200):
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            self.connected = True
            self.port      = port
            if self.ser.in_waiting > 0:
                self.ser.reset_input_buffer()
            return True
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            return False

    def disconnect(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connected = False

    def send_command(self, angles, min_interval=0.03):
        """Sends command to hardware with rate limiting and deadband filtering."""
        if not self.connected or not self.ser:
            return False
        
        now = time.time()
        # Rate limit to prevent serial buffer congestion
        if now - self.last_send_time < min_interval:
            return False

        # Deadband: Only send if angles have changed by at least 1 degree
        if np.max(np.abs(np.array(angles) - self.last_sent)) < 0.5:
            return False

        cmd = (f"SET,{int(angles[0])},{int(angles[1])},{int(angles[2])},"
               f"{int(angles[3])},{int(angles[4])},{int(angles[5])}\n")
        try:
            self.ser.write(cmd.encode('utf-8'))
            self.last_sent = np.array(angles, dtype=float).copy()
            self.last_send_time = now
            return True
        except Exception as e:
            st.error(f"Send Error: {e}")
            return False

    def send_reset(self):
        if not self.connected or not self.ser:
            return False
        try:
            self.ser.write(b"RESET\n")
            return True
        except Exception as e:
            st.error(f"Reset Error: {e}")
            return False


# ==========================================
# ADVANCED GESTURE CONTROLLER (MEDIAPIPE AI)
# ==========================================
class GestureController:
    """
    High-performance MediaPipe Hand Gesture Controller.
    Supports both modern MediaPipe Tasks (0.10+) and legacy APIs.
    Includes low-pass filtering, scale invariance, and rich visual HUD.
    """
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm base
    ]

    def __init__(self):
        self.cap = None
        self.available = True
        self.error_message = None
        self.detector = None
        self.use_tasks_api = False
        self.smoothed_angles = DEFAULT_ANGLES.copy()
        
        self._init_mediapipe()

    def _ensure_task_model(self):
        """Downloads the MediaPipe hand landmarker task model if missing."""
        if not os.path.exists(TASK_MODEL_PATH):
            try:
                urllib.request.urlretrieve(TASK_MODEL_URL, TASK_MODEL_PATH)
            except Exception as e:
                self.error_message = f"Failed to download task model: {e}"

    def _init_mediapipe(self):
        """Initializes the hand tracking backend with robust fallback."""
        try:
            import mediapipe as mp
            if hasattr(mp, 'tasks'):
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                
                self._ensure_task_model()
                base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.6,
                    min_hand_presence_confidence=0.6,
                    min_tracking_confidence=0.5
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.use_tasks_api = True
            elif hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands'):
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.5
                )
                self.use_tasks_api = False
            else:
                self.available = False
                self.error_message = "MediaPipe hand solution not found in installed package."
        except Exception as e:
            self.available = False
            self.error_message = f"MediaPipe initialization error: {e}"

    def open_camera(self, camera_idx=0):
        """Safely opens the webcam video capture."""
        if self.cap is not None and self.cap.isOpened():
            return True
        try:
            # DirectShow on Windows for fast instant startup
            self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(camera_idx)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                return True
            else:
                self.error_message = f"Cannot open Camera Index {camera_idx}"
                return False
        except Exception as e:
            self.error_message = str(e)
            return False

    def close_camera(self):
        """Safely releases the webcam."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def process_frame(self, frame, smoothing_alpha=0.35, invert_x=False, invert_y=False):
        """
        Processes a single video frame, extracts landmarks, computes joint angles,
        and renders a cyberpunk HUD overlay.
        """
        # Horizontal flip for natural mirror feedback
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        raw_angles = None
        landmarks_norm = []

        # Hand Detection
        if self.use_tasks_api and self.detector is not None:
            import mediapipe as mp
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks:
                hand = results.hand_landmarks[0]
                landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand]
        elif hasattr(self, 'hands') and self.hands is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                landmarks_norm = [(lm.x, lm.y, lm.z) for lm in hand.landmark]

        # Landmark Processing and HUD Overlay
        if landmarks_norm and len(landmarks_norm) >= 21:
            raw_angles, pinch_ratio = self._calculate_joint_angles(landmarks_norm, invert_x, invert_y)
            
            # Exponential Moving Average (EMA) smoothing
            self.smoothed_angles = (smoothing_alpha * raw_angles + 
                                    (1.0 - smoothing_alpha) * self.smoothed_angles)
            
            # Draw Hand Skeleton & HUD
            self._render_hud(frame, landmarks_norm, self.smoothed_angles, pinch_ratio, h, w)
        else:
            # Standby Overlay
            cv2.putText(frame, "NO HAND DETECTED - SHOW HAND TO CAMERA", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

        return frame, (self.smoothed_angles.copy() if landmarks_norm else None)

    def _calculate_joint_angles(self, lms, invert_x=False, invert_y=False):
        """
        Scale-invariant mathematical mapping from 3D hand landmarks to 6-DOF robot angles.
        """
        wrist     = np.array(lms[0])
        thumb_tip = np.array(lms[4])
        index_mcp = np.array(lms[5])
        index_tip = np.array(lms[8])
        mid_mcp   = np.array(lms[9])
        mid_tip   = np.array(lms[12])
        pinky_mcp = np.array(lms[17])

        # Scale factor (palm length) to make gestures distance-invariant
        palm_size = np.linalg.norm(mid_mcp[:2] - wrist[:2])
        if palm_size < 1e-4:
            palm_size = 0.2

        # 1. Waist (J0): Hand X Position [0° - 180°]
        x_norm = np.clip((wrist[0] - 0.20) / (0.80 - 0.20), 0.0, 1.0)
        if invert_x:
            x_norm = 1.0 - x_norm
        waist = x_norm * 180.0

        # 2. Shoulder (J1): Hand Y Position [0° - 180°]
        y_norm = np.clip((wrist[1] - 0.20) / (0.80 - 0.20), 0.0, 1.0)
        shoulder = y_norm * 180.0 if invert_y else (1.0 - y_norm) * 180.0

        # 3. Elbow (J2): Hand Extension / Reach [0° - 180°]
        reach = np.linalg.norm(mid_tip[:2] - wrist[:2]) / palm_size
        elbow = np.clip((reach - 1.0) / (2.1 - 1.0) * 180.0, 0.0, 180.0)

        # 4. Wrist Pitch (J3): Vertical hand tilt [0° - 180°]
        dx = mid_mcp[0] - wrist[0]
        dy = mid_mcp[1] - wrist[1]
        pitch_angle = np.degrees(np.arctan2(-dy, dx))
        pitch = np.clip(pitch_angle, 0.0, 180.0)

        # 5. Wrist Roll (J4): Knuckle line roll [0° - 180°]
        dx_roll = pinky_mcp[0] - index_mcp[0]
        dy_roll = pinky_mcp[1] - index_mcp[1]
        roll_angle = np.degrees(np.arctan2(dy_roll, dx_roll))
        roll = np.clip(90.0 + roll_angle, 0.0, 180.0)

        # 6. Gripper (J5): Pinch distance between Thumb & Index Tips [0° - 90°]
        pinch_dist = np.linalg.norm(thumb_tip[:2] - index_tip[:2])
        pinch_ratio = pinch_dist / palm_size
        # ratio < 0.25 -> Closed (0°), ratio > 0.80 -> Open (90°)
        gripper = np.clip((pinch_ratio - 0.25) / (0.80 - 0.25) * 90.0, 0.0, 90.0)

        angles = np.array([waist, shoulder, elbow, pitch, roll, gripper], dtype=np.float64)
        return angles, pinch_ratio

    def _render_hud(self, frame, lms, angles, pinch_ratio, h, w):
        """Draws the Cyberpunk-styled HUD and landmarks onto the camera feed."""
        coords = [(int(lm[0] * w), int(lm[1] * h)) for lm in lms]

        # Draw hand bones
        for p1_idx, p2_idx in self.HAND_CONNECTIONS:
            cv2.line(frame, coords[p1_idx], coords[p2_idx], (255, 140, 0), 2, cv2.LINE_AA)

        # Draw landmark nodes
        for idx, pt in enumerate(coords):
            color = (0, 255, 255) if idx in [4, 8, 12, 16, 20] else (50, 205, 50)
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)

        # Draw Pinch Line between Thumb & Index Tip
        p_thumb = coords[4]
        p_index = coords[8]
        is_pinched = (angles[5] < 20.0)
        pinch_color = (0, 0, 255) if is_pinched else (0, 255, 128)
        cv2.line(frame, p_thumb, p_index, pinch_color, 3, cv2.LINE_AA)
        
        # Center of pinch
        mid_pinch = ((p_thumb[0] + p_index[0]) // 2, (p_thumb[1] + p_index[1]) // 2)
        cv2.circle(frame, mid_pinch, 6, pinch_color, -1, cv2.LINE_AA)

        # HUD Top Banner
        cv2.rectangle(frame, (0, 0), (w, 40), (15, 23, 42), -1)
        cv2.putText(frame, "AI GESTURE TRACKING ACTIVE", (15, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2, cv2.LINE_AA)
        
        pinch_label = "GRIPPER: PINCHED [CLOSED]" if is_pinched else f"GRIPPER: OPEN ({int(angles[5])}°)"
        cv2.putText(frame, pinch_label, (w - 280, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, pinch_color, 2, cv2.LINE_AA)

        # HUD Bottom Panel
        cv2.rectangle(frame, (0, h - 35), (w, h), (15, 23, 42), -1)
        hud_text = (f"W:{int(angles[0])}° | S:{int(angles[1])}° | E:{int(angles[2])}° | "
                    f"P:{int(angles[3])}° | R:{int(angles[4])}° | G:{int(angles[5])}°")
        cv2.putText(frame, hud_text, (15, h - 12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 232, 240), 1, cv2.LINE_AA)


# ==========================================
# GEMINI AI VOICE CONTROLLER
# ==========================================
class GeminiVoiceController:
    def __init__(self):
        self.model = None
        self.available = False
        self.error_message = None
        self.speech_recognizer = None
        self.command_history = []
        self.max_history = 15
        
        if SPEECH_AVAILABLE:
            self.speech_recognizer = sr.Recognizer()
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        if not GEMINI_AVAILABLE:
            self.error_message = "Install: pip install google-generativeai"
            return
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.error_message = "GEMINI_API_KEY environment variable not set"
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            self.available = True
        except Exception as e:
            self.error_message = f"Gemini Init Error: {e}"
            self.available = False
    
    def get_system_prompt(self) -> str:
        return """You are a robotic arm command interpreter. Convert natural language into strict JSON.
JOINTS: waist (0-180), shoulder (0-180), elbow (0-180), wrist_pitch (0-180), wrist_roll (0-180), gripper (0-90).
ACTIONS:
- "move": relative movement (joint, direction ["up","down","left","right"], degrees [default 10])
- "set": absolute angle (joint, angle)
- "reset": reset home pose
- "gripper_open": open gripper
- "gripper_close": close gripper

OUTPUT ONLY VALID JSON:
Example: {"action": "move", "joint": "shoulder", "direction": "up", "degrees": 20}
Example: {"action": "gripper_close"}
"""
    
    def listen_and_execute(self, session_state) -> dict:
        result = {"success": False, "transcript": None, "parsed_command": None, "error": None}
        if not SPEECH_AVAILABLE or not self.speech_recognizer:
            result["error"] = "SpeechRecognition or Microphone unavailable"
            return result
        
        try:
            with sr.Microphone() as source:
                self.speech_recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = self.speech_recognizer.listen(source, timeout=4, phrase_time_limit=6)
            transcript = self.speech_recognizer.recognize_google(audio).lower().strip()
            result["transcript"] = transcript
        except Exception as e:
            result["error"] = f"Voice recognition failed: {e}"
            return result
        
        if not self.available or not self.model:
            result["error"] = self.error_message or "Gemini not available"
            return result
        
        try:
            prompt = f"{self.get_system_prompt()}\n\nUser Command: \"{transcript}\""
            response = self.model.generate_content(prompt)
            clean_text = re.sub(r'```json\s*|\s*```', '', response.text).strip()
            parsed = json.loads(clean_text)
            result["parsed_command"] = parsed
            result["success"] = self._apply_command(parsed, session_state)
            self._add_to_history(transcript, parsed, result["success"])
        except Exception as e:
            result["error"] = f"Gemini interpretation failed: {e}"
        return result

    def _apply_command(self, cmd: dict, session_state) -> bool:
        try:
            action = cmd.get("action")
            curr = session_state.current_angles.copy().astype(float)
            jmap = {"waist": 0, "shoulder": 1, "elbow": 2, "wrist_pitch": 3, "wrist_roll": 4, "gripper": 5}
            
            if action == "move":
                jname = cmd.get("joint", "").lower()
                if jname not in jmap: return False
                idx = jmap[jname]
                deg = cmd.get("degrees", 10)
                dir_ = cmd.get("direction", "").lower()
                delta = deg if dir_ in ["up", "right"] else -deg
                curr[idx] = np.clip(curr[idx] + delta, JOINT_LIMITS[idx][0], JOINT_LIMITS[idx][1])
            elif action == "set":
                jname = cmd.get("joint", "").lower()
                if jname not in jmap or "angle" not in cmd: return False
                idx = jmap[jname]
                curr[idx] = np.clip(cmd["angle"], JOINT_LIMITS[idx][0], JOINT_LIMITS[idx][1])
            elif action == "reset":
                curr[:] = DEFAULT_ANGLES
            elif action == "gripper_open":
                curr[5] = 90
            elif action == "gripper_close":
                curr[5] = 0
            else:
                return False

            session_state.current_angles = curr
            compute_velocities(session_state.current_angles.copy())
            if session_state.robot.connected:
                session_state.robot.send_command(session_state.current_angles)
            return True
        except Exception:
            return False

    def _add_to_history(self, transcript, command, success):
        self.command_history.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "transcript": transcript,
            "command": str(command),
            "status": "✅" if success else "❌"
        })
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)


# ==========================================
# VELOCITY COMPUTATION
# ==========================================
def compute_velocities(current_angles: np.ndarray) -> np.ndarray:
    now = time.monotonic()
    dt  = max(now - st.session_state.prev_timestamp, MIN_DT)

    delta_angles = current_angles.astype(float) - st.session_state.prev_angles.astype(float)
    raw_velocities = delta_angles / dt
    
    if not hasattr(st.session_state, 'prev_velocities'):
        st.session_state.prev_velocities = np.zeros(6, dtype=np.float64)
    
    filtered_velocities = (VELOCITY_ALPHA * raw_velocities + 
                          (1.0 - VELOCITY_ALPHA) * st.session_state.prev_velocities)
    clamped_velocities = np.clip(filtered_velocities, -VELOCITY_CLAMP, VELOCITY_CLAMP)
    
    st.session_state.prev_angles = current_angles.copy()
    st.session_state.prev_timestamp = now
    st.session_state.prev_velocities = clamped_velocities.copy()
    st.session_state.joint_velocities = clamped_velocities
    return clamped_velocities


# ==========================================
# SLIDER CALLBACK
# ==========================================
def slider_changed(joint_idx: int, slider_key: str):
    new_val = st.session_state[slider_key]
    st.session_state.current_angles[joint_idx] = new_val
    compute_velocities(st.session_state.current_angles.copy())


# ==========================================
# 3D VISUALIZATION HELPERS
# ==========================================
def create_3d_arm_visualization(joint_angles_deg, kinematics):
    _, transforms = kinematics.forward_kinematics(joint_angles_deg)
    fig = go.Figure()
    
    positions = [np.array([0., 0., 0.])]
    for t in transforms:
        positions.append(t[:3, 3].copy())
    
    # Floor Grid
    for xs, ys, zs in [([-0.4, 0.4], [0, 0], [0, 0]), ([0, 0], [-0.4, 0.4], [0, 0])]:
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                   line=dict(color='#1f293d', width=2, dash='dash'),
                                   showlegend=False))
    
    # Base Column
    d1 = kinematics.dh_params[0, 1]
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, d1],
                               mode='lines',
                               line=dict(color='#64748b', width=14),
                               name='Base Link'))
    
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#38bdf8']
    names  = ['Waist', 'Shoulder', 'Elbow', 'Wrist Pitch', 'Wrist Roll', 'Gripper']
    
    for i in range(len(positions) - 1):
        p0, p1 = positions[i], positions[i + 1]
        if np.linalg.norm(p1 - p0) < 1e-6:
            continue
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode='lines', line=dict(color=colors[i], width=11),
            name=names[i], showlegend=True))
        fig.add_trace(go.Scatter3d(
            x=[p0[0]], y=[p0[1]], z=[p0[2]],
            mode='markers', marker=dict(size=7, color='#f8fafc'),
            showlegend=False))
    
    fig.add_trace(go.Scatter3d(
        x=[positions[-1][0]], y=[positions[-1][1]], z=[positions[-1][2]],
        mode='markers',
        marker=dict(size=9, color='#fbbf24', symbol='diamond'),
        name='End-Effector'))
    
    r = 0.45
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[-r, r]),
            yaxis=dict(title='Y', range=[-r, r]),
            zaxis=dict(title='Z', range=[0, r]),
            bgcolor='#0b0f19',
            aspectmode='cube'),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#111827',
        height=480,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig


def create_workspace_visualization(current_pos):
    fig = go.Figure()
    for xs, ys, zs in [([-0.4, 0.4], [0, 0], [0, 0]), ([0, 0], [-0.4, 0.4], [0, 0])]:
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                   line=dict(color='#3b82f6', width=2, dash='dot'),
                                   showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0.08], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='#ef4444', width=4), name='X Axis'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0.08], z=[0, 0], mode='lines', line=dict(color='#10b981', width=4), name='Y Axis'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 0.08], mode='lines', line=dict(color='#3b82f6', width=4), name='Z Axis'))
    fig.add_trace(go.Scatter3d(
        x=[current_pos[0]], y=[current_pos[1]], z=[current_pos[2]],
        mode='markers', marker=dict(size=8, color='#fbbf24', symbol='diamond'),
        name='Current Pose'))
    fig.update_layout(
        scene=dict(xaxis=dict(range=[-0.4, 0.4]), yaxis=dict(range=[-0.4, 0.4]), zaxis=dict(range=[0, 0.45]),
                   bgcolor='#0b0f19', aspectmode='cube'),
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='#111827', height=280,
        legend=dict(orientation='h', xanchor='center', x=0.5))
    return fig


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state():
    if 'robot' not in st.session_state:
        st.session_state.robot = RobotController()
    if 'voice_ctrl' not in st.session_state:
        st.session_state.voice_ctrl = GeminiVoiceController()
    if 'gesture_ctrl' not in st.session_state:
        st.session_state.gesture_ctrl = GestureController()

    if 'current_angles' not in st.session_state:
        st.session_state.current_angles = DEFAULT_ANGLES.copy()
    if 'prev_angles' not in st.session_state:
        st.session_state.prev_angles = DEFAULT_ANGLES.copy()
    if 'prev_timestamp' not in st.session_state:
        st.session_state.prev_timestamp = time.monotonic()
    if 'joint_velocities' not in st.session_state:
        st.session_state.joint_velocities = np.zeros(6, dtype=np.float64)
    if 'prev_velocities' not in st.session_state:
        st.session_state.prev_velocities = np.zeros(6, dtype=np.float64)
    if 'control_mode' not in st.session_state:
        st.session_state.control_mode = "Manual Control"
    if 'gesture_active' not in st.session_state:
        st.session_state.gesture_active = False


# ==========================================
# MAIN APPLICATION
# ==========================================
def main():
    init_session_state()

    kin    = st.session_state.robot.kinematics
    angles = st.session_state.current_angles.astype(float)

    # Kinematic Calculations
    pose     = kin.get_end_effector_pose(angles)
    jacobian = kin.calculate_jacobian(angles)
    vel_data = kin.calculate_end_effector_velocity(angles, st.session_state.joint_velocities)
    manip    = kin.calculate_manipulability(angles)

    # Header Bar
    header_cols = st.columns([1, 6, 1, 2])
    with header_cols[1]:
        st.markdown(
            "<h1 style='margin:0; font-size:1.85rem;'>🦾 6-DOF Robotic Arm Control "
            "<span style='font-size:0.55em; color:#60a5fa; font-weight:400;'>V3 — Gesture & AI Suite</span></h1>",
            unsafe_allow_html=True)
    with header_cols[3]:
        if st.session_state.robot.connected:
            st.markdown("<div style='text-align:right; color:#10b981; font-weight:700;'>🟢 ESP32 Connected</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align:right; color:#ef4444; font-weight:700;'>🔴 Disconnected</div>",
                        unsafe_allow_html=True)
    st.divider()

    # Mode Selector
    st.session_state.control_mode = st.radio(
        "Control Mode", ["Manual Control", "Gesture Control", "Voice Control"],
        horizontal=True, label_visibility="collapsed")

    col_left, col_right = st.columns([1.4, 1.1])

    # ── LEFT PANEL: 3D SIMULATION & TELEMETRY ────────────────────────────────
    with col_left:
        st.markdown("<div class='bento-container'>", unsafe_allow_html=True)
        tab_sim, tab_model = st.tabs(["📊 Real-Time Kinematic Simulation", "🎨 3D GLB Model & Reach"])

        with tab_sim:
            st.plotly_chart(create_3d_arm_visualization(angles, kin), use_container_width=True)

            # Live Metrics
            st.markdown("#### 🔬 Telemetry & Velocity Engine")
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.markdown(
                    f"<div class='metric-card'><b>End-Effector Pos (m)</b><br>"
                    f"X: {pose['position'][0]:.3f}<br>"
                    f"Y: {pose['position'][1]:.3f}<br>"
                    f"Z: {pose['position'][2]:.3f}</div>",
                    unsafe_allow_html=True)

            with c2:
                v_class = "warning" if vel_data['linear_speed'] > 0.4 else "success"
                vel_badges = "".join(
                    f"<span class='vel-badge'>{JOINT_NAMES[i][0]}: {st.session_state.joint_velocities[i]:+.1f}°/s</span>"
                    for i in range(6))
                st.markdown(
                    f"<div class='metric-card {v_class}'><b>EE Velocity (Auto)</b><br>"
                    f"Linear: {vel_data['linear_speed']:.3f} m/s<br>"
                    f"Angular: {vel_data['angular_speed']:.3f} rad/s<br>"
                    f"{vel_badges}</div>",
                    unsafe_allow_html=True)

            with c3:
                m_class = "danger" if manip['is_singular'] else ("warning" if manip['condition_number'] > 50 else "success")
                st.markdown(
                    f"<div class='metric-card {m_class}'><b>Manipulability</b><br>"
                    f"Index: {manip['manipulability']:.4f}<br>"
                    f"Condition: {manip['condition_number']:.1f}</div>",
                    unsafe_allow_html=True)

            with c4:
                s_status = "⚠️ Near Singular" if manip['is_singular'] else "✅ Optimal Dexterity"
                s_class = "danger" if manip['is_singular'] else "success"
                st.markdown(
                    f"<div class='metric-card {s_class}'><b>Singularity State</b><br>"
                    f"{s_status}</div>",
                    unsafe_allow_html=True)

            # Orientation
            ori = pose['orientation']
            st.markdown(
                f"<div class='metric-card'><b>End-Effector Euler Orientation:</b>"
                f"&nbsp;&nbsp; Roll: <b>{ori[0]:.1f}°</b> &nbsp;|&nbsp; Pitch: <b>{ori[1]:.1f}°</b> &nbsp;|&nbsp; Yaw: <b>{ori[2]:.1f}°</b></div>",
                unsafe_allow_html=True)

            # Matrices Accordion
            with st.expander("🧮 View Transformation & Jacobian Matrices (Live Update)"):
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.markdown("**Homogeneous Transformation Matrix (T)**")
                    T_df = pd.DataFrame(pose['T'], columns=['X', 'Y', 'Z', 'W'], index=['X', 'Y', 'Z', 'Base'])
                    st.dataframe(T_df.style.format("{:.4f}"), use_container_width=True)
                with mc2:
                    st.markdown("**Jacobian Matrix (J)**")
                    J_df = pd.DataFrame(jacobian, columns=['J1', 'J2', 'J3', 'J4', 'J5', 'J6'],
                                        index=['vₓ', 'vᵧ', 'v_z', 'ωₓ', 'ωᵧ', 'ω_z'])
                    st.dataframe(J_df.style.format("{:.4f}"), use_container_width=True)

        with tab_model:
            st.components.v1.html(render_glb_model(GLB_PATH), height=510)
            st.markdown("### 3D Workspace Range")
            st.plotly_chart(create_workspace_visualization(pose['position']), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT PANEL: CONTROL MODES ───────────────────────────────────────────
    with col_right:
        st.markdown("<div class='bento-container'>", unsafe_allow_html=True)

        # ── MODE 1: MANUAL CONTROL ───────────────────────────────────────────
        if st.session_state.control_mode == "Manual Control":
            st.subheader("🎮 Manual Joint Control")
            st.markdown("<small style='color:#94a3b8;'>Joint velocities are calculated automatically in real time.</small>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            for i, (name, lim) in enumerate(zip(JOINT_NAMES, JOINT_LIMITS)):
                slider_key = f"sl_{i}"
                st.markdown(
                    f'<div class="joint-label">'
                    f'<span>{name} ({lim[0]}° - {lim[1]}°)</span>'
                    f'<span class="joint-value">{int(st.session_state.current_angles[i])}°</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.slider(
                    label=name, min_value=lim[0], max_value=lim[1],
                    value=int(st.session_state.current_angles[i]),
                    key=slider_key, on_change=slider_changed, args=(i, slider_key),
                    label_visibility="collapsed"
                )

            auto_send = st.checkbox("🔗 Auto-send commands to hardware", value=True, key="auto_send")
            if auto_send and st.session_state.robot.connected:
                st.session_state.robot.send_command(st.session_state.current_angles)

            st.markdown("<br>", unsafe_allow_html=True)
            btn1, btn2, btn3 = st.columns(3)
            with btn1:
                if st.button("💾 SAVE POSE"):
                    st.session_state.robot.sequence.append({
                        'angles': st.session_state.current_angles.copy(),
                        'position': pose['position'].copy()
                    })
                    st.toast(f"✅ Saved waypoint #{len(st.session_state.robot.sequence)}")
            with btn2:
                if st.button("▶ RUN SEQ"):
                    if st.session_state.robot.connected:
                        st.toast("🔄 Replaying trajectory sequence...")
                    else:
                        st.warning("⚠️ Connect ESP32 first")
            with btn3:
                if st.button("🔄 HOME RESET"):
                    if st.session_state.robot.connected:
                        st.session_state.robot.send_reset()
                    st.session_state.current_angles   = DEFAULT_ANGLES.copy()
                    st.session_state.prev_angles      = DEFAULT_ANGLES.copy()
                    st.session_state.prev_timestamp   = time.monotonic()
                    st.session_state.joint_velocities = np.zeros(6, dtype=np.float64)
                    st.rerun()

        # ── MODE 2: GESTURE CONTROL (FULL IMPLEMENTATION) ────────────────────
        elif st.session_state.control_mode == "Gesture Control":
            st.subheader("✋ Real-Time Gesture AI Control")
            gc = st.session_state.gesture_ctrl

            if not gc.available:
                st.error(f"⚠️ Gesture Tracking Unavailable: {gc.error_message}")
                st.info("Tip: Make sure OpenCV and MediaPipe are installed and your webcam is connected.")
            else:
                st.markdown("""
                **Gesture Mappings:**
                - **Hand X (Left ↔ Right)** → Waist (Base)
                - **Hand Y (Up ↕ Down)** → Shoulder
                - **Hand Extension (Reach)** → Elbow
                - **Hand Tilt & Roll** → Wrist Pitch & Roll
                - **Pinch (Thumb + Index)** → Gripper Close/Open
                """)

                # Settings Bar
                set_c1, set_c2 = st.columns(2)
                with set_c1:
                    smoothing = st.slider("Smoothing Filter (EMA)", min_value=0.1, max_value=0.9, value=0.35, step=0.05)
                with set_c2:
                    cam_idx = st.selectbox("Camera Device", [0, 1, 2], index=0)

                opt_c1, opt_c2, opt_c3 = st.columns(3)
                with opt_c1:
                    inv_x = st.checkbox("Invert X (Waist)", value=False)
                with opt_c2:
                    inv_y = st.checkbox("Invert Y (Shoulder)", value=False)
                with opt_c3:
                    sync_hw = st.checkbox("Live Serial Sync", value=True)

                st.markdown("<br>", unsafe_allow_html=True)
                ctrl_c1, ctrl_c2 = st.columns(2)
                
                with ctrl_c1:
                    start_clicked = st.button("▶ START CAMERA TRACKING", type="primary", use_container_width=True)
                with ctrl_c2:
                    stop_clicked = st.button("⏹ STOP CAMERA", type="secondary", use_container_width=True)

                if start_clicked:
                    st.session_state.gesture_active = True
                if stop_clicked:
                    st.session_state.gesture_active = False
                    gc.close_camera()

                # Live Video Loop Container
                video_placeholder = st.empty()
                status_placeholder = st.empty()

                if st.session_state.gesture_active:
                    if gc.open_camera(cam_idx):
                        status_placeholder.markdown("<div class='voice-status voice-listening'>🟢 Camera Active — Tracking Hand Gestures</div>", unsafe_allow_html=True)
                        
                        # Streaming Loop
                        while st.session_state.gesture_active:
                            ret, frame = gc.cap.read()
                            if not ret:
                                status_placeholder.error("⚠️ Lost camera feed.")
                                st.session_state.gesture_active = False
                                gc.close_camera()
                                break

                            processed_frame, live_angles = gc.process_frame(
                                frame, smoothing_alpha=smoothing, invert_x=inv_x, invert_y=inv_y)

                            # Display annotated HUD video frame
                            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

                            # Update robot state & send to hardware
                            if live_angles is not None:
                                st.session_state.current_angles = live_angles.copy()
                                compute_velocities(st.session_state.current_angles.copy())

                                if sync_hw and st.session_state.robot.connected:
                                    st.session_state.robot.send_command(st.session_state.current_angles)

                            time.sleep(0.01) # Yield CPU
                    else:
                        st.error(f"❌ Failed to open camera {cam_idx}. {gc.error_message}")
                        st.session_state.gesture_active = False

        # ── MODE 3: GEMINI VOICE CONTROL ─────────────────────────────────────
        elif st.session_state.control_mode == "Voice Control":
            st.subheader("🎤 Gemini Natural Language Control")
            v_ctrl = st.session_state.voice_ctrl

            if not v_ctrl.available:
                st.error(f"⚠️ {v_ctrl.error_message}")
                st.info("""
                **Quick Setup:**
                1. Set API key: `$env:GEMINI_API_KEY="your-api-key"` (PowerShell)
                2. Install: `pip install google-generativeai SpeechRecognition pyaudio`
                3. Restart Streamlit
                """)
            else:
                st.success("✅ Gemini Voice AI Engine Ready")
                st.markdown("""
                **Try Speaking Naturally:**
                - *"Move shoulder up by 25 degrees"*
                - *"Rotate waist left"*
                - *"Set elbow to 120 degrees"*
                - *"Close the gripper"* / *"Open gripper"*
                - *"Reset robot position"*
                """)

                v_status = st.empty()
                if st.button("🎤 LISTEN & EXECUTE COMMAND", type="primary", use_container_width=True):
                    v_status.markdown("<div class='voice-status voice-listening'>🎤 Listening... Speak command now</div>", unsafe_allow_html=True)
                    res = v_ctrl.listen_and_execute(st.session_state)
                    v_status.empty()

                    if res["success"]:
                        st.success(f"✅ Executed: \"{res['transcript']}\"")
                        st.json(res["parsed_command"])
                        st.rerun()
                    else:
                        st.error(f"❌ {res['error']}")
                        if res["transcript"]:
                            st.warning(f"Heard: \"{res['transcript']}\"")

                # Command History
                st.divider()
                st.markdown("#### 📜 Voice Command History")
                if v_ctrl.command_history:
                    st.dataframe(pd.DataFrame(v_ctrl.command_history[::-1]), use_container_width=True, hide_index=True)
                else:
                    st.caption("No commands recorded yet.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── BOTTOM PANEL: HARDWARE & INVERSE KINEMATICS ──────────────────────────
    st.divider()
    b_c1, b_c2, b_c3 = st.columns(3)

    with b_c1:
        st.markdown("<div class='bento-container'>", unsafe_allow_html=True)
        st.subheader("🔌 Serial Hardware Link")
        ports = [p.device for p in serial.tools.list_ports.comports()]
        sel_port = st.selectbox("Select COM Port", ports if ports else ["No Ports Found"])

        if st.session_state.robot.connected:
            if st.button("🔴 Disconnect ESP32", type="primary"):
                st.session_state.robot.disconnect()
                st.rerun()
        else:
            if st.button("🟢 Connect to ESP32", type="primary"):
                if st.session_state.robot.connect(sel_port, baud=115200):
                    st.success("✅ Connected to ESP32!")
                    st.session_state.robot.send_command(st.session_state.current_angles)
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with b_c2:
        st.markdown("<div class='bento-container'>", unsafe_allow_html=True)
        st.subheader("🎯 3D Inverse Kinematics")
        tx, ty, tz = st.columns(3)
        tgt_x = tx.number_input("X (m)", value=0.25, step=0.01)
        tgt_y = ty.number_input("Y (m)", value=0.0, step=0.01)
        tgt_z = tz.number_input("Z (m)", value=0.20, step=0.01)

        if st.button("⚡ Solve & Move to Pose", type="primary"):
            try:
                tgt_pos = np.array([tgt_x, tgt_y, tgt_z], dtype=np.float64)
                tgt_ori = np.array([0, 90, 0], dtype=np.float64)
                sol = kin.inverse_kinematics(tgt_pos, tgt_ori, st.session_state.current_angles)

                if sol['converged']:
                    st.session_state.current_angles = sol['angles'].astype(np.float64)
                    compute_velocities(st.session_state.current_angles.copy())
                    st.success(f"✅ Solved in {sol['iterations']} iterations")
                    if st.session_state.robot.connected:
                        st.session_state.robot.send_command(st.session_state.current_angles)
                    st.rerun()
                else:
                    st.error("❌ Target position unreachable or out of workspace bounds")
            except Exception as e:
                st.error(f"❌ IK Calculation Error: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with b_c3:
        st.markdown("<div class='bento-container'>", unsafe_allow_html=True)
        st.subheader("📋 Trajectory Sequence")
        st.metric("Stored Waypoints", len(st.session_state.robot.sequence))
        if st.session_state.robot.sequence:
            for i, p in enumerate(st.session_state.robot.sequence[-4:]):
                st.text(f"#{len(st.session_state.robot.sequence)-3+i}: Angles: {np.round(p['angles'], 1)}")
        else:
            st.caption("No waypoints saved yet. Use 'SAVE POSE' in manual mode.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
