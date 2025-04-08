import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
from tkbootstrap.dialogs import Querybox
import mediapipe as mp
import pyttsx3
import time
import threading
import numpy as np
import sys
import subprocess

# TensorFlow import with auto-install
try:
    from tensorflow.keras.models import load_model
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    from tensorflow.keras.models import load_model

# Initialize main window
win = ttk.Window(themename="darkly")
width = win.winfo_screenwidth()
height = win.winfo_screenheight()
win.geometry(f"{width}x{height}")
win.title('Sign Language Translator')

# Global variables
global cap, label1, upCount, running, gesture_buffer
cap = None
running = False
gesture_buffer = []
upCount = ttk.StringVar()
upCount.set("No hand detected")

# MediaPipe configuration
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load model and class names
try:
    model = load_model('gesture_model.h5')
    class_names = np.load('class_names.npy', allow_pickle=True)
    model_loaded = True
except Exception as e:
    print(f"Model loading error: {e}")
    model_loaded = False

# UI Components
header_frame = ttk.Frame(win)
header_frame.pack(pady=15, padx=300, fill=X)
ttk.Label(header_frame, text='Sign Language Translator', font=('Helvetica', 24, 'bold'), bootstyle="light").pack(pady=10)

video_frame = ttk.Frame(win)
video_frame.pack(pady=20)

webcam_container = ttk.Frame(video_frame, width=640, height=480)
webcam_container.pack_propagate(False)
webcam_container.pack(side=LEFT, padx=20)

control_frame = ttk.Frame(video_frame)
control_frame.pack(side=RIGHT, padx=20, fill=Y)

# Gesture processing functions
def preprocess_hand_region(hand_roi):
    img = cv2.resize(hand_roi, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.expand_dims(img / 255.0, axis=0)

def classify_gesture(hand_roi):
    global gesture_buffer
    if not model_loaded or hand_roi.size == 0:
        return "Model not loaded", 0
    
    processed = preprocess_hand_region(hand_roi)
    prediction = model.predict(processed)
    class_index = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    gesture_buffer.append(class_names[class_index])
    if len(gesture_buffer) > 5:
        gesture_buffer.pop(0)
    
    return max(set(gesture_buffer), key=gesture_buffer.count), confidence

# Video processing thread
def process_frames():
    global cap, running
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        gesture = "No hand detected"
        confidence = 0
        
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                x_min = int(min(lm.x * w for lm in landmarks.landmark))
                y_min = int(min(lm.y * h for lm in landmarks.landmark))
                x_max = int(max(lm.x * w for lm in landmarks.landmark))
                y_max = int(max(lm.y * h for lm in landmarks.landmark))
                
                if x_max > x_min and y_max > y_min:
                    hand_roi = frame[y_min:y_max, x_min:x_max]
                    gesture, confidence = classify_gesture(hand_roi)
                    cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (0,255,0), 2)
        
        win.after(0, update_ui, frame, gesture, confidence)
        time.sleep(0.03)

# UI update function
def update_ui(frame, gesture, confidence):
    global label1
    upCount.set(gesture)
    
    try:
        img = Image.fromarray(frame)
        img = img.resize((640, 480), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        label1.imgtk = imgtk
        label1.configure(image=imgtk)
    except Exception as e:
        print(f"UI update error: {e}")

# Webcam control functions
def init_webcam():
    global cap, label1, running
    stop_webcam()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    label1 = ttk.Label(webcam_container)
    label1.pack(fill=BOTH, expand=True)
    
    running = True
    threading.Thread(target=process_frames, daemon=True).start()

def stop_webcam():
    global cap, running
    running = False
    if cap:
        cap.release()

# Text-to-speech
def voice():
    text = upCount.get()
    if text not in ["No hand detected", "Model not loaded"]:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

# GUI components
result_frame = ttk.Frame(win)
result_frame.pack(pady=20, fill=X)

confidence_meter = ttk.Meter(
    result_frame,
    bootstyle="success",
    amountused=0,
    metersize=180,
    subtext="Confidence",
    interactive=False
)
confidence_meter.pack(pady=10)

ttk.Button(control_frame, text='Live Camera', bootstyle="primary", 
          command=init_webcam).pack(pady=10)
ttk.Button(control_frame, text='Speak Result', bootstyle="success",
          command=voice).pack(pady=10)
ttk.Button(control_frame, text='Exit', bootstyle="danger",
          command=lambda: [stop_webcam(), win.destroy()]).pack(pady=10)

# Initialize
init_webcam()
win.mainloop()
