import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import os
import time
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from urdu_tts import speak_urdu

# ==============================================================
# ISHARALINK - REAL-TIME PSL TO URDU WORD & SPEECH TRANSLATOR
# ==============================================================

MODEL_PATH = "isharalink_landmarks_model.keras"
LABELS_PATH = "class_labels.json"
FONT_PATH = "C:/Windows/Fonts/segoeui.ttf"

print("\n" + "="*65)
print("     ISHARALINK - URDU WORD & SENTENCE FORMATION SYSTEM")
print("="*65)

if not os.path.exists(MODEL_PATH):
    print(f"[Error] Model {MODEL_PATH} not found!")
    raise SystemExit

# 1. Load Landmark Model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# 2. Load Class Labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)

# Extract only the Urdu letter/character (e.g. "Ain ع" -> "ع", "Aliph ا" -> "ا")
urdu_chars = {}
for k, v in class_labels.items():
    parts = v.strip().split()
    urdu_part = parts[-1] if len(parts) > 1 else v
    urdu_chars[k] = urdu_part

# 3. MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

def extract_normalized_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    rel_coords = coords - wrist
    palm_size = np.linalg.norm(coords[9] - wrist)
    if palm_size > 0:
        rel_coords = rel_coords / palm_size
    return rel_coords.flatten()

def render_urdu_text(frame, text, pos, font_size=28, color=(255, 255, 255)):
    """Draws beautifully joined and reshaped Urdu ligatures on OpenCV frame"""
    if not text:
        return frame
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        font = ImageFont.truetype(FONT_PATH, font_size)
        draw.text(pos, bidi_text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as err:
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

# 4. Camera Setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Error] Camera could not be opened!")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Sentence and Word Building State
current_letters = []  # List of letters and spaces forming the sentence
prediction_history = deque(maxlen=6)

# Auto-hold timer for steady sign entry
HOLD_THRESHOLD_SEC = 1.3
hold_start_time = None
last_hold_char = None
last_added_char = None

speaking_status = ""
speaking_time = 0

# Quick Domain Presets
DOMAIN_PRESETS = {
    ord('1'): "مجھے ڈاکٹر سے ملنا ہے",
    ord('2'): "مجھے مدد چاہیے",
    ord('3'): "شدید درد ہو رہا ہے",
    ord('4'): "دوائی لے کر آئیں",
    ord('5'): "السلام علیکم، شکریہ"
}

print("\n" + "="*65)
print(" HOW TO BUILD WORDS & SENTENCES:")
print("   - Show sign: Green skeleton & letter appears")
print("   - [SPACE]: Add current letter to word (or hold for 1.3s)")
print("   - [S]: Add SPACE (to start a new word)")
print("   - [ENTER]: SPEAK full joined word/sentence (Urdu Voice)")
print("   - [BACKSPACE]: Delete last letter")
print("   - [C]: Clear entire text")
print("   - [1-5]: Quick Emergency Phrases")
print("   - [Q]: Quit")
print("="*65 + "\n")

speak_urdu("اشارہ لنک میں خوش آمدید")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    display_frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    active_sign_full = "Show Hand Sign..."
    active_urdu_char = ""
    active_conf = 0.0
    hold_progress = 0.0

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            display_frame,
            lm,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )

        feats = extract_normalized_landmarks(lm)
        batch = np.expand_dims(feats, axis=0)

        preds = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx] * 100)

        prediction_history.append(idx)
        most_common_idx = Counter(prediction_history).most_common(1)[0][0]

        active_sign_full = class_labels.get(str(most_common_idx), f"Sign {most_common_idx}")
        active_urdu_char = urdu_chars.get(str(most_common_idx), "")
        active_conf = conf

        # Auto-hold commit logic
        now = time.time()
        if active_conf > 80.0 and active_urdu_char:
            if active_urdu_char == last_hold_char:
                if hold_start_time is not None:
                    elapsed = now - hold_start_time
                    hold_progress = min(1.0, elapsed / HOLD_THRESHOLD_SEC)
                    if elapsed >= HOLD_THRESHOLD_SEC:
                        if active_urdu_char != last_added_char:
                            current_letters.append(active_urdu_char)
                            last_added_char = active_urdu_char
                        hold_start_time = None
            else:
                last_hold_char = active_urdu_char
                hold_start_time = now
        else:
            hold_start_time = None
            hold_progress = 0.0
    else:
        hold_start_time = None
        last_hold_char = None
        last_added_char = None

    # Top Recognition Banner
    banner_color = (0, 255, 0) if active_conf > 75 else (0, 165, 255)
    cv2.rectangle(display_frame, (10, 10), (w - 10, 85), (25, 25, 25), -1)
    cv2.rectangle(display_frame, (10, 10), (w - 10, 85), banner_color, 2)

    # English sign name + confidence
    cv2.putText(display_frame, f"Sign: {active_sign_full}", (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Confidence: {active_conf:.1f}%", (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_color, 2)

    # Active Urdu Character Preview (Large in top-right)
    if active_urdu_char:
        display_frame = render_urdu_text(
            display_frame,
            f"حرف: {active_urdu_char}",
            (w - 140, 20),
            font_size=36,
            color=(0, 255, 0)
        )

    # Hold progress visual bar
    if hold_progress > 0:
        bar_len = int(120 * hold_progress)
        cv2.rectangle(display_frame, (20, 78), (20 + bar_len, 83), (0, 255, 0), -1)

    # Bottom Joined Word / Sentence Banner
    raw_sentence = "".join(current_letters)
    joined_display = raw_sentence if raw_sentence else "(کوئی لفظ نہیں)"

    cv2.rectangle(display_frame, (10, h - 120), (w - 10, h - 10), (20, 20, 20), -1)
    cv2.rectangle(display_frame, (10, h - 120), (w - 10, h - 10), (0, 200, 255), 2)

    # Render Connected Urdu Word/Sentence with proper ligatures
    display_frame = render_urdu_text(
        display_frame,
        f"مکمل لفظ / جملہ: {joined_display}",
        (25, h - 110),
        font_size=30,
        color=(255, 255, 255)
    )

    # Speaking status or instructions
    if time.time() - speaking_time < 3.0 and speaking_status:
        display_frame = render_urdu_text(
            display_frame,
            f"🔊 آواز: {speaking_status}",
            (25, h - 50),
            font_size=24,
            color=(0, 255, 0)
        )
    else:
        cv2.putText(display_frame, "[SPACE] Add Letter | [S] Space | [ENTER] Speak | [BACKSPACE] Undo | [C] Clear",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1)

    cv2.imshow("IsharaLink - PSL to Urdu Speech Translator", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Add letter
        if active_urdu_char:
            current_letters.append(active_urdu_char)
            last_added_char = active_urdu_char
    elif key == ord('s'):  # Add space between words
        current_letters.append(" ")
        last_added_char = " "
    elif key == 8 or key == 127:  # Backspace
        if current_letters:
            current_letters.pop()
    elif key == ord('c'):  # Clear
        current_letters.clear()
        last_added_char = None
    elif key == 13 or key == 10:  # Enter -> Speak Urdu
        if current_letters:
            full_text = "".join(current_letters).strip()
            speak_urdu(full_text)
            speaking_status = full_text
            speaking_time = time.time()
            print(f"[Speaking]: {full_text}")
    elif key in DOMAIN_PRESETS:
        preset_text = DOMAIN_PRESETS[key]
        speak_urdu(preset_text)
        speaking_status = preset_text
        speaking_time = time.time()
        print(f"[Preset]: {preset_text}")

cap.release()
hands.close()
cv2.destroyAllWindows()
