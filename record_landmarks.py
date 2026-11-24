import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time

# ==========================================
# ISHARALINK - PSL SKELETAL LANDMARK COLLECTOR
# ==========================================

OUTPUT_FILE = "landmarks_dataset.npz"
LABELS_FILE = "class_labels.json"
SAMPLES_PER_SIGN = 40  # ~2 seconds per sign

# Load 37 PSL Class Labels
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        class_labels = json.load(f)
else:
    class_labels = {"0": "Ain ع", "1": "Aliph ا", "2": "Bay ب", "3": "Tay ت"}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def extract_normalized_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    wrist = coords[0]
    rel_coords = coords - wrist
    palm_size = np.linalg.norm(coords[9] - wrist)
    if palm_size > 0:
        rel_coords = rel_coords / palm_size
    return rel_coords.flatten()

# Open Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Error] Camera could not be opened!")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load existing dataset if present
existing_X = []
existing_y = []
if os.path.exists(OUTPUT_FILE):
    try:
        data = np.load(OUTPUT_FILE)
        existing_X = list(data["X"])
        existing_y = list(data["y"])
        print(f"Loaded existing dataset: {len(existing_X)} samples across {len(set(existing_y))} signs.")
    except Exception:
        pass

current_sign_index = 0
total_signs = len(class_labels)
sign_keys = sorted(class_labels.keys(), key=lambda x: int(x))

is_recording = False
recording_frames = []
countdown_start = None
message_flash = ""
message_time = 0

print("\n" + "="*55)
print("     ISHARALINK - LANDMARK DATASET COLLECTOR")
print("="*55)
print(" CONTROLS:")
print("   [SPACE] - Start 3-second countdown to record sign")
print("   [N]     - Next Sign")
print("   [P]     - Previous Sign")
print("   [C]     - Clear / Delete recordings of CURRENT sign")
print("   [R]     - Reset / Delete ALL recordings")
print("   [Q]     - Save and Quit")
print("="*55 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    display_frame = frame.copy()

    current_key = sign_keys[current_sign_index]
    current_name = class_labels[current_key]
    current_label_int = int(current_key)

    # Check how many samples already recorded for this sign
    samples_count = sum(1 for y in existing_y if y == current_label_int)

    # MediaPipe detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    hand_in_frame = False
    current_features = None

    if results.multi_hand_landmarks:
        hand_in_frame = True
        lm = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(
            display_frame,
            lm,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
        current_features = extract_normalized_landmarks(lm)

    # Countdown and Recording Logic
    now = time.time()
    status_text = "Press SPACE to Record"
    status_color = (255, 255, 255)

    if countdown_start is not None:
        elapsed = now - countdown_start
        if elapsed < 2.5:
            remaining = 3 - int(elapsed)
            status_text = f"Get Ready... {remaining}"
            status_color = (0, 165, 255)
            cv2.circle(display_frame, (w // 2, h // 2), 60, (0, 165, 255), -1)
            cv2.putText(display_frame, str(remaining), (w // 2 - 20, h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
        else:
            is_recording = True
            countdown_start = None
            recording_frames = []

    if is_recording:
        if hand_in_frame and current_features is not None:
            recording_frames.append(current_features)
            progress = len(recording_frames) / SAMPLES_PER_SIGN
            status_text = f"RECORDING: {len(recording_frames)}/{SAMPLES_PER_SIGN}"
            status_color = (0, 255, 0)

            bar_w = int(w * progress)
            cv2.rectangle(display_frame, (0, h - 20), (bar_w, h), (0, 255, 0), -1)

            if len(recording_frames) >= SAMPLES_PER_SIGN:
                existing_X.extend(recording_frames)
                existing_y.extend([current_label_int] * len(recording_frames))
                np.savez_compressed(
                    OUTPUT_FILE,
                    X=np.array(existing_X, dtype=np.float32),
                    y=np.array(existing_y, dtype=np.int32)
                )
                message_flash = f"Saved 40 samples for {current_name}!"
                message_time = time.time()
                is_recording = False
        else:
            status_text = "Keep Hand in Frame!"
            status_color = (0, 0, 255)

    # Top Banner UI
    cv2.rectangle(display_frame, (10, 10), (w - 10, 105), (20, 20, 20), -1)
    cv2.rectangle(display_frame, (10, 10), (w - 10, 105), (0, 200, 255), 2)

    cv2.putText(display_frame, f"Sign [{current_sign_index+1}/{total_signs}]: {current_name}",
                (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Recorded Samples: {samples_count} | Status: {status_text}",
                (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # Flash notification message
    if time.time() - message_time < 2.5 and message_flash:
        cv2.putText(display_frame, message_flash, (25, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Bottom Instructions
    cv2.putText(display_frame, "[SPACE] Record | [N]ext | [P]rev | [C]lear Sign | [R]eset All | [Q]uit",
                (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv2.imshow("IsharaLink - Landmark Data Collector", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and not is_recording:
        countdown_start = time.time()
    elif key == ord('n') and not is_recording:
        current_sign_index = (current_sign_index + 1) % total_signs
    elif key == ord('p') and not is_recording:
        current_sign_index = (current_sign_index - 1) % total_signs
    elif key == ord('c') and not is_recording:
        # Delete only current sign's recordings
        new_X = [x for x, y in zip(existing_X, existing_y) if y != current_label_int]
        new_y = [y for y in existing_y if y != current_label_int]
        existing_X = new_X
        existing_y = new_y
        if len(existing_X) > 0:
            np.savez_compressed(OUTPUT_FILE, X=np.array(existing_X, dtype=np.float32), y=np.array(existing_y, dtype=np.int32))
        else:
            if os.path.exists(OUTPUT_FILE):
                os.remove(OUTPUT_FILE)
        message_flash = f"Cleared all recordings for {current_name}!"
        message_time = time.time()
        print(f"[Cleared] Deleted recordings for: {current_name}")
    elif key == ord('r') and not is_recording:
        # Reset entire dataset
        existing_X = []
        existing_y = []
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        message_flash = "Reset All! Dataset is now empty."
        message_time = time.time()
        print("[Reset] Entire landmark dataset cleared!")

if len(existing_X) > 0:
    np.savez_compressed(
        OUTPUT_FILE,
        X=np.array(existing_X, dtype=np.float32),
        y=np.array(existing_y, dtype=np.int32)
    )

cap.release()
hands.close()
cv2.destroyAllWindows()
