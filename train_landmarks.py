import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os

# ==========================================
# ISHARALINK - TRAIN LANDMARK NEURAL NETWORK
# ==========================================

DATA_PATH = "landmarks_dataset.npz"
LABELS_PATH = "class_labels.json"
MODEL_SAVE_PATH = "isharalink_landmarks_model.keras"

print("\n" + "="*50)
print("     ISHARALINK - LANDMARK MODEL TRAINING")
print("="*50)

if not os.path.exists(DATA_PATH):
    print(f"\n[Error] {DATA_PATH} not found! Run record_landmarks.py first.")
    raise SystemExit

data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]

print(f"\nTotal Samples: {len(X)}")
print(f"Feature Dimension: {X.shape[1]} (21 joints x 3: dx, dy, dz)")

unique_classes = sorted(np.unique(y))
num_classes = len(unique_classes)
print(f"Recorded Classes: {num_classes}")

# Read Class Labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_labels = json.load(f)

for c in unique_classes:
    print(f"   - Class {c}: {class_labels.get(str(c), f'Class {c}')} ({np.sum(y == c)} samples)")

if num_classes < 2:
    print("\n[Warning] Please record at least 2 different signs before training!")
    raise SystemExit

# Build Lightweight Landmark MLP Classifier
# Max classes is total class_labels count (37)
total_possible_classes = max(len(class_labels), max(unique_classes) + 1)

model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(total_possible_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

print("\nTraining Model (15 epochs)...")
history = model.fit(
    X, y,
    epochs=15,
    batch_size=16,
    validation_split=0.2 if len(X) >= 20 else 0.0,
    verbose=1
)

model.save(MODEL_SAVE_PATH)
print(f"\nModel successfully saved to: {MODEL_SAVE_PATH}")
print("Training completed!")
