import os
import random
import shutil
import math
import uuid
from pathlib import Path
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ================== CONFIG ==================
dataset_dir = "dataset"                 # <-- dost ka dataset (PC path)
balanced_root = "balanced_dataset"
strategy = "undersample"                # or "oversample"
img_size = 224
batch_size = 32
train_frac = 0.80
initial_epochs = 10
random_seed = 42
# ===========================================

random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# -------- 1. Dataset sanity check ----------
dataset_path = Path(dataset_dir)
classes = [p.name for p in dataset_path.iterdir() if p.is_dir()]
if not classes:
    raise SystemExit("No class folders found inside dataset/")
print("Detected classes:", classes)

# -------- 2. Count images ------------------
class_files = {}
counts = {}
for c in classes:
    files = [p for p in (dataset_path / c).glob("*") if p.is_file()]
    class_files[c] = files
    counts[c] = len(files)
print("Original counts:", counts)

# -------- 3. Balancing ---------------------
target_count = min(counts.values())
print("Balancing strategy:", strategy, "Target:", target_count)

pool_dir = Path(balanced_root) / "pool"
if pool_dir.exists():
    shutil.rmtree(pool_dir)
pool_dir.mkdir(parents=True)

for c in classes:
    (pool_dir / c).mkdir(parents=True)
    files = class_files[c]

    if len(files) > target_count:
        files = random.sample(files, target_count)

    for f in files:
        shutil.copy2(f, pool_dir / c / f.name)

# -------- 4. Train / Val split -------------
train_dir = Path(balanced_root) / "train"
val_dir = Path(balanced_root) / "val"

if train_dir.exists():
    shutil.rmtree(train_dir)
if val_dir.exists():
    shutil.rmtree(val_dir)

for c in classes:
    (train_dir / c).mkdir(parents=True)
    (val_dir / c).mkdir(parents=True)

    all_files = list((pool_dir / c).glob("*"))
    random.shuffle(all_files)

    n_train = int(len(all_files) * train_frac)
    for f in all_files[:n_train]:
        shutil.copy2(f, train_dir / c / f.name)
    for f in all_files[n_train:]:
        shutil.copy2(f, val_dir / c / f.name)

print("Balanced dataset ready.")

# -------- 5. Data Generators ---------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

num_classes = train_gen.num_classes
print("Class indices:", train_gen.class_indices)

# -------- 6. Model -------------------------
base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(img_size, img_size, 3)
)
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256)(x)
x = BatchNormalization()(x)
x = tf.keras.activations.relu(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(base.input, outputs)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------- 7. Training ----------------------
callbacks = [
    ModelCheckpoint(
        "best_balanced_noaug.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=initial_epochs,
    steps_per_epoch=math.ceil(train_gen.samples / batch_size),
    validation_steps=math.ceil(val_gen.samples / batch_size),
    callbacks=callbacks
)

# -------- 8. Save Model --------------------
model.save("best_balanced_noaug.keras")
print("Model saved as best_balanced_noaug.keras")
