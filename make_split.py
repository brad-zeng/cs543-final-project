import os
import csv
import random

# ========== CONFIG ==========
DATA_ROOT = "data"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
# ============================

random.seed(SEED)

samples = []

for folder in sorted(os.listdir(DATA_ROOT)):
    folder_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    label = 1 if folder.lower() == "real" else 0
    generator = folder.lower()

    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append(
                (os.path.join(folder, fname), label, generator)
            )

random.shuffle(samples)

n = len(samples)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_samples = samples[:n_train]
val_samples = samples[n_train:n_train + n_val]
test_samples = samples[n_train + n_val:]

def write_csv(filename, rows):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "generator"])
        writer.writerows(rows)

write_csv("train.csv", train_samples)
write_csv("val.csv", val_samples)
write_csv("test.csv", test_samples)

print(f"Train: {len(train_samples)}")
print(f"Val:   {len(val_samples)}")
print(f"Test:  {len(test_samples)}")
