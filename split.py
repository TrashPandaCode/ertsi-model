import os
import shutil
import random
from glob import glob

SEED = 42
VAL_RATIO = 0.1
TEST_RATIO = 0.2  # only applies to real data
INPUT_ROOT = "data_orig"
OUTPUT_ROOT = "data"

random.seed(SEED)

def split_rooms(room_paths, test_ratio=0.0, val_ratio=0.1):
    random.shuffle(room_paths)
    n_total = len(room_paths)
    n_test = int(test_ratio * n_total)
    n_val = int(val_ratio * n_total)

    test = room_paths[:n_test]
    val = room_paths[n_test:n_test + n_val]
    train = room_paths[n_test + n_val:]

    return train, val, test

def copy_rooms(room_list, split, domain):
    for room_path in room_list:
        dest = os.path.join(OUTPUT_ROOT, split, domain, os.path.basename(room_path))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copytree(room_path, dest)

# Split REAL dataset
real_rooms = [d for d in glob(os.path.join(INPUT_ROOT, "real", "*")) if os.path.isdir(d)]
real_train, real_val, real_test = split_rooms(real_rooms, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO)

# Split SYNTH dataset
synth_rooms = [d for d in glob(os.path.join(INPUT_ROOT, "synth", "*")) if os.path.isdir(d)]
synth_train, synth_val, _ = split_rooms(synth_rooms, test_ratio=0.0, val_ratio=VAL_RATIO)  # no synth test

# Copy files
copy_rooms(real_train, "train", "real")
copy_rooms(real_val, "val", "real")
copy_rooms(real_test, "test", "real")

copy_rooms(synth_train, "train", "synth")
copy_rooms(synth_val, "val", "synth")

print(f"  Real -> Train: {len(real_train)}, Val: {len(real_val)}, Test: {len(real_test)}")
print(f"  Synth -> Train: {len(synth_train)}, Val: {len(synth_val)}")