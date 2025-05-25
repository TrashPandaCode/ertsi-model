import random
from glob import glob
import os
import shutil

# This script splits the dataset into train, validation, and test sets.

DATA_ROOT = "data"
OUTPUT_ROOT = "data_split"
SEED = 42
TEST_RATIO = 0.2
VAL_RATIO = 0.1

random.seed(SEED)

room_dirs = [d for d in glob(os.path.join(DATA_ROOT, "*")) if os.path.isdir(d)]
random.shuffle(room_dirs)

n_total = len(room_dirs)
n_test = int(n_total * TEST_RATIO)
n_val = int(n_total * VAL_RATIO)

test_rooms = room_dirs[:n_test]
val_rooms = room_dirs[n_test:n_test + n_val]
train_rooms = room_dirs[n_test + n_val:]

splits = [("train", train_rooms), 
          ("val", val_rooms), 
          ("test", test_rooms)]

for split_name, rooms in splits:
    for room in rooms:
        dest = os.path.join(OUTPUT_ROOT, split_name, os.path.basename(room))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copytree(room, dest)

print(f"Data split into {len(train_rooms)} train, {len(val_rooms)} val, {len(test_rooms)} test rooms.")