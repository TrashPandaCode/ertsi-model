import os
import glob
import pandas as pd
import argparse

INPUT_DIR = "data"
THRESHOLD = 2.2
MIN_FREQ = 100
LOG_FILE = "prepare_log.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
args = parser.parse_args()

log_entries = []

# Locate all rt60.csv files
csv_files = glob.glob(os.path.join(INPUT_DIR, "**", "rt60.csv"), recursive=True)

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    modified = False

    for i in range(len(df)):
        freq = df.at[i, "Frequency (Hz)"]
        rt60 = df.at[i, "RT60 (s)"]

        if freq < MIN_FREQ:
            continue  # Skip low frequencies

        if rt60 > THRESHOLD:
            # Compute replacement value
            if i == 0:
                new_val = df.at[i + 1, "RT60 (s)"]
            elif i == len(df) - 1:
                new_val = df.at[i - 1, "RT60 (s)"]
            else:
                new_val = (df.at[i - 1, "RT60 (s)"] + df.at[i + 1, "RT60 (s)"]) / 2

            # Log it
            log_entries.append({
                "file": csv_path,
                "frequency": freq,
                "original": rt60,
                "replaced_with": new_val
            })

            # Replace if not dry-run
            if not args.dry_run:
                df.at[i, "RT60 (s)"] = new_val
                modified = True

    if modified and not args.dry_run:
        df.to_csv(csv_path, index=False)
        print(f"🛠️  Modified: {csv_path}")
    else:
        print(f"✅ No changes needed: {csv_path}")

# Save log
if log_entries:
    log_df = pd.DataFrame(log_entries)
    log_df.to_csv(LOG_FILE, index=False)
    print(f"📝 Log saved to {LOG_FILE}")
else:
    print("✅ No RT60 values exceeded threshold. Nothing to log.")
