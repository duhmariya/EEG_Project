"""
EpiConnectome — EDF to TXT Converter
Extracts ictal (seizure) segments from Siena EDF files
using the seizure annotation files.

Usage:
    python 01_siena_to_txt_converter.py

Output:
    One TXT file per seizure in the output folder
    Format: (samples x channels) plain text matrix
"""

import os
import mne
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

mne.set_log_level('WARNING')

# ══════════════════════════════════════════════════════════════
#  CHANGE THESE TWO PATHS FOR YOUR MACHINE
# ══════════════════════════════════════════════════════════════
EDF_ROOT    = r"C:\Users\mariy\Desktop\Siena\siena-scalp-eeg-database-1.0.0"
OUTPUT_DIR  = r"C:\Users\mariy\Desktop\Siena\siena-scalp-eeg-database-1.0.0\siena_txt_format"
# ══════════════════════════════════════════════════════════════

# Padding around seizure — seconds before and after to include
PRE_SEIZURE_PAD  = 30   # seconds before seizure start
POST_SEIZURE_PAD = 30   # seconds after seizure end

# Standard 16 channels to extract (10-20 system)
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3',  'C3',  'Cz', 'C4', 'T4',
    'T5',  'P3',  'Pz', 'P4'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_time(time_str):
    """Parse time string HH.MM.SS into a datetime object."""
    time_str = time_str.strip()
    return datetime.strptime(time_str, "%H.%M.%S")


def time_to_seconds(t):
    """Convert datetime to seconds since midnight."""
    return t.hour * 3600 + t.minute * 60 + t.second


def parse_annotation_file(annot_path):
    """
    Parse Siena annotation file.
    Returns list of dicts with keys:
        seizure_n, filename, reg_start, reg_end, seiz_start, seiz_end
    """
    seizures = []
    current = {}

    with open(annot_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.lower().startswith('seizure n'):
            if current.get('filename'):
                seizures.append(current)
            current = {}
            try:
                current['seizure_n'] = int(line.split()[-1])
            except:
                current['seizure_n'] = len(seizures) + 1

        elif line.lower().startswith('file name:'):
            current['filename'] = line.split(':', 1)[1].strip()

        elif line.lower().startswith('registration start time:'):
            try:
                current['reg_start'] = parse_time(line.split(':', 1)[1].strip())
            except:
                pass

        elif line.lower().startswith('registration end time:'):
            try:
                current['reg_end'] = parse_time(line.split(':', 1)[1].strip())
            except:
                pass

        elif line.lower().startswith('seizure start time:'):
            try:
                current['seiz_start'] = parse_time(line.split(':', 1)[1].strip())
            except:
                pass

        elif line.lower().startswith('seizure end time:'):
            try:
                current['seiz_end'] = parse_time(line.split(':', 1)[1].strip())
            except:
                pass

    # Don't forget the last seizure
    if current.get('filename'):
        seizures.append(current)

    return seizures


def extract_seizure_segment(edf_path, reg_start, seiz_start, seiz_end, sfreq=512):
    """
    Extract the seizure segment from an EDF file.
    Returns numpy array of shape (samples, 16 channels).
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Find matching channels (case-insensitive)
    available = [ch.upper().strip() for ch in raw.ch_names]
    target_upper = [ch.upper() for ch in TARGET_CHANNELS]

    picks = []
    found_channels = []
    for t_ch in target_upper:
        for i, a_ch in enumerate(available):
            # Match exactly or with spaces stripped
            if a_ch == t_ch or a_ch.replace(' ', '') == t_ch:
                picks.append(i)
                found_channels.append(raw.ch_names[i])
                break

    if len(picks) == 0:
        raise ValueError(f"No matching channels found in {edf_path}")

    print(f"   Matched {len(picks)}/16 channels: {found_channels}")

    # Calculate seizure offset from registration start
    reg_start_s  = time_to_seconds(reg_start)
    seiz_start_s = time_to_seconds(seiz_start)
    seiz_end_s   = time_to_seconds(seiz_end)

    # Handle midnight crossing
    if seiz_start_s < reg_start_s:
        seiz_start_s += 86400
    if seiz_end_s < reg_start_s:
        seiz_end_s += 86400

    # Offset from start of EDF file
    offset_start = seiz_start_s - reg_start_s
    offset_end   = seiz_end_s   - reg_start_s

    # Add padding
    t_start = max(0, offset_start - PRE_SEIZURE_PAD)
    t_end   = min(raw.times[-1], offset_end + POST_SEIZURE_PAD)

    print(f"   Seizure window: {offset_start:.1f}s – {offset_end:.1f}s (duration: {offset_end-offset_start:.1f}s)")
    print(f"   Extracting: {t_start:.1f}s – {t_end:.1f}s (with ±{PRE_SEIZURE_PAD}s padding)")

    # Crop to seizure segment
    raw_cropped = raw.copy().crop(tmin=t_start, tmax=t_end)
    raw_cropped.pick(picks)

    # Get data and transpose to (samples, channels)
    data = raw_cropped.get_data().T  # (samples, channels)

    return data, found_channels


def process_all_patients():
    """Main function — finds all annotation files and converts EDF → TXT."""

    # Find all annotation files
    annot_files = []
    for patient_dir in sorted(os.listdir(EDF_ROOT)):
        patient_path = os.path.join(EDF_ROOT, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        for f in os.listdir(patient_path):
            if f.endswith('.txt') and patient_dir in f:
                annot_files.append(os.path.join(patient_path, f))

    if not annot_files:
        print("❌ No annotation files found. Looking for any .txt files...")
        for patient_dir in sorted(os.listdir(EDF_ROOT)):
            patient_path = os.path.join(EDF_ROOT, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            for f in os.listdir(patient_path):
                if f.endswith('.txt'):
                    annot_files.append(os.path.join(patient_path, f))

    print(f"Found {len(annot_files)} annotation files")
    for a in annot_files:
        print(f"   {a}")

    success = 0
    failed  = 0
    skipped = 0

    for annot_path in sorted(annot_files):
        patient_dir = os.path.dirname(annot_path)
        patient_id  = os.path.basename(patient_dir)

        print(f"\n{'='*60}")
        print(f"Patient: {patient_id}")
        print(f"{'='*60}")

        try:
            seizures = parse_annotation_file(annot_path)
            print(f"Found {len(seizures)} seizures")
        except Exception as e:
            print(f"❌ Failed to parse annotation: {e}")
            failed += 1
            continue

        for sz in seizures:
            n         = sz.get('seizure_n', '?')
            edf_name  = sz.get('filename', '')
            reg_start = sz.get('reg_start')
            seiz_start = sz.get('seiz_start')
            seiz_end   = sz.get('seiz_end')

            if not all([edf_name, reg_start, seiz_start, seiz_end]):
                print(f"   ⚠️  Seizure {n}: missing info, skipping")
                skipped += 1
                continue

            # Output filename
            out_name = f"{patient_id}_seizure{n}.txt"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            # Skip if already converted
            if os.path.exists(out_path):
                print(f"   ⏭️  Seizure {n}: already exists ({out_name}), skipping")
                skipped += 1
                continue

            # Find EDF file
            edf_path = os.path.join(patient_dir, edf_name)
            if not os.path.exists(edf_path):
                print(f"   ❌ Seizure {n}: EDF not found: {edf_path}")
                failed += 1
                continue

            print(f"\n   Seizure {n}: {edf_name}")
            print(f"   Seizure time: {seiz_start.strftime('%H:%M:%S')} → {seiz_end.strftime('%H:%M:%S')}")

            try:
                data, channels = extract_seizure_segment(
                    edf_path, reg_start, seiz_start, seiz_end)

                np.savetxt(out_path, data, fmt='%.6f')

                print(f"   ✅ Saved: {out_name} — shape {data.shape}")
                success += 1

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"CONVERSION COMPLETE")
    print(f"  Success:  {success}")
    print(f"  Skipped:  {skipped} (already existed)")
    print(f"  Failed:   {failed}")
    print(f"  Output:   {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_all_patients()
