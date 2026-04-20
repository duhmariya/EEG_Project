import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
INPUT_DIR = r"C:\Users\mariy\Desktop\mne_plot_con_circle\mne_plot_con_circle\siena-scalp-eeg-database-1.0.0\siena_txt_format\Siena_Epilepsy_Feature_Matrices_10s"
OUTPUT_DIR = os.path.join(INPUT_DIR, "Subject_Wise_PNG")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_SHEETS = ["Sample_Entropy", "Spectral_Entropy", "Skewness", "Variance"]
BANDS = ["alpha", "beta", "theta"]

# Graph Y-axis thresholds
Y_LIMITS = {
    "Sample_Entropy": (0, 3),
    "Spectral_Entropy": (0, 3),
    "Skewness": (-1, 1),
    "Variance": None  # dynamic per file or subject
}

# =====================================================
# HELPERS
# =====================================================
def extract_subject(filename):
    match = re.search(r"(PN\d+)", filename)
    return match.group(1) if match else "Unknown"

def load_subject_files(folder):
    subjects = {}
    for file in os.listdir(folder):
        if not file.endswith(".xlsx") or file.startswith("~$"):
            continue
        subject = extract_subject(file)
        subjects.setdefault(subject, []).append(os.path.join(folder, file))
    return subjects

def load_file_data(file_path):
    """
    Load features for a single seizure/file
    """
    data = {band: {feat: pd.Series(dtype=float) for feat in FEATURE_SHEETS} for band in BANDS}
    xls = pd.ExcelFile(file_path)
    for feat in FEATURE_SHEETS:
        if feat not in xls.sheet_names:
            continue
        df = pd.read_excel(xls, sheet_name=feat, index_col=0)
        df.columns = [c.lower() for c in df.columns]
        for band in BANDS:
            if band in df.columns:
                data[band][feat] = df[band]
    return data

def average_subject_data(files):
    """
    Average features across all seizures/files for the subject
    """
    data = {band: {feat: [] for feat in FEATURE_SHEETS} for band in BANDS}
    for file in files:
        file_data = load_file_data(file)
        for band in BANDS:
            for feat in FEATURE_SHEETS:
                if not file_data[band][feat].empty:
                    data[band][feat].append(file_data[band][feat])
    averaged = {}
    for band in BANDS:
        averaged[band] = {}
        for feat in FEATURE_SHEETS:
            if data[band][feat]:
                averaged[band][feat] = pd.concat(data[band][feat], axis=1).mean(axis=1)
            else:
                averaged[band][feat] = pd.Series(dtype=float)
    return averaged

def plot_data(title, data, out_file):
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(title, fontsize=16)

    for row, band in enumerate(BANDS):
        for col, feat in enumerate(FEATURE_SHEETS):
            ax = axes[row, col]
            values = data[band][feat]

            if values.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f"{band.capitalize()} – {feat}")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            ax.bar(values.index, values.values)

            # Apply thresholds
            y_limit = Y_LIMITS[feat]
            if y_limit is None:
                y_limit = (0, values.max() * 1.1)
            ax.set_ylim(y_limit)

            ax.set_title(f"{band.capitalize()} – {feat}")
            ax.set_xlabel("Channels")
            ax.set_ylabel(feat)
            ax.tick_params(axis="x", rotation=90)
            ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved PNG: {out_file}")

# =====================================================
# MAIN
# =====================================================
def main():
    subject_files = load_subject_files(INPUT_DIR)
    if not subject_files:
        print("❌ No Excel files found in the input directory.")
        return

    for subject, files in subject_files.items():
        print(f"\nProcessing {subject} ({len(files)} files)")

        # --- Plot each seizure separately ---
        for file in files:
            seizure_name = os.path.splitext(os.path.basename(file))[0]
            file_data = load_file_data(file)
            out_file = os.path.join(OUTPUT_DIR, f"{seizure_name}_Channels_vs_Features.png")
            plot_data(f"{seizure_name} – Channels vs Features", file_data, out_file)

        # --- Plot subject-average across all seizures ---
        averaged_data = average_subject_data(files)
        out_file_avg = os.path.join(OUTPUT_DIR, f"{subject}_AVERAGE_Channels_vs_Features.png")
        plot_data(f"{subject} – AVERAGE Channels vs Features", averaged_data, out_file_avg)

    print("\n✅ ALL PNGs GENERATED (seizure-wise + subject-average)")

if __name__ == "__main__":
    main()
