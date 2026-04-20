# EpiConnectome 🧠

### A Reproducible Pipeline and Educational Resource for EEG Connectivity Analysis of Epileptic Seizures

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![MNE](https://img.shields.io/badge/MNE--Python-1.6+-green.svg)](https://mne.tools/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-PhysioNet%20Siena-orange.svg)](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
[![Brainhack](https://img.shields.io/badge/Brainhack-School%202026-purple.svg)](https://school-brainhack.github.io/)

---

## What is EpiConnectome?

EpiConnectome is an **open-source, fully reproducible Python pipeline** for analyzing EEG functional connectivity during epileptic seizures. It is built entirely on public data and open-source tools — meaning anyone can download, run, and learn from it without needing special access or proprietary software.

> **This project is not a novel clinical discovery.**
> It is a **shareable, well-documented pipeline** that:
> 1. ✅ Works on public data — anyone can reproduce the results
> 2. ✅ Uses only open-source tools — no barriers to entry
> 3. ✅ Includes educational documentation — teaches others how it works
> 4. ✅ Provides benchmark connectivity values — useful reference for the community
> 5. ✅ Can be adapted to other EEG datasets — generalizable beyond epilepsy

---

## Dataset

**Siena Scalp EEG Database (v1.0.0)** — PhysioNet

| Property | Value |
|---|---|
| Patients | 14 subjects |
| Seizures | 47 seizures |
| Channels | 29 scalp electrodes (10-20 system) |
| Sampling rate | 512 Hz |
| Format | EDF + seizure annotations |

> ⚠️ The raw EEG data (13.0 GB) is **not included** in this repository due to file size.

**Download the dataset:**
```bash
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/
```
Or visit: https://physionet.org/content/siena-scalp-eeg/1.0.0/

**Citation:**
> Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/5d4a-j060

---

## Pipeline Overview

```
Raw EEG (EDF)
     │
     ▼
01_siena_to_txt_converter.py   →  Convert EDF → TXT (seizure segments)
     │
     ▼
02_channelbasedcode_Siena.py   →  Preprocessing + wPLI Connectivity
     │  - Filtering (4–40 Hz)
     │  - Bad channel detection (LOF)
     │  - ICA artifact removal (EOG + muscle)
     │  - Epoching (10s windows)
     │  - wPLI connectivity (θ / α / β)
     │  - Graph theory metrics
     │  - Circle plots + bar charts
     │  - Excel connectivity matrices
     ▼
03_siena_feature_extraction.py →  EEG Feature Extraction
     │  - Sample Entropy
     │  - Spectral Entropy
     │  - Variance
     │  - Skewness
     ▼
04_siencehisto.py              →  Visualization of connectivity matrices
```

---

## Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/duhmariya/EEG_Project.git
cd EEG_Project
```

**Step 2: Create conda environment (recommended)**
```bash
conda create -n epiconnectome python=3.11
conda activate epiconnectome
```

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Convert EDF files to TXT
```bash
python scripts/01_siena_to_txt_converter.py
```
- A folder dialog will open
- Select the folder containing your downloaded Siena EDF files
- Output: TXT files with seizure-segment EEG data

### Step 2 — Run connectivity analysis
```bash
python scripts/02_channelbasedcode_Siena.py
```
- Select the folder containing your TXT files
- The pipeline will preprocess and compute wPLI for all files
- Output: Excel matrices + connectivity circle plots + bar charts

### Step 3 — Extract EEG features
```bash
python scripts/03_siena_feature_extraction.py
```
- Select the same TXT folder
- Output: Feature heatmaps + aggregated Excel file

### Step 4 — Visualize connectivity matrices
```bash
python scripts/04_siencehisto.py
```
- Select the folder containing output Excel files
- Output: Bar chart plots for each file and frequency band

---

## Key Parameters

| Parameter | Value | Description |
|---|---|---|
| Channels | 16 (standard 10-20) | Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4 |
| Sampling rate | 512 Hz → 1000 Hz | Original → resampled |
| Epoch length | 10 seconds | Non-overlapping windows |
| Connectivity metric | wPLI | Weighted Phase Lag Index |
| Frequency bands | θ (6–8 Hz), α (8–12 Hz), β (12–30 Hz) | |
| ICA components | 15 (infomax) | EOG + muscle artifact removal |
| Threshold | 0.5 (network), 0.3 (visualization) | |
| Random seed | 42 | Full reproducibility |

---

## Sample Outputs

### Connectivity Circle Plots (wPLI)

| Theta Band | Alpha Band | Beta Band |
|:---:|:---:|:---:|
| ![Theta](results/sample_outputs/circle_Theta.png) | ![Alpha](results/sample_outputs/circle_Alpha.png) | ![Beta](results/sample_outputs/circle_Beta.png) |

*Color scale: dark (low wPLI) → yellow (high wPLI). Connections above visualization threshold shown.*

### Channel Connectivity Bar Charts

| Theta (6–8 Hz) | Alpha (8–12 Hz) | Beta (12–30 Hz) |
|:---:|:---:|:---:|
| ![Theta Bar](results/sample_outputs/Connectivity_theta__6-8_hz__channel_bar.png) | ![Alpha Bar](results/sample_outputs/Connectivity_alpha__8-12_hz__channel_bar.png) | ![Beta Bar](results/sample_outputs/Connectivity_beta__12-30_hz__channel_bar.png) |

*Red dashed line = threshold (0.3). All channels exceed threshold, indicating strong seizure-period connectivity.*

---

## Output Files

For each processed file, the pipeline generates:

```
{subject_folder}/
├── Siena_Epilepsy_Channel_analysis_10s/
│   └── {file_stem}/
│       ├── circle_Theta.png
│       ├── circle_Alpha.png
│       ├── circle_Beta.png
│       └── bar_charts/
├── Siena_Epilepsy_matrices_10s/
│   └── Siena_Epilepsy_Channel_10s_{file_stem}.xlsx
│       ├── Sheet: Theta   (16×16 wPLI matrix)
│       ├── Sheet: Alpha   (16×16 wPLI matrix)
│       └── Sheet: Beta    (16×16 wPLI matrix)
└── Siena_Epilepsy_quality_control_10s/
    ├── 所有檔案_壞通道統計.xlsx
    └── 所有檔案_Channel網路指標統計.xlsx
```

---

## Graph Theory Metrics

For each frequency band, the pipeline computes:

| Metric | Description |
|---|---|
| Global Efficiency | How efficiently information is exchanged across the network |
| Average Clustering | Tendency of nodes to form local clusters |
| Network Density | Fraction of possible connections that are active |
| Modularity | Degree of separation into functional communities |
| Node Strength | Sum of connection weights for each channel |
| Average Degree | Mean number of connections per channel |

---

## Project Structure

```
EEG_Project/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
│
├── scripts/
│   ├── 01_siena_to_txt_converter.py
│   ├── 02_channelbasedcode_Siena.py
│   ├── 03_siena_feature_extraction.py
│   └── 04_siencehisto.py
│
├── notebooks/
│   └── EpiConnectome_Tutorial.ipynb      ← coming soon
│
└── results/
    └── sample_outputs/
        ├── circle_Theta.png
        ├── circle_Alpha.png
        ├── circle_Beta.png
        ├── Connectivity_theta__6-8_hz__channel_bar.png
        ├── Connectivity_alpha__8-12_hz__channel_bar.png
        ├── Connectivity_beta__12-30_hz__channel_bar.png
        └── Connectivity.xlsx
```

---

## Roadmap

- [x] EDF → TXT conversion
- [x] Preprocessing pipeline (filtering, bad channels, ICA)
- [x] wPLI connectivity computation (θ, α, β)
- [x] Graph theory metrics (efficiency, clustering, modularity)
- [x] Circle plot visualizations
- [x] Channel bar chart visualizations
- [x] Excel connectivity matrix output
- [x] EEG feature extraction (SampEn, SpecEn, Var, Skew)
- [ ] Jupyter tutorial notebook
- [ ] Source-level analysis (dSPM + HCPMMP1)
- [ ] GNN / GATv2 application on connectivity matrices
- [ ] Full 47-seizure connectivity atlas release

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `mne` | EEG preprocessing and source analysis |
| `mne-connectivity` | wPLI and spectral connectivity |
| `numpy / scipy` | Numerical computation |
| `pandas` | Data handling and Excel export |
| `matplotlib / seaborn` | Visualization |
| `networkx` | Graph theory metrics |
| `scikit-learn` | Preprocessing utilities |
| `openpyxl` | Excel file writing |

---

## Reproducibility

All scripts use a fixed random seed (`RANDOM_SEED = 42`) and deterministic MNE settings:
```python
os.environ['PYTHONHASHSEED'] = '42'
mne.set_config('MNE_RANDOM_SEED', '42')
mne.set_config('MNE_USE_NUMBA', 'false')
```
Running the pipeline on the same input files will always produce identical results.

---

## Contributing

This project welcomes contributions from the Brainhack community. If you find a bug, want to improve documentation, or want to extend the pipeline to a new dataset, please open an issue or submit a pull request.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **Siena Scalp EEG Database** — Paolo Detti, University of Siena
- **PhysioNet** — Goldberger et al., 2000
- **MNE-Python** — Gramfort et al., 2013
- **Brainhack School 2026**

---

*Built as part of Brainhack School 2026 · github.com/duhmariya/EEG_Project*
