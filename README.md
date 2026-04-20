\# Siena EEG Cognitive Assessment Project



This project uses Graph Neural Networks (GNNs) and the Attention mechanism (GATv2) to analyze resting-state EEG data.



\## Dataset

The raw EEG data used in this project is the \*\*Siena Scalp EEG Database (v1.0.0)\*\*. 

Due to the file size (13.0 GB), the data is not included in this repository.



You can access the files here:

\* \*\*PhysioNet Link:\*\* \[Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/)

\* \*\*Direct Terminal Download:\*\* `wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/`



\## Project Structure

\* `siena\_feature\_extraction.py`: Processes the raw EDF files.

\* `channelbasedcode\_Siena.py`: Brain network construction and GNN implementation.

