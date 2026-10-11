# Data

The dataset is not included in this repository. Download the CIC-IDS2017 CSV files from the
[Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) (check the
dataset's terms and citation request there), then place this one file in this folder:

```
data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
```

The original notebook was run on a Kaggle copy of the file, in which the dash in labels such as
"Web Attack – Brute Force" is corrupted into a different character. The loader in `src/ids_eval.py`
matches labels by substring, so it works with either version.

CSV files in this folder are gitignored.
