# ML-IDS Benchmark: Anomaly-Based Intrusion Detection on CIC-IDS2017 Web Attacks

A replication of the benchmarking study by Maseer et al. (IEEE Access, 2021) on web-attack traffic from the CIC-IDS2017 dataset. It compares 13 model configurations from 9 algorithm families (ANN, decision tree, k-NN with k = 1 to 5, naive Bayes, random forest, SVM, K-Means, EM, SOM) on a four-class problem (benign, brute force, XSS, SQL injection) using 5-fold stratified cross-validation.

Originally a class project (CS 446, Summer 2026). The notebook was run on Google Colab. This repository also adds an evaluation script for the checks that accuracy alone cannot show on a heavily imbalanced dataset (see [Extended evaluation](#extended-evaluation)).

## Dataset

`Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` from [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html). It is not included in this repository; see [`data/README.md`](data/README.md).

After dropping rows with missing or infinite values: **170,231 flows x 78 numeric features**.

| Class | Flows | Share |
| :--- | ---: | ---: |
| Benign (0) | 168,051 | 98.72% |
| Web Attack - Brute Force (1) | 1,507 | 0.89% |
| Web Attack - XSS (2) | 652 | 0.38% |
| Web Attack - SQL Injection (3) | 21 | 0.01% |

A classifier that always predicts "Benign" scores **98.72% accuracy** on this data. That is the number every accuracy figure below has to be compared against.

## Method

- **Labels:** raw labels mapped to four classes by substring match. (Some copies of the CSV corrupt the dash in "Web Attack - Brute Force", which breaks exact-string matching and silently removes every attack row.)
- **Features:** `StandardScaler`, then clipped to [-3, 3] (the interval used in the paper).
- **Models:** hyperparameters from the paper where it reports them: decision tree (entropy, balanced class weights), random forest (100 trees, balanced), SVM (RBF kernel, balanced, capped at `max_iter=1000`), k-NN with k = 1 to 5, ANN (`MLPClassifier`, `max_iter=500`), and K-Means / EM / SOM with 4 clusters. Unsupervised models are fit on features only. The paper's CNN is not implemented.
- **Evaluation:** 5-fold stratified cross-validation (shuffled, seed 42). Precision, recall and F1 are weighted averages. T1 is mean training time per fold and T2 is mean prediction time per fold.

## Results (original notebook run)

| Algorithm | Accuracy | Precision | Recall | F1-Score | T1 (s) | T2 (s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANN | 0.9925 | 0.9939 | 0.9925 | 0.9909 | 83.5984 | 0.1220 |
| Decision Tree | 0.9956 | 0.9957 | 0.9956 | 0.9957 | 2.8120 | 0.0067 |
| k-NN (k=1) | 0.9952 | 0.9952 | 0.9952 | 0.9952 | 0.0222 | 44.3212 |
| k-NN (k=2) | 0.9957 | 0.9950 | 0.9957 | 0.9950 | 0.0213 | 44.5334 |
| k-NN (k=3) | 0.9954 | 0.9951 | 0.9954 | 0.9952 | 0.0212 | 44.6190 |
| k-NN (k=4) | 0.9957 | 0.9950 | 0.9957 | 0.9950 | 0.0256 | 44.0870 |
| k-NN (k=5) | 0.9954 | 0.9950 | 0.9954 | 0.9951 | 0.0238 | 44.0449 |
| Naive Bayes | 0.9883 | 0.9902 | 0.9883 | 0.9885 | 0.2174 | 0.0679 |
| Random Forest | 0.9956 | 0.9952 | 0.9956 | 0.9953 | 20.3281 | 0.1426 |
| SVM | 0.7670 | 0.9874 | 0.7670 | 0.8588 | 39.1379 | 10.9934 |
| K-Means | 0.3026 | 0.9662 | 0.3026 | 0.4327 | 0.6749 | 0.0070 |
| EM | 0.2607 | 0.9568 | 0.2607 | 0.3741 | 47.7705 | 0.1750 |
| SOM | 0.2862 | 0.9433 | 0.2862 | 0.4157 | 3.8073 | 0.6557 |

Source: [`notebooks/Final_Project.ipynb`](notebooks/Final_Project.ipynb), also saved as [`results/benchmark_results_notebook.csv`](results/benchmark_results_notebook.csv).

**What the table supports:**

- Supervised models other than the SVM all reach at least 98.8% accuracy.
- Among the top group (decision tree, random forest, k-NN), accuracy differs by about 0.05 points or less, so cost separates them. The decision tree takes 2.8 s to train and 0.007 s to predict a fold. Random forest takes 20.3 s to train, and k-NN takes about 44 s to predict a fold.

**How to read it carefully:**

- **Accuracy is close to the baseline.** The best supervised accuracies (about 99.5% to 99.6%) are roughly 0.8 points above the 98.72% majority-class baseline. Weighted recall equals accuracy by definition, which is why those two columns match.
- **Weighted metrics are dominated by Benign.** With 21 SQL-injection flows out of 170,231, a model can miss that class entirely and barely change its weighted F1. Per-class recall and macro-F1 are the informative metrics here.
- **The unsupervised rows are not a fair ranking.** The notebook compares each model's cluster index directly with the class id, but cluster numbering is arbitrary, so the 26% to 30% accuracy partly reflects numbering, not detection ability. The extended script maps clusters to classes by majority vote on the training fold.
- **The SVM result is not a representative SVM benchmark.** The 1,000-iteration cap likely stopped training before convergence (the notebook suppresses warnings, so this was not recorded). The extended script records whether a model hit its cap.
- **Timings are from one Colab session** with unrecorded hardware, on one day of traffic, so they should not be compared directly with the paper's reported times.

## Extended evaluation

[`src/ids_eval.py`](src/ids_eval.py) re-runs the same 13 configurations and adds:

- a majority-class baseline row,
- per-class recall and F1, and macro-F1, from pooled out-of-fold predictions,
- the scaler fit inside each fold (the notebook fits it on all the data before splitting),
- cluster-to-label mapping for K-Means / EM / SOM (labels are used only on the training fold to name clusters),
- a duplicate-row count, and `--dedupe` to drop exact duplicates before cross-validation,
- a flag for models that hit their iteration cap,
- confusion matrices as JSON.

```bash
python -m src.ids_eval --csv data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
python -m src.ids_eval --csv data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv --dedupe
python -m src.ids_eval --csv data/...csv --models forest,tree   # subset by name fragment
```

Outputs go to `results/` (`extended_cv.csv`, `confusion_matrices.json`, `data_report.json`, with a `_dedup` suffix for the `--dedupe` run). The unit tests in `tests/` check the evaluation logic on small synthetic data; they do not reproduce the real results.

## Reproduce

```bash
git clone https://github.com/jaydenxaf/ML-IDS-Benchmark.git
cd ML-IDS-Benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
# download the CSV into data/ (see data/README.md), then:
python -m src.ids_eval --csv data/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
python -m pytest
```

The original notebook expects the CSV at `/content/` (Colab) and installs `sklearn-som` in its first cell.

## Repository structure

```
.
├── notebooks/Final_Project.ipynb   # original Colab notebook, with outputs
├── src/ids_eval.py                 # extended evaluation (per-class metrics, baseline, cluster mapping)
├── tests/                          # unit tests on synthetic data
├── results/                        # result tables
├── data/                           # put the CSV here (gitignored)
├── requirements.txt
└── requirements-dev.txt
```

## Limitations

- **One day, one attack family.** Only the Thursday-morning web-attack file is used, not the full dataset.
- **Random splits on flow records.** Flows from the same attack session can land in both train and test folds, and there is no temporal split, so scores are likely optimistic for real deployment.
- **Duplicates were not checked in the original run.** CIC-IDS2017 contains repeated flow records, which can inflate scores for memorizing models such as k-NN. The extended script reports the duplicate count and supports a deduplicated run.
- **Tiny rare classes.** SQL injection has 21 samples, so per-fold estimates for that class are very noisy.
- **No CNN**, which the paper includes.

## Roadmap

- [ ] Run `src/ids_eval.py` on the full data and commit `extended_cv.csv` and `confusion_matrices.json`
- [ ] Run the deduplicated variant and compare
- [ ] Add a time-based or session-grouped split
- [ ] Record hardware and library versions with the results

## References

- Z. K. Maseer, R. Yusof, N. Bahaman, S. A. Mostafa, C. F. M. Foozy, "Benchmarking of Machine Learning for Anomaly Based Intrusion Detection Systems in the CICIDS2017 Dataset," *IEEE Access*, vol. 9, pp. 22351-22370, 2021. doi:10.1109/ACCESS.2021.3056614
- I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," *ICISSP*, 2018. (CIC-IDS2017)

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Author

Built by [@jaydenxaf](https://github.com/jaydenxaf).
