# Translation pipeline evaluation

Measures translation correctness and latency of the speech translator's text-translation stage
(no speech recognition, no speech output) for Hindi, Spanish and English:

- Offline: Argos Translate
- Online: Google Translate
- Directions: HI→EN, EN→HI, ES→EN, EN→ES
- Test set: 15 parallel sentences (5 short, 5 medium, 5 long)

## 1. Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python eval_translation.py setup      # downloads the Argos hi<->en and es<->en models
```

## 2. Run the evaluation

```bash
rm -rf results                         # start clean
export ARGOS_COMPUTE_TYPE=float32      # required: the default int8 gives garbage for ES→EN on some CPUs
python eval_translation.py run --trials 5
```

Options: `--trials 5` (offline trials per sentence), `--online-trials 3`, `--delay 2.0`
(seconds between online requests).

This writes `results/eval_sheet.xlsx`. Do not press Ctrl+C while it runs; the file is
saved only at the end.

### If online rows show FAIL

Google may be rate-limiting or blocking the request. Check with:

```bash
python eval_translation.py probe       # one request; prints OK or STILL BLOCKED
```

When the probe prints OK (switch network, e.g. phone hotspot, if needed), re-run only
the failed rows. Offline rows and your ratings are kept:

```bash
python eval_translation.py run --retry-failed
```

## 3. Rate the translations

Open `results/eval_sheet.xlsx`, sheet **Translations**. In the **Correct (Y/N)** column,
enter `Y` if the output conveys the source meaning correctly, otherwise `N`. Leave failed
rows blank. Save and close the file.

## 4. Generate plots and tables

```bash
python eval_translation.py analyze
```

Output in `results/`:

| File | Content |
|---|---|
| `summary.csv` | Accuracy with 95% CI, chrF, failures, latency mean/SD/median/p95 per direction and mode |
| `failure_examples.csv` | Every incorrect or failed translation |
| `figs/fig1_accuracy.png` | Correctness per direction, offline vs online |
| `figs/fig2_latency_box.png` | Latency distribution per direction and mode |
| `figs/fig3_accuracy_by_length.png` | Correctness vs sentence length |
| `figs/fig4_latency_vs_length.png` | Latency vs sentence length |
