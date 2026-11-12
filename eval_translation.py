#!/usr/bin/env python3
"""
Translation-pipeline evaluation (text only: no ASR, no TTS).

  python eval_translation.py run [--trials 5] [--online-trials 3] [--delay 2.0]
        -> results/eval_sheet.xlsx
  (rater fills the 'Correct (Y/N)' column with Y or N)
  python eval_translation.py run --retry-failed
        -> re-runs ONLY rows whose status is FAIL (keeps good rows and your ratings)
  python eval_translation.py analyze
        -> results/figs/*.png, results/summary.csv, results/failure_examples.csv

Requires: pip install deep-translator argostranslate openpyxl pandas matplotlib numpy
Optional: pip install sacrebleu   (adds an automatic chrF score vs. the parallel reference)
"""
import argparse
import datetime
import os
import platform
import random
import socket
import statistics
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

OUT_DIR = "results"
XLSX = os.path.join(OUT_DIR, "eval_sheet.xlsx")
FIG_DIR = os.path.join(OUT_DIR, "figs")

# Parallel test set: sentence i has the same meaning in EN, ES and HI.
# Length buckets (by English word count): short <=5, medium 6-11, long >=12.
# Each non-English side doubles as the human reference for the other direction.
TEST_SET = [
    ("Hello, how are you?", "Hola, ¿cómo estás?", "नमस्ते, आप कैसे हैं?"),
    ("Where is the train station?", "¿Dónde está la estación de tren?", "रेलवे स्टेशन कहाँ है?"),
    ("I need some water.", "Necesito un poco de agua.", "मुझे थोड़ा पानी चाहिए।"),
    ("What time is it?", "¿Qué hora es?", "अभी क्या समय हुआ है?"),
    ("Thank you for your help.", "Gracias por tu ayuda.", "आपकी मदद के लिए धन्यवाद।"),
    ("I would like to book a table for two people tonight.",
     "Quisiera reservar una mesa para dos personas esta noche.",
     "मैं आज रात दो लोगों के लिए एक मेज़ बुक करना चाहता हूँ।"),
    ("The meeting has been moved to tomorrow morning.",
     "La reunión se ha cambiado para mañana por la mañana.",
     "बैठक कल सुबह के लिए बदल दी गई है।"),
    ("Can you tell me how to get to the hospital?",
     "¿Puede decirme cómo llegar al hospital?",
     "क्या आप मुझे बता सकते हैं कि अस्पताल कैसे पहुँचें?"),
    ("My flight was delayed because of the heavy rain.",
     "Mi vuelo se retrasó por la fuerte lluvia.",
     "भारी बारिश के कारण मेरी उड़ान देर से चली।"),
    ("She teaches mathematics at the local school.",
     "Ella enseña matemáticas en la escuela local.",
     "वह स्थानीय स्कूल में गणित पढ़ाती है।"),
    ("If you have any questions about the project, please send me an email before Friday.",
     "Si tienes alguna pregunta sobre el proyecto, por favor envíame un correo electrónico antes del viernes.",
     "यदि आपके पास परियोजना के बारे में कोई प्रश्न है, तो कृपया शुक्रवार से पहले मुझे ईमेल भेजें।"),
    ("The doctor said that I should drink more water and sleep at least eight hours every night.",
     "El médico dijo que debería beber más agua y dormir al menos ocho horas cada noche.",
     "डॉक्टर ने कहा कि मुझे ज़्यादा पानी पीना चाहिए और हर रात कम से कम आठ घंटे सोना चाहिए।"),
    ("When we arrived at the airport, we discovered that our luggage had been sent to a different city.",
     "Cuando llegamos al aeropuerto, descubrimos que nuestro equipaje había sido enviado a otra ciudad.",
     "जब हम हवाई अड्डे पहुँचे, तो हमें पता चला कि हमारा सामान किसी दूसरे शहर भेज दिया गया था।"),
    ("Although the restaurant was crowded, the waiter found us a quiet table near the window.",
     "Aunque el restaurante estaba lleno, el camarero nos encontró una mesa tranquila junto a la ventana.",
     "हालाँकि रेस्तरां में बहुत भीड़ थी, फिर भी वेटर ने हमें खिड़की के पास एक शांत मेज़ दिला दी।"),
    ("Our company is planning to open a new office in the city next year to support more customers.",
     "Nuestra empresa planea abrir una nueva oficina en la ciudad el próximo año para atender a más clientes.",
     "हमारी कंपनी अगले साल अधिक ग्राहकों की सेवा करने के लिए शहर में एक नई शाखा खोलने की योजना बना रही है।"),
]
LANG_IDX = {"en": 0, "es": 1, "hi": 2}
DIRECTIONS = [("hi", "en"), ("en", "hi"), ("es", "en"), ("en", "es")]
MODES = ["OFFLINE", "ONLINE"]
BUCKETS = ["short (<=5)", "medium (6-11)", "long (>=12)"]

HEADER = ["row_id", "direction", "mode", "sentence_id", "source_text", "source_words",
          "reference_(parallel)", "system_output", "chrF_(auto)", "latency_ms_mean",
          "latency_ms_sd", "ok_trials", "status", "Correct (Y/N)", "rater_notes"]


def length_bucket(sid):
    n = len(TEST_SET[int(sid) - 1][0].split())
    return BUCKETS[0] if n <= 5 else (BUCKETS[1] if n <= 11 else BUCKETS[2])


# ---------------------------------------------------------------------------
# Pipeline code copied from translator_refactored.py so behaviour is identical
# ---------------------------------------------------------------------------
def format_translation(text):
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    first = text.split()[0].lower() if text.split() else ""
    if first in ["how", "what", "why", "when", "where", "who", "which", "can", "could", "is", "are"]:
        if not text.endswith("?"):
            text += "?"
    elif not text.endswith((".", "!", "?")):
        text += "."
    return text


def build_offline():
    try:
        from argostranslate import translate as at
        langs = at.get_installed_languages()
    except Exception as e:
        print("Argos unavailable:", e)
        return {}
    out = {}
    for s, t in DIRECTIONS:
        try:
            sl = next(l for l in langs if l.code == s)
            tl = next(l for l in langs if l.code == t)
            out[(s, t)] = sl.get_translation(tl)
        except StopIteration:
            print(f"WARNING: Argos model {s}->{t} not installed")
    return out


def offline_translate(tr, text, tgt):
    # Argos wraps translations in CachedTranslation: a repeat of the same text returns in
    # microseconds. Clear it so every timed call is a real (uncached) translation.
    cache = getattr(tr, "cache", None)
    if isinstance(cache, dict):
        cache.clear()
    out = tr.translate(text)
    return format_translation(out) if tgt == "en" else out   # same as app


def online_translate(src, tgt, text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source=src, target=tgt).translate(text)  # app applies no formatting


# ---------------------------------------------------------------------------
# Online-call helpers: pacing, retry with backoff
# ---------------------------------------------------------------------------
def is_rate_limit(e):
    s = str(e).lower()
    return "too many requests" in s or "429" in s or type(e).__name__ == "TooManyRequests"


def pace(delay):
    """Sleep delay + random jitter so requests are not evenly spaced."""
    time.sleep(delay + random.uniform(0, 0.5 * delay))


def online_call(src, tgt, text, max_attempts=3):
    """Returns (output, ms). Only the successful attempt is timed; backoff waits are not."""
    last = None
    for a in range(max_attempts):
        try:
            t0 = time.perf_counter()
            out = online_translate(src, tgt, text)
            return out, (time.perf_counter() - t0) * 1000
        except Exception as e:
            last = e
            if a == max_attempts - 1:
                break
            wait = 15 * (3 ** a) if is_rate_limit(e) else 2   # 15 s, 45 s for rate limits
            print(f"    {type(e).__name__}; retrying in {wait}s")
            time.sleep(wait)
    raise last


# ---------------------------------------------------------------------------
# Network measurements
# ---------------------------------------------------------------------------
def measure_ping(n=10, host="8.8.8.8", port=53):
    """TCP connect RTT in ms (same probe the app uses for is_connected)."""
    vals = []
    for _ in range(n):
        try:
            t = time.perf_counter()
            socket.create_connection((host, port), timeout=2).close()
            vals.append((time.perf_counter() - t) * 1000)
        except OSError:
            pass
        time.sleep(0.2)
    return vals


def measure_download_mbps(nbytes=5_000_000):
    import ssl
    url = f"https://speed.cloudflare.com/__down?bytes={nbytes}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    contexts = []
    try:
        import certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    contexts.append(None)                              # system default
    contexts.append(ssl._create_unverified_context())  # last resort (macOS cert issues)
    for ctx in contexts:
        try:
            t = time.perf_counter()
            data = urllib.request.urlopen(req, timeout=30, context=ctx).read()
            return len(data) * 8 / 1e6 / (time.perf_counter() - t)
        except Exception as e:
            last = e
    print("  download test failed:", last)
    return None


# ---------------------------------------------------------------------------
# RUN phase
# ---------------------------------------------------------------------------
def load_previous():
    """For --retry-failed: keep rows with status OK (and their ratings), re-run the rest."""
    prev_rows, prev_trials = {}, []
    if not os.path.exists(XLSX):
        print("No existing sheet found; running everything.")
        return prev_rows, prev_trials
    old = pd.read_excel(XLSX, sheet_name="Translations")
    oldt = pd.read_excel(XLSX, sheet_name="Latency_Trials")
    for rec in old.to_dict("records"):
        if str(rec["status"]).startswith("OK"):
            key = (rec["direction"], rec["mode"], int(rec["sentence_id"]))
            prev_rows[key] = ["" if pd.isna(rec[h]) else rec[h] for h in HEADER]
    for rec in oldt.to_dict("records"):
        key = (rec["direction"], rec["mode"], int(rec["sentence_id"]))
        if key in prev_rows:
            lat = None if pd.isna(rec["latency_ms"]) else float(rec["latency_ms"])
            prev_trials.append((rec["direction"], rec["mode"], int(rec["sentence_id"]),
                                int(rec["trial"]), lat, int(rec["ok"])))
    print(f"Keeping {len(prev_rows)} good rows from the previous run.")
    return prev_rows, prev_trials


def run(trials, online_trials, delay, retry_failed):
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.styles import PatternFill, Font
    try:
        import sacrebleu
    except ImportError:
        sacrebleu = None

    os.makedirs(OUT_DIR, exist_ok=True)
    offline = build_offline()
    prev_rows, prev_trials = load_previous() if retry_failed else ({}, [])

    print("Measuring network...")
    ping = measure_ping()
    mbps = measure_download_mbps()
    ping_txt = f"{np.mean(ping):.1f} ms" if ping else "n/a"
    mbps_txt = f"{mbps:.1f} Mbps" if mbps else "n/a"
    print(f"  ping mean={ping_txt}, download={mbps_txt}")

    new_rows, new_trials = {}, []
    state = {"consec": 0, "dead": False}   # circuit breaker for online throttling
    skip_msg = "SKIPPED: online throttled; rerun with --retry-failed later"

    for (src, tgt) in DIRECTIONS:
        dname = f"{src.upper()}->{tgt.upper()}"
        for mode in MODES:
            pending = [sid for sid in range(1, len(TEST_SET) + 1) if (dname, mode, sid) not in prev_rows]
            if not pending:
                continue
            n_tr = trials if mode == "OFFLINE" else online_trials

            # warm-up so model load / first-connection cost is not counted
            try:
                if mode == "OFFLINE" and (src, tgt) in offline:
                    offline_translate(offline[(src, tgt)], "test", tgt)
                elif mode == "ONLINE" and not state["dead"]:
                    pace(delay)
                    online_call(src, tgt, "test")
            except Exception:
                pass

            for sid in pending:
                sent = TEST_SET[sid - 1]
                source, ref = sent[LANG_IDX[src]], sent[LANG_IDX[tgt]]
                lats, first_out, err = [], "", ""
                for k in range(n_tr):
                    if mode == "ONLINE" and state["dead"]:
                        err = skip_msg
                        break
                    try:
                        if mode == "OFFLINE":
                            if (src, tgt) not in offline:
                                raise RuntimeError("Argos model missing")
                            t0 = time.perf_counter()
                            out = offline_translate(offline[(src, tgt)], source, tgt)
                            ms = (time.perf_counter() - t0) * 1000
                        else:
                            pace(delay)
                            out, ms = online_call(src, tgt, source)
                        if not out or not out.strip():
                            raise RuntimeError("empty output")
                        lats.append(ms)
                        first_out = first_out or out
                        new_trials.append((dname, mode, sid, k + 1, ms, 1))
                        if mode == "ONLINE":
                            state["consec"] = 0
                    except Exception as e:
                        err = f"{type(e).__name__}: {str(e)[:70]}"
                        new_trials.append((dname, mode, sid, k + 1, None, 0))
                        if mode == "ONLINE":
                            state["consec"] += 1
                            if state["consec"] >= 3:
                                state["dead"] = True
                                print("\n!! 3 consecutive online failures: Google is throttling this IP.\n"
                                      "   Skipping the remaining online requests. Wait 30-60 min (or switch\n"
                                      "   network/hotspot) and run:  python eval_translation.py run --retry-failed\n")
                chrf = ""
                if sacrebleu and first_out:
                    chrf = round(sacrebleu.sentence_chrf(first_out, [ref]).score, 1)
                ok = len(lats) > 0
                new_rows[(dname, mode, sid)] = [
                    f"{dname}|{mode}|{sid}", dname, mode, sid, source, len(source.split()),
                    ref, first_out, chrf,
                    round(statistics.mean(lats), 2) if ok else "",
                    round(statistics.stdev(lats), 2) if len(lats) > 1 else "",
                    len(lats), "OK" if ok else f"FAIL: {err}",
                    "",   # rater fills Y/N; failed rows stay blank and are excluded from accuracy
                    "",
                ]
            print(f"done {dname} {mode}")

    rows = []
    for (src, tgt) in DIRECTIONS:
        dname = f"{src.upper()}->{tgt.upper()}"
        for mode in MODES:
            for sid in range(1, len(TEST_SET) + 1):
                key = (dname, mode, sid)
                rows.append(prev_rows.get(key) or new_rows[key])
    trial_rows = prev_trials + new_trials

    wb = Workbook()
    ws = wb.active
    ws.title = "Translations"
    ws.append(HEADER)
    for r in rows:
        ws.append(r)
    for c in ws[1]:
        c.font = Font(bold=True)
    ws["N1"].fill = PatternFill("solid", fgColor="FFFF00")
    dv = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"N2:N{len(rows)+1}")
    for col, w in zip("ABCDEFGHIJKLMNO", [20, 10, 9, 8, 50, 8, 50, 50, 9, 11, 11, 9, 14, 12, 30]):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"

    wt = wb.create_sheet("Latency_Trials")
    wt.append(["direction", "mode", "sentence_id", "trial", "latency_ms", "ok"])
    for r in trial_rows:
        wt.append(list(r))

    wm = wb.create_sheet("Meta")
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except Exception:
            return "n/a"
    en_len = [len(s[0].split()) for s in TEST_SET]
    meta = [
        ("timestamp", datetime.datetime.now().isoformat(timespec="seconds")),
        ("platform", platform.platform()), ("python", sys.version.split()[0]),
        ("cpu", platform.processor()),
        ("argostranslate", ver("argostranslate")), ("deep-translator", ver("deep-translator")),
        ("n_sentences", len(TEST_SET)),
        ("mean_source_words_EN", round(np.mean(en_len), 1)),
        ("mean_source_words_ES", round(np.mean([len(s[1].split()) for s in TEST_SET]), 1)),
        ("mean_source_words_HI", round(np.mean([len(s[2].split()) for s in TEST_SET]), 1)),
        ("trials_per_sentence_offline", trials),
        ("trials_per_sentence_online", online_trials),
        ("online_request_spacing_s", f"{delay} + up to {0.5*delay} jitter"),
        ("ran_with_retry_failed", retry_failed),
        ("ping_ms_mean", round(np.mean(ping), 1) if ping else "n/a"),
        ("ping_ms_sd", round(np.std(ping, ddof=1), 1) if len(ping) > 1 else "n/a"),
        ("ping_samples", len(ping)),
        ("download_mbps", round(mbps, 1) if mbps else "n/a"),
    ]
    for m in meta:
        wm.append(list(m))
    wb.save(XLSX)
    n_fail = sum(1 for r in rows if not str(r[12]).startswith("OK"))
    print(f"\nSaved {XLSX}  ({n_fail} rows FAILED)")
    if n_fail:
        print(f"Re-run only the failed rows with: python {sys.argv[0]} run --retry-failed")
    print(f"Then fill the 'Correct (Y/N)' column, save, and run: python {sys.argv[0]} analyze")


# ---------------------------------------------------------------------------
# ANALYZE phase
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_excel(XLSX, sheet_name="Translations")
    lt = pd.read_excel(XLSX, sheet_name="Latency_Trials")
    df["rating"] = df["Correct (Y/N)"].astype(str).str.strip().str.upper()
    df["is_ok"] = df["status"].astype(str).str.startswith("OK")
    df["rated"] = df["is_ok"] & df["rating"].isin(["Y", "N"])
    df["is_correct"] = (df["rating"] == "Y").astype(int)
    df["length_bucket"] = df["sentence_id"].map(length_bucket)
    df["lat"] = pd.to_numeric(df["latency_ms_mean"], errors="coerce")

    unrated = (df["is_ok"] & ~df["rated"]).sum()
    failed = (~df["is_ok"]).sum()
    if unrated:
        print(f"WARNING: {unrated} successful translations have no Y/N rating and are excluded from accuracy.")
    if failed:
        print(f"NOTE: {failed} rows failed (no output); they are excluded from accuracy and reported separately.")
    rated = df[df["rated"]]

    # ---- summary table
    rows = []
    for (d, m), g in df.groupby(["direction", "mode"]):
        r = g[g["rated"]]
        k, n = int(r["is_correct"].sum()), len(r)
        lo, hi = wilson(k, n)
        sel = (lt.direction == d) & (lt["mode"] == m)
        tl = lt[sel & (lt.ok == 1)]["latency_ms"]
        rows.append(dict(
            direction=d, mode=m, n_sentences=len(g), n_rated=n, n_correct=k,
            accuracy_pct=100 * k / n if n else np.nan, ci95_lo=100 * lo, ci95_hi=100 * hi,
            failed_translations=int((~g["is_ok"]).sum()),
            mean_chrF=pd.to_numeric(g.loc[g["is_ok"], "chrF_(auto)"], errors="coerce").mean(),
            n_trials=int(sel.sum()), failed_trials=int(sel.sum() - len(tl)),
            lat_mean_ms=tl.mean(), lat_sd_ms=tl.std(), lat_median_ms=tl.median(),
            lat_p95_ms=tl.quantile(0.95)
        ))
    S = pd.DataFrame(rows).round(2)
    S.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print(S.to_string(index=False))

    dirs = [f"{s.upper()}->{t.upper()}" for s, t in DIRECTIONS]
    colors = {"OFFLINE": "#4C72B0", "ONLINE": "#DD8452"}
    w = 0.38

    # ---- Fig 1: accuracy with Wilson CI
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(MODES):
        sub = S[S["mode"] == m].set_index("direction").reindex(dirs)
        x = np.arange(len(dirs)) + (i - 0.5) * w
        acc = sub.accuracy_pct.fillna(0).values
        err = [np.nan_to_num(acc - sub.ci95_lo.values), np.nan_to_num(sub.ci95_hi.values - acc)]
        ax.bar(x, acc, w, yerr=err, capsize=3, label=m.title(), color=colors[m])
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels(dirs)
    ax.set_ylabel("Human-judged correct (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Translation correctness (95% Wilson CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_accuracy.png"), dpi=300)
    plt.close(fig)

    # ---- Fig 2: latency distributions
    lok = lt[lt.ok == 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    pos, data, cols, ticks = [], [], [], []
    for j, d in enumerate(dirs):
        for i, m in enumerate(MODES):
            v = lok[(lok.direction == d) & (lok["mode"] == m)]["latency_ms"].values
            data.append(v if len(v) else [np.nan])
            pos.append(j * 3 + i)
            cols.append(colors[m])
        ticks.append(j * 3 + 0.5)
    bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    ax.set_xticks(ticks)
    ax.set_xticklabels(dirs)
    ax.set_yscale("log")
    ax.set_ylabel("Translation latency (ms, log scale)")
    ax.set_title("Per-sentence translation latency, all successful trials")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in MODES], labels=["Offline", "Online"])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_latency_box.png"), dpi=300)
    plt.close(fig)

    # ---- Fig 3: accuracy by sentence length
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(MODES):
        acc = [100 * rated[(rated["mode"] == m) & (rated.length_bucket == b)]["is_correct"].mean() for b in BUCKETS]
        ax.bar(np.arange(3) + (i - 0.5) * w, np.nan_to_num(acc), w, label=m.title(), color=colors[m])
    ax.set_xticks(range(3))
    ax.set_xticklabels(BUCKETS)
    ax.set_xlabel("Source length (English word count of the sentence)")
    ax.set_ylabel("Correct (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Correctness vs. sentence length (all directions pooled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_accuracy_by_length.png"), dpi=300)
    plt.close(fig)

    # ---- Fig 4: latency vs length (uses all successful rows, no rating needed)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in MODES:
        g = df[(df["mode"] == m) & df["is_ok"]]
        en_words = g["sentence_id"].map(lambda s: len(TEST_SET[int(s) - 1][0].split()))
        ax.scatter(en_words, g["lat"], s=18, alpha=0.6, label=m.title(), color=colors[m])
    ax.set_yscale("log")
    ax.set_xlabel("Sentence length (English words)")
    ax.set_ylabel("Mean latency (ms, log)")
    ax.set_title("Latency vs. sentence length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_latency_vs_length.png"), dpi=300)
    plt.close(fig)

    # ---- Failure examples for the paper (wrong translations AND no-output failures)
    bad = df[((df["rated"]) & (df["is_correct"] == 0)) | (~df["is_ok"])]
    bad = bad[["direction", "mode", "status", "source_text", "reference_(parallel)", "system_output", "rater_notes"]]
    bad.to_csv(os.path.join(OUT_DIR, "failure_examples.csv"), index=False)
    print(f"\nFigures in {FIG_DIR}/, summary.csv and failure_examples.csv in {OUT_DIR}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "analyze"])
    ap.add_argument("--trials", type=int, default=5, help="trials per sentence, OFFLINE mode")
    ap.add_argument("--online-trials", type=int, default=3, help="trials per sentence, ONLINE mode")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between online requests (+50%% jitter)")
    ap.add_argument("--retry-failed", action="store_true", help="re-run only rows whose status is FAIL")
    a = ap.parse_args()
    if a.phase == "run":
        run(a.trials, a.online_trials, a.delay, a.retry_failed)
    else:
        analyze()
