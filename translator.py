import os
import json
import subprocess
import tempfile
import platform
import threading
import tkinter as tk
import pyaudio
import socket
import speech_recognition as sr
from deep_translator import GoogleTranslator

from vosk import Model, KaldiRecognizer
from argostranslate import translate as argostranslate
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from wordfreq import zipf_frequency

MODE = "OFFLINE"
LANG_PAIR = "HI_EN"
LANG_MODE = "HI_TO_EN"

def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1.5)
        return True
    except OSError:
        return False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_model(lang_code):
    for name in os.listdir(BASE_DIR):
        lower = name.lower()
        if lower.startswith("vosk-model") and f"-{lang_code}" in lower:
            return os.path.join(BASE_DIR, name)
    return None

EN_MODEL_PATH = find_model("en")
HI_MODEL_PATH = find_model("hi")
ES_MODEL_PATH = find_model("es")

langs = argostranslate.get_installed_languages()
def get_translator(src, tgt):
    try:
        s = next(l for l in langs if l.code == src)
        t = next(l for l in langs if l.code == tgt)
        return s.get_translation(t)
    except StopIteration:
        return None

translator_hi_en = get_translator("hi", "en")
translator_en_hi = get_translator("en", "hi")
translator_es_en = get_translator("es", "en")
translator_en_es = get_translator("en", "es")

nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

AUX_VERBS = {
    "am","is","are","was","were","be","been","being",
    "do","does","did","have","has","had",
    "will","would","shall","should","may","might","must","can","could"
}

STOP_WORDS = {
    "the","a","an","in","on","at","of","to","for","from",
    "and","or","but","if","then","this","that","these","those"
}

def get_wordnet_pos(tag):
    if tag.startswith("J"): return wn.ADJ
    if tag.startswith("V"): return wn.VERB
    if tag.startswith("R"): return wn.ADV
    return None

def get_simpler_word(word, pos=None):
    if not pos:
        return word
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        return word
    candidates = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            name = lemma.name().lower()
            if "_" not in name:
                candidates.add(name)
    if not candidates:
        return word
    best = max(candidates, key=lambda w: zipf_frequency(w, "en"))
    return best if zipf_frequency(best, "en") > (zipf_frequency(word, "en") + 0.3) else word

def simplify_text(text):
    tokens = text.split()
    tagged = pos_tag(tokens)
    output = []
    for token, tag in tagged:
        clean = token.strip(".,?!:;").lower()
        if clean in AUX_VERBS or clean in STOP_WORDS or tag.startswith("N"):
            output.append(token)
            continue
        wn_pos = get_wordnet_pos(tag)
        simple = get_simpler_word(clean, wn_pos)
        if simple != clean:
            replacement = simple.capitalize() if token[0].isupper() else simple
            output.append(token.lower().replace(clean, replacement))
        else:
            output.append(token)
    return " ".join(output)

def format_translation(text):
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    first_word = text.split()[0].lower() if text.split() else ""
    if first_word in ["how", "what", "why", "when", "where", "who", "which", "can", "could", "is", "are"]:
        if not text.endswith("?"):
            text += "?"
    elif not text.endswith((".", "!", "?")):
        text += "."
    return text

if platform.system() == "Windows":
    PIPER_BIN = os.path.join(BASE_DIR, "piper_windows_amd64", "piper", "piper.exe")
    PIPER_MODEL = os.path.join(BASE_DIR, "piper_windows_amd64", "piper", "en_US-lessac-medium.onnx")
else:
    PIPER_BIN = os.path.join(BASE_DIR, "piper", "piper")
    PIPER_MODEL = os.path.join(BASE_DIR, "piper", "en_US-lessac-medium.onnx")
    PIPER_CONFIG = PIPER_MODEL + ".json"

def play_audio(text, lang="en"):
    if not text.strip():
        return
    if lang == "en" and os.path.exists(PIPER_BIN) and os.path.exists(PIPER_MODEL):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            if platform.system() == "Windows":
                subprocess.run([PIPER_BIN, "--model", PIPER_MODEL, "--output_file", tmp_wav],
                               input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(tmp_wav):
                    import winsound
                    winsound.PlaySound(tmp_wav, winsound.SND_FILENAME)
            else:
                subprocess.run([PIPER_BIN, "-m", PIPER_MODEL, "-c", PIPER_CONFIG, "-f", tmp_wav],
                               input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(tmp_wav):
                    subprocess.run(["aplay", "-q", tmp_wav])
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
    else:
        if platform.system() != "Windows":
            subprocess.run(["espeak-ng", f"-v{lang}", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

class ModernTranslatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Speech Translator")
        self.root.geometry("480x320")
        self.root.configure(bg="#000000")
        self.root.resizable(False, False)

        self.last_source = ""
        self.last_target = ""

        self.build_ui()
        threading.Thread(target=assistant_loop, args=(self,), daemon=True).start()

    def build_ui(self):
        top_frame = tk.Frame(self.root, bg="#000000")
        top_frame.pack(fill="x", pady=5)

        self.light_canvas = tk.Canvas(top_frame, width=15, height=15, bg="#000000", highlightthickness=0)
        self.light_canvas.pack(side="left", padx=10)
        self.light = self.light_canvas.create_oval(2, 2, 13, 13, fill="gray")

        self.container = tk.Frame(self.root, bg="#000000")
        self.container.pack(expand=True, pady=(0, 20))

        self.src_label = tk.Label(self.container, text="", font=("Segoe UI", 13), fg="#AAAAAA", bg="#000000", wraplength=450, justify="center")
        self.src_label.pack(pady=3)

        self.tgt_label = tk.Label(self.container, text="", font=("Segoe UI", 20, "bold"), fg="white", bg="#000000", wraplength=450, justify="center")
        self.tgt_label.pack(pady=3)

        btn_frame = tk.Frame(self.root, bg="#000000")
        btn_frame.pack(pady=(0, 20))

        tk.Button(btn_frame, text="✨ Simplify", font=("Segoe UI", 11, "bold"), bg="#222222", fg="white", relief="flat", command=self.simplify).pack(side="left", padx=6)
        tk.Button(btn_frame, text="🔄 Swap", font=("Segoe UI", 11, "bold"), bg="#222222", fg="white", relief="flat", command=self.swap_languages).pack(side="left", padx=6)
        self.mode_btn = tk.Button(btn_frame, text="🌐 Online", font=("Segoe UI", 11, "bold"), bg="#222222", fg="white", relief="flat", command=self.toggle_mode)
        self.mode_btn.pack(side="left", padx=6)
        tk.Button(btn_frame, text="🌍 Lang", font=("Segoe UI", 11, "bold"), bg="#222222", fg="white", relief="flat", command=self.toggle_language_pair).pack(side="left", padx=6)

    def set_status(self, text, indicator="gray"):
        self.root.after(0, lambda: self.tgt_label.config(text=text))
        self.root.after(0, lambda: self.src_label.config(text=""))
        self.root.after(0, lambda: self.light_canvas.itemconfig(self.light, fill=indicator))

    def show_results(self, src_text, tgt_text):
        self.last_source = src_text
        self.last_target = tgt_text
        self.root.after(0, lambda: self.src_label.config(text=src_text))
        self.root.after(0, lambda: self.tgt_label.config(text=tgt_text))

    def toggle_mode(self):
        global MODE
        MODE = "ONLINE" if MODE == "OFFLINE" else "OFFLINE"
        self.mode_btn.config(text="📴 Offline" if MODE == "ONLINE" else "🌐 Online")

    def swap_languages(self):
        global LANG_MODE
        if LANG_PAIR == "HI_EN":
            LANG_MODE = "EN_TO_HI" if LANG_MODE == "HI_TO_EN" else "HI_TO_EN"
            label = "Hindi → English" if LANG_MODE == "HI_TO_EN" else "English → Hindi"
        else:
            LANG_MODE = "EN_TO_ES" if LANG_MODE == "ES_TO_EN" else "ES_TO_EN"
            label = "Spanish → English" if LANG_MODE == "ES_TO_EN" else "English → Spanish"
        self.set_status(f"Mode: {label}")

    def toggle_language_pair(self):
        global LANG_PAIR, LANG_MODE
        if LANG_PAIR == "HI_EN":
            LANG_PAIR = "ES_EN"
            LANG_MODE = "ES_TO_EN"
            self.set_status("Mode: Spanish → English")
        else:
            LANG_PAIR = "HI_EN"
            LANG_MODE = "HI_TO_EN"
            self.set_status("Mode: Hindi → English")

    def simplify(self):
        if self.last_target and ("TO_EN" in LANG_MODE):
            simplified = simplify_text(self.last_target)
            self.tgt_label.config(text=simplified)
            threading.Thread(target=play_audio, args=(simplified, "en"), daemon=True).start()

def assistant_loop(ui):
    ui.set_status("Waiting for wake word...", "gray")

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=2048)

    en_rec = KaldiRecognizer(Model(EN_MODEL_PATH), 16000) if EN_MODEL_PATH else None
    hi_rec = KaldiRecognizer(Model(HI_MODEL_PATH), 16000) if HI_MODEL_PATH else None
    es_rec = KaldiRecognizer(Model(ES_MODEL_PATH), 16000) if ES_MODEL_PATH else None

    recognizer_online = sr.Recognizer()
    listening_mode = False

    while True:
        if MODE == "ONLINE":
            if not is_connected():
                ui.set_status("❌ No Internet Connection", "red")
                continue
            try:
                with sr.Microphone() as source:
                    audio = recognizer_online.listen(source, phrase_time_limit=4)
                src_lang = "hi-IN" if LANG_MODE == "HI_TO_EN" else ("es-ES" if LANG_MODE == "ES_TO_EN" else "en-IN")
                text = recognizer_online.recognize_google(audio, language=src_lang)
                
                src_code, tgt_code = LANG_MODE.lower().split("_to_")
                translated = GoogleTranslator(source=src_code, target=tgt_code).translate(text)
                
                ui.show_results(text, translated)
                play_audio(translated, lang=tgt_code)
            except Exception:
                pass
            continue

        data = stream.read(2048, exception_on_overflow=False)

        if not listening_mode:
            if en_rec and en_rec.AcceptWaveform(data):
                cmd = json.loads(en_rec.Result()).get("text", "").lower()
                if "hello" in cmd:
                    listening_mode = True
                    en_rec.Reset()
                    ui.set_status("Listening...", "#00FF00")
            continue

        active_rec = hi_rec if LANG_MODE == "HI_TO_EN" else (es_rec if LANG_MODE == "ES_TO_EN" else en_rec)
        if active_rec and active_rec.AcceptWaveform(data):
            spoken_text = json.loads(active_rec.Result()).get("text", "").strip()
            if not spoken_text:
                continue

            if LANG_MODE.startswith("EN") and any(w in spoken_text.lower() for w in ["stop", "pause", "exit"]):
                listening_mode = False
                ui.set_status("Waiting for wake word...", "gray")
                continue

            if LANG_MODE == "HI_TO_EN" and translator_hi_en:
                translated = format_translation(translator_hi_en.translate(spoken_text))
                tgt_lang = "en"
            elif LANG_MODE == "EN_TO_HI" and translator_en_hi:
                translated = translator_en_hi.translate(spoken_text)
                tgt_lang = "hi"
            elif LANG_MODE == "ES_TO_EN" and translator_es_en:
                translated = format_translation(translator_es_en.translate(spoken_text))
                tgt_lang = "en"
            elif LANG_MODE == "EN_TO_ES" and translator_en_es:
                translated = translator_en_es.translate(spoken_text)
                tgt_lang = "es"
            else:
                translated = spoken_text
                tgt_lang = "en"

            ui.show_results(spoken_text, translated)
            threading.Thread(target=play_audio, args=(translated, tgt_lang), daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernTranslatorUI(root)
    root.mainloop()