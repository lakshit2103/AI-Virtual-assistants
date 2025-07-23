# FINAL FIXED & OPTIMIZED VERSION
# AI-Powered Virtual Teaching Assistant with Chat, Voice, Quiz, Math Solver, Student Mode

import io, os, re, json, threading, datetime, requests, pdfplumber
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from deep_translator import GoogleTranslator
from duckduckgo_search import DDGS
import wikipedia, sympy as sp
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
from dotenv import load_dotenv

# ---------------- Setup ----------------
st.set_page_config(page_title="AI Teaching Assistant", layout="wide")
load_dotenv()

# Models
HF_MODELS = {"Flan-T5-Base (HF)": "google/flan-t5-base"}
OLLAMA_MODELS = {}  # Optional
ALL_MODELS = {**HF_MODELS, **OLLAMA_MODELS}

# ---------------- Text-to-Speech ----------------
def init_tts():
    try:
        tts = pyttsx3.init()
        tts.setProperty("rate", 170)
        return tts
    except Exception:
        return None

tts_engine = init_tts()
tts_lock = threading.Lock()

def speak_async(text):
    if not tts_engine:
        return
    def _speak():
        with tts_lock:
            try:
                tts_engine.say(text)
                tts_engine.runAndWait()
            except Exception:
                pass
    threading.Thread(target=_speak, daemon=True).start()

# ---------------- Helper Functions ----------------
def safe_translate(text, lang):
    try:
        return text if lang == "en" else GoogleTranslator().translate(text, target=lang)
    except:
        return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

@st.cache_resource(show_spinner=False)
def load_hf_pipe(model_id):
    try:
        return pipeline("text2text-generation", model=model_id)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

def solve_math(q):
    try:
        x = sp.symbols("x")
        if "=" in q:
            lhs, rhs = q.split("=", 1)
            return str(sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs))))
        if "integrate" in q:
            return str(sp.integrate(sp.sympify(q.split("integrate", 1)[-1]), x)) + " + C"
        if "differentiate" in q:
            return str(sp.diff(sp.sympify(q.split("differentiate", 1)[-1]), x))
        return str(sp.simplify(sp.sympify(q)))
    except Exception as e:
        return f"Math error: {e}"

def realtime_answer(q):
    ql = q.lower()
    if "bitcoin" in ql and "price" in ql:
        try:
            res = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json", timeout=5)
            rate = res.json()["bpi"]["USD"]["rate"]
            return f"Bitcoin price ≈ ${rate}"
        except:
            return None
    try:
        return wikipedia.summary(q, sentences=2)
    except:
        pass
    try:
        with DDGS() as d:
            res = list(d.text(q, max_results=1))
            return res[0]['body'] if res else None
    except:
        return None

def build_prompt(q, ctx=""):
    base = (
        "You are a knowledgeable assistant.\n"
        "Answer in 2+ detailed paragraphs with examples.\n"
    )
    if ctx:
        base += f"Context: {ctx}\n"
    return base + f"Question: {q}\nAnswer:"

def model_answer(model, prompt):
    if model in HF_MODELS:
        pipe = load_hf_pipe(HF_MODELS[model])
        if pipe:
            res = pipe(prompt, max_new_tokens=500, do_sample=False)
            return res[0]["generated_text"].strip()
    return "Model not available."

def student_view(q, a):
    prompt = (
        f"Rewrite the answer below for a 14-year-old student.\n"
        f"Original Question: {q}\n"
        f"Original Answer: {a}\n\n"
        "1. Simplified explanation\n2. Examples\n3. Diagram idea\n4. Mnemonic\n5. 3 Practice questions"
    )
    pipe = load_hf_pipe(HF_MODELS["Flan-T5-Base (HF)"])
    if pipe:
        res = pipe(prompt, max_new_tokens=500)
        return res[0]["generated_text"].strip()
    return a

# ---------------- State Init ----------------
ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("model", list(ALL_MODELS)[0])
ss.setdefault("voice_text", "")
ss.setdefault("quiz_history", [])

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")
ss.model = st.sidebar.selectbox("Model", list(ALL_MODELS))
lang = st.sidebar.selectbox("🌍 Output Language", ["en", "hi", "fr", "de"])
student_mode = st.sidebar.toggle("🎓 Student Mode")
speak_enabled = st.sidebar.toggle("🔊 Speak Answer")
voice_mode = st.sidebar.toggle("🎙️ Voice Input")

# ---------------- Main ----------------
st.title("🤖 Virtual Teaching Assistant")
query = ""

if voice_mode:
    if st.button("🎤 Speak Now"):
        r = sr.Recognizer()
        try:
            with sr.Microphone() as src:
                st.info("Listening...")
                r.adjust_for_ambient_noise(src, duration=1)
                audio = r.listen(src, timeout=10, phrase_time_limit=15)
                ss.voice_text = r.recognize_google(audio)
                st.success(f"You said: {ss.voice_text}")
                st.rerun()
        except Exception as e:
            st.error(f"Voice input error: {e}")
    query = ss.voice_text
else:
    query = st.text_input("Ask your question")

if st.button("✅ Submit") and query.strip():
    q = query.strip()
    lang_detected = detect_language(q)
    q_en = safe_translate(q, "en") if lang_detected != "en" else q
    if re.search(r"[\d\+\-\*/=]|integrate|differentiate", q_en, re.I):
        answer = solve_math(q_en)
    else:
        answer = realtime_answer(q_en) or model_answer(ss.model, build_prompt(q_en))
    if student_mode:
        answer = student_view(q_en, answer)
    answer_disp = safe_translate(answer, lang)
    ss.chat.append((q, answer_disp))
    if speak_enabled:
        speak_async(answer_disp)
    ss.voice_text = ""
    st.rerun()

st.markdown("---")
st.subheader("💬 Conversation History")
for user_msg, bot_msg in ss.chat[-5:][::-1]:
    st.markdown(f"**👤 You:** {user_msg}")
    st.markdown(f"**🤖 Assistant:** {bot_msg}")
    st.markdown("---")
