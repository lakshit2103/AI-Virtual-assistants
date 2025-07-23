import io, re, threading, datetime, requests, pdfplumber, random
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect
from deep_translator import GoogleTranslator
from duckduckgo_search import DDGS
import wikipedia, sympy as sp
import speech_recognition as sr
import pyttsx3
import pythoncom  # For Windows COM initialization
from transformers import pipeline
import torch
import time
import json

# Streamlit Configuration
st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

# Model Configuration - Using best open-source model for text generation and quiz creation
MODEL = "microsoft/DialoGPT-large"  # Best available conversational model for teaching assistant

# Model Loading (Cached) - Optimized for teaching and quiz generation
@st.cache_resource(show_spinner=False)
def load_hf_pipe():
    try:
        # Use text-generation pipeline for quiz creation
        return pipeline("text-generation", model=MODEL, max_length=2048, device=-1, 
                       do_sample=True, temperature=0.7)
    except Exception as e:
        st.error(f"Error loading model {MODEL}: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_qa_pipe():
    try:
        # Use question-answering pipeline for better responses
        return pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
    except Exception as e:
        st.warning(f"QA model not available: {e}")
        return None

# Text-to-Speech (Windows COM Fix)
def init_tts():
    try:
        pythoncom.CoInitialize()
        tts = pyttsx3.init()
        tts.setProperty("rate", 150)
        voices = tts.getProperty("voices")
        if voices:
            tts.setProperty("voice", voices[0].id)
        return tts
    except Exception as e:
        st.warning(f"TTS unavailable: {e}")
        return None

def speak_with_sapi(text: str):
    """Alternative TTS using Windows SAPI"""
    try:
        import subprocess
        import tempfile
        import os
        
        vbs_script = f'''
        Set speech = CreateObject("SAPI.SpVoice")
        speech.Rate = 1
        speech.Speak "{text[:200]}"
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vbs', delete=False) as f:
            f.write(vbs_script)
            temp_file = f.name
        
        subprocess.run(['cscript', '//nologo', temp_file], 
                      creationflags=subprocess.CREATE_NO_WINDOW)
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"SAPI TTS error: {e}")

def speak_async(text: str):
    if not text:
        return
    
    def _run():
        try:
            pythoncom.CoInitialize()
            tts = pyttsx3.init()
            tts.setProperty("rate", 150)
            voices = tts.getProperty("voices")
            if voices:
                tts.setProperty("voice", voices[0].id)
            tts.say(text[:200])
            tts.runAndWait()
            pythoncom.CoUninitialize()
            
        except Exception:
            try:
                speak_with_sapi(text)
            except Exception as e:
                print(f"All TTS methods failed: {e}")
    
    threading.Thread(target=_run, daemon=True).start()

# Helper Functions
def safe_translate(text: str, target_lang: str) -> str:
    if target_lang == "en" or not text:
        return text
    try:
        translator = GoogleTranslator(source="auto", target=target_lang)
        return translator.translate(text[:2000])
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text

def detect_language(text: str) -> str:
    try:
        return detect(text) if text else "en"
    except Exception:
        return "en"

MATH_RE = re.compile(r"(integrate|differentiate|solve|derive|\d+[\+\-\*/\^=])", re.I)

# PDF Processing
def pdf_to_lines(data: bytes) -> list:
    if not data:
        return []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return [line.strip() for page in pdf.pages for line in (page.extract_text() or "").split("\n") if line.strip()]
    except Exception as e:
        st.error(f"PDF processing failed: {e}")
        return []

def find_context(query: str, lines: list) -> str:
    if not lines or not query:
        return ""
    query_words = query.lower().split()
    for line in lines:
        if any(word in line.lower() for word in query_words):
            return line
    return ""

# Math Solver
def solve_math(q: str) -> str:
    try:
        x = sp.symbols('x')
        if "=" in q:
            lhs, rhs = q.split("=", 1)
            sol = sp.solve(sp.Eq(sp.sympify(lhs), sp.sympify(rhs)))
            return f"Solutions: {sol}"
        if "integrate" in q.lower():
            expr = q.lower().split("integrate", 1)[-1].strip()
            return f"∫({expr})dx = {sp.integrate(sp.sympify(expr), x)} + C"
        if "differentiate" in q.lower():
            expr = q.lower().split("differentiate", 1)[-1].strip()
            return f"d/dx({expr}) = {sp.diff(sp.sympify(expr), x)}"
        return str(sp.simplify(sp.sympify(q)))
    except Exception as e:
        return f"Math error: {e}. Please check your expression."

# Real-time Information
def get_realtime_info(q: str) -> str:
    ql = q.lower()
    
    # Bitcoin price
    if "bitcoin" in ql and "price" in ql:
        try:
            if "inr" in ql:
                rate = requests.get("https://api.coindesk.com/v1/bpi/currentprice/INR.json", timeout=5).json()["bpi"]["INR"]["rate"]
                return f"Bitcoin price: ₹{rate}"
            else:
                rate = requests.get("https://api.coindesk.com/v1/bpi/currentprice.json", timeout=5).json()["bpi"]["USD"]["rate"]
                return f"Bitcoin price: ${rate}"
        except Exception:
            pass
    
    # Current PM of India
    if "prime minister of india" in ql:
        return "Narendra Modi is the current Prime Minister of India (since 2014)."
    
    # Wikipedia search
    try:
        search_term = q.strip()
        if len(search_term) > 5:
            return wikipedia.summary(search_term, sentences=2)
    except Exception:
        pass
    
    # DuckDuckGo search
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=1))
            if results:
                return results[0]["body"][:300]
    except Exception:
        pass
    
    return ""

# Enhanced AI Model Response with better context handling
def get_ai_response(prompt: str) -> str:
    try:
        # Try QA pipeline first for better responses
        qa_pipe = load_qa_pipe()
        if qa_pipe and "?" in prompt:
            # Extract question from prompt
            question = prompt.split("Question:")[-1].split("Answer:")[0].strip()
            context = prompt.split("Context:")[-1].split("Question:")[0].strip() if "Context:" in prompt else ""
            
            if context:
                try:
                    result = qa_pipe(question=question, context=context)
                    if result['score'] > 0.1:  # Only use if confidence is reasonable
                        return result['answer']
                except:
                    pass
        
        # Fallback to text generation
        pipe = load_hf_pipe()
        if not pipe:
            return "AI model not available. Please try again."
        
        result = pipe(
            prompt, 
            max_new_tokens=200, 
            do_sample=True, 
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe.tokenizer, 'eos_token_id') else 50256,
            return_full_text=False
        )
        
        response = result[0]["generated_text"]
        
        # Clean up the response
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
        else:
            answer = response.strip()
        
        # Remove any remaining prompt fragments
        lines = answer.split('\n')
        clean_lines = []
        for line in lines:
            if not line.strip().startswith(('Question:', 'You are', 'Context:', 'Generate')):
                clean_lines.append(line)
        
        final_answer = '\n'.join(clean_lines).strip()
        return final_answer if final_answer else "I'm sorry, I couldn't generate a proper response."
        
    except Exception as e:
        return f"Error generating response: {e}"

# AI Model Response
def create_prompt(question: str, context: str = "") -> str:
    base_prompt = (
        "You are a helpful AI teaching assistant. Provide clear, educational answers "
        "suitable for high school students. Keep responses concise but informative.\n\n"
    )
    
    if context:
        base_prompt += f"Context: {context}\n\n"
    
    return f"{base_prompt}Question: {question}\nAnswer:"
def enhance_for_students(question: str, answer: str) -> str:
    enhanced_prompt = f"""
    Make this explanation more student-friendly:
    
    Question: {question}
    Original Answer: {answer}
    
    Provide:
    1. Simple explanation in bullet points
    2. One real-world example
    3. A memory tip
    
    Keep it concise and engaging for high school students.
    """
    
    try:
        return get_ai_response(enhanced_prompt)
    except Exception:
        return answer

# IMPROVED Quiz Generation - Using best practices for HuggingFace models
def generate_quiz_with_llm(topic: str, num_questions: int = 5) -> list:
    if not topic.strip():
        return []
    
    # Optimized prompt structure for better quiz generation
    quiz_prompt = f"""Generate a multiple choice quiz about {topic}.

Format each question exactly like this:
Q: What is the main concept in {topic}?
A) First option
B) Second option  
C) Third option
D) Fourth option
Answer: A

Q: Which statement about {topic} is correct?
A) First option
B) Second option
C) Third option  
D) Fourth option
Answer: B

Create {num_questions} questions about {topic}:

Q:"""
    
    try:
        pipe = load_hf_pipe()
        if not pipe:
            st.error("AI model not available for quiz generation")
            return []
        
        # Generate with optimized parameters
        result = pipe(
            quiz_prompt, 
            max_new_tokens=1500, 
            do_sample=True, 
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=pipe.tokenizer.eos_token_id if hasattr(pipe.tokenizer, 'eos_token_id') else 50256,
            return_full_text=False
        )
        
        quiz_text = result[0]["generated_text"]
        
        # Parse the generated quiz
        questions = parse_quiz_text_improved(quiz_text, num_questions)
        
        if not questions:
            st.error("Failed to parse quiz questions. Please try again with a different topic.")
            return []
        
        return questions[:num_questions]
        
    except Exception as e:
        st.error(f"Quiz generation failed: {e}")
        return []

def parse_quiz_text_improved(text: str, num_questions: int) -> list:
    """Enhanced quiz parsing with multiple pattern matching strategies"""
    questions = []
    
    # Clean the text
    text = text.strip()
    
    # Try multiple parsing strategies
    
    # Strategy 1: Q: pattern
    q_pattern = r'Q:\s*([^?]+\?)'
    q_matches = re.finditer(q_pattern, text, re.MULTILINE | re.DOTALL)
    
    for match in q_matches:
        question_start = match.start()
        question_text = match.group(1).strip()
        
        # Find the text after this question
        next_match = None
        for other_match in re.finditer(q_pattern, text, re.MULTILINE | re.DOTALL):
            if other_match.start() > question_start:
                next_match = other_match
                break
        
        if next_match:
            question_block = text[question_start:next_match.start()]
        else:
            question_block = text[question_start:]
        
        # Extract options and answer
        options = []
        answer = None
        
        # Find options A), B), C), D)
        option_pattern = r'([A-D]\))\s*([^\n]+)'
        option_matches = re.findall(option_pattern, question_block)
        
        for opt_letter, opt_text in option_matches:
            options.append(f"{opt_letter} {opt_text.strip()}")
        
        # Find answer
        answer_patterns = [
            r'Answer:\s*([A-D])',
            r'Correct.*?:\s*([A-D])',
            r'The answer is\s*([A-D])'
        ]
        
        for pattern in answer_patterns:
            answer_match = re.search(pattern, question_block, re.IGNORECASE)
            if answer_match:
                answer = answer_match.group(1).upper()
                break
        
        # Validate and add question
        if question_text and len(options) >= 4 and answer in ['A', 'B', 'C', 'D']:
            questions.append({
                'question': question_text,
                'options': options[:4],  # Take first 4 options
                'answer': answer
            })
    
    # Strategy 2: If Strategy 1 fails, try numbered questions
    if not questions:
        question_pattern = r'(?:Question\s*\d+|Q\d+):\s*([^?]+\?)'
        question_matches = re.finditer(question_pattern, text, re.MULTILINE | re.DOTALL)
        
        for match in question_matches:
            question_start = match.start()
            question_text = match.group(1).strip()
            
            # Similar logic as Strategy 1
            next_match = None
            for other_match in re.finditer(question_pattern, text, re.MULTILINE | re.DOTALL):
                if other_match.start() > question_start:
                    next_match = other_match
                    break
            
            if next_match:
                question_block = text[question_start:next_match.start()]
            else:
                question_block = text[question_start:]
            
            options = []
            answer = None
            
            option_pattern = r'([A-D]\))\s*([^\n]+)'
            option_matches = re.findall(option_pattern, question_block)
            
            for opt_letter, opt_text in option_matches:
                options.append(f"{opt_letter} {opt_text.strip()}")
            
            answer_patterns = [
                r'Answer:\s*([A-D])',
                r'Correct.*?:\s*([A-D])',
                r'The answer is\s*([A-D])'
            ]
            
            for pattern in answer_patterns:
                answer_match = re.search(pattern, question_block, re.IGNORECASE)
                if answer_match:
                    answer = answer_match.group(1).upper()
                    break
            
            if question_text and len(options) >= 4 and answer in ['A', 'B', 'C', 'D']:
                questions.append({
                    'question': question_text,
                    'options': options[:4],
                    'answer': answer
                })
    
    return questions

# Session State Management
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'voice_input' not in st.session_state:
    st.session_state.voice_input = ""

# Sidebar Configuration
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown(f"**Model**: {MODEL}")
student_mode = st.sidebar.checkbox("🎓 Enhanced Student Mode", help="Detailed explanations with examples")
speak_answers = st.sidebar.checkbox("🔊 Speak Answers", help="Read answers aloud")
voice_mode = st.sidebar.checkbox("🎤 Voice Mode", help="Voice input enabled")
language = st.sidebar.selectbox("🌍 Output Language", ["en", "hi", "fr", "de", "es"])

# PDF Upload
uploaded_file = st.sidebar.file_uploader("📄 Upload PDF", type="pdf")
pdf_context = None
if uploaded_file:
    pdf_context = pdf_to_lines(uploaded_file.read())
    if pdf_context:
        st.sidebar.success(f"PDF loaded: {len(pdf_context)} lines")

# Main Interface
st.title("🤖 AI-Powered Virtual Teaching Assistant")
st.markdown("Ask questions, solve math problems, generate quizzes, and more!")

# Tabs
chat_tab, quiz_tab, dashboard_tab = st.tabs(["💬 Chat", "📝 Quiz", "📊 Dashboard"])

# Chat Tab
with chat_tab:
    # Voice input section
    if voice_mode:
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🎤 Listen"):
                try:
                    recognizer = sr.Recognizer()
                    with sr.Microphone() as source:
                        st.info("🔴 Listening...")
                        recognizer.adjust_for_ambient_noise(source)
                        audio = recognizer.listen(source, timeout=5)
                    
                    st.session_state.voice_input = recognizer.recognize_google(audio)
                    st.success(f"Heard: {st.session_state.voice_input}")
                except sr.UnknownValueError:
                    st.error("Could not understand audio")
                except Exception as e:
                    st.error(f"Voice error: {e}")
    
    # Text input
    user_input = st.text_input(
        "💭 Ask your question:", 
        value=st.session_state.voice_input if voice_mode else "",
        key="user_question"
    )
    
    if st.button("✅ Submit") and user_input:
        with st.spinner("🤔 Thinking..."):
            # Process the question
            english_question = safe_translate(user_input, "en") if detect_language(user_input) != "en" else user_input
            
            # Check if it's a math question
            if MATH_RE.search(english_question):
                answer = solve_math(english_question)
            else:
                # Try to get real-time info first
                answer = get_realtime_info(english_question)
                
                # If no real-time info, use AI model
                if not answer:
                    context = find_context(english_question, pdf_context) if pdf_context else ""
                    prompt = create_prompt(english_question, context)
                    answer = get_ai_response(prompt)
            
            # Enhance for students if enabled
            if student_mode and answer:
                answer = enhance_for_students(english_question, answer)
            
            # Translate answer if needed
            final_answer = safe_translate(answer, language)
            
            # Store in chat history
            st.session_state.chat_history.append({
                'question': user_input,
                'answer': final_answer,
                'timestamp': datetime.datetime.now()
            })
            
            # Speak answer if enabled
            if speak_answers:
                speak_async(final_answer)
        
        # Clear voice input
        st.session_state.voice_input = ""
        st.rerun()
    
    # Display chat history
    st.markdown("---")
    st.subheader("💬 Chat History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history[-10:])):
            with st.container():
                st.markdown(f"**👤 You:** {chat['question']}")
                st.markdown(f"**🤖 Assistant:** {chat['answer']}")
                st.markdown(f"*{chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*")
                st.markdown("---")
    else:
        st.info("No conversations yet. Ask me anything!")

# Quiz Tab
with quiz_tab:
    st.subheader("📝 Quiz Generator")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        quiz_topic = st.text_input("📚 Enter quiz topic:", placeholder="e.g., Python Programming, Calculus, Physics")
    with col2:
        num_questions = st.selectbox("Questions:", [3, 5, 7, 10])
    
    if st.button("🚀 Generate Quiz"):
        if quiz_topic:
            with st.spinner("🔄 Generating quiz..."):
                questions = generate_quiz_with_llm(quiz_topic, num_questions)
                
                if questions:
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_answers = {}
                    st.session_state.current_topic = quiz_topic
                    st.success(f"✅ Generated {len(questions)} questions on {quiz_topic}!")
                    st.rerun()
                else:
                    st.error("Failed to generate quiz. Please try a different topic or try again.")
        else:
            st.warning("Please enter a topic.")
    
    # Display quiz
    if st.session_state.quiz_questions:
        st.markdown("---")
        st.subheader(f"📋 Quiz: {st.session_state.get('current_topic', 'Unknown Topic')}")
        
        # Quiz questions
        for i, q in enumerate(st.session_state.quiz_questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            
            # Radio buttons for options
            answer_key = f"q_{i}"
            selected = st.radio(
                f"Select answer for Q{i+1}:",
                q['options'],
                key=f"quiz_{i}_{len(st.session_state.quiz_questions)}_{hash(str(st.session_state.quiz_questions))}",
                index=None
            )
            
            if selected:
                st.session_state.quiz_answers[answer_key] = selected[0]  # Store A, B, C, or D
            
            st.markdown("")
        
        # Submit quiz
        if st.button("📊 Submit Quiz"):
            if len(st.session_state.quiz_answers) == len(st.session_state.quiz_questions):
                # Calculate score
                score = 0
                for i, q in enumerate(st.session_state.quiz_questions):
                    user_answer = st.session_state.quiz_answers.get(f"q_{i}")
                    correct_answer = q['answer']
                    if user_answer == correct_answer:
                        score += 1
                
                percentage = (score / len(st.session_state.quiz_questions)) * 100
                
                # Display results
                st.success(f"🎉 Quiz Complete! Score: {score}/{len(st.session_state.quiz_questions)} ({percentage:.1f}%)")
                
                # Store in history
                st.session_state.quiz_history.append({
                    'date': datetime.datetime.now(),
                    'topic': st.session_state.get('current_topic', 'Unknown'),
                    'score': score,
                    'total': len(st.session_state.quiz_questions),
                    'percentage': percentage
                })
                
                # Speak result
                if speak_answers:
                    speak_async(f"Quiz completed! You scored {score} out of {len(st.session_state.quiz_questions)}")
                
                # Show detailed results
                with st.expander("📋 Detailed Results"):
                    for i, q in enumerate(st.session_state.quiz_questions):
                        user_answer = st.session_state.quiz_answers.get(f"q_{i}")
                        correct_answer = q['answer']
                        is_correct = user_answer == correct_answer
                        
                        st.markdown(f"**Q{i+1}:** {q['question']}")
                        st.markdown(f"**Your answer:** {user_answer}")
                        st.markdown(f"**Correct answer:** {correct_answer}")
                        st.markdown("✅ Correct!" if is_correct else "❌ Incorrect")
                        st.markdown("---")
                
                # Clear quiz
                st.session_state.quiz_questions = []
                st.session_state.quiz_answers = {}
                
                if percentage == 100:
                    st.balloons()
                    
            else:
                st.warning("Please answer all questions before submitting.")

# Dashboard Tab
with dashboard_tab:
    st.subheader("📈 Performance Dashboard")
    
    if st.session_state.quiz_history:
        # Create DataFrame
        df = pd.DataFrame(st.session_state.quiz_history)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Quizzes", len(df))
        with col2:
            st.metric("Average Score", f"{df['percentage'].mean():.1f}%")
        with col3:
            st.metric("Best Score", f"{df['percentage'].max():.1f}%")
        
        # Recent results table
        st.markdown("---")
        st.subheader("📊 Recent Quiz Results")
        
        display_df = df.copy()
        display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['Score'] = display_df.apply(lambda x: f"{x['score']}/{x['total']}", axis=1)
        display_df['Percentage'] = display_df['percentage'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(
            display_df[['Date', 'topic', 'Score', 'Percentage']].tail(10),
            use_container_width=True
        )
        
        # Performance chart
        if len(df) > 1:
            st.markdown("---")
            st.subheader("📈 Performance Over Time")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['date'], df['percentage'], marker='o', linewidth=2, markersize=8)
            ax.set_ylabel('Score (%)')
            ax.set_title('Quiz Performance Over Time')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Clear history button
        st.markdown("---")
        if st.button("🗑️ Clear History"):
            st.session_state.quiz_history = []
            st.success("History cleared!")
            st.rerun()
    else:
        st.info("📊 No quiz history yet. Take some quizzes to see your performance!")

# Footer
st.markdown("---")
st.markdown("**🤖 AI Teaching Assistant** | Enhanced Quiz Generation | Student-Friendly Learning")