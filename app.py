import streamlit as st
import speech_recognition as sr
import spacy
import pandas as pd
from collections import Counter
from audiorecorder import audiorecorder
import io
from pydub import AudioSegment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from heapq import nlargest
from transformers import pipeline

st.set_page_config(layout="wide", page_title="NLP Analysis Dashboard")

def local_css():
    """Injects custom CSS for a more beautiful UI."""
    css = """
    <style>
        /* Base */
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }

        /* Title */
        .stTitle {
            font-weight: 700;
            color: #79AFFF; /* A nice blue */
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #FAFAFA;
        }

        /* Radio Buttons (Mode Selector) */
        div[data-testid="stRadio"] > label {
            background-color: #262730;
            padding: 10px 14px;
            border-radius: 8px;
            margin: 0 4px;
            transition: all 0.2s ease-in-out;
        }
        /* Selected Radio Button */
        div[data-testid="stRadio"] > label[data-baseweb="radio"] > div[data-checked="true"] {
            color: #79AFFF;
        }
        div[data-testid="stRadio"] label:hover {
            background-color: #31333F;
        }

        /* Buttons */
        div[data-testid="stButton"] > button {
            background-color: #79AFFF;
            color: #0E1117;
            border-radius: 8px;
            font-weight: 600;
            padding: 10px 20px;
            border: none;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 14px 0 rgba(0, 118, 255, 0.3);
            width: 100%; /* Make buttons fill their columns */
        }
        div[data-testid="stButton"] > button:hover {
            background-color: #A3C6FF;
            box-shadow: 0 6px 20px 0 rgba(0, 118, 255, 0.35);
        }
        div[data-testid="stButton"] > button:focus:not(:active) {
            border: 2px solid #A3C6FF;
            box-shadow: 0 0 0 2px #A3C6FF;
        }

        /* Stop Recording Button (special case) */
        div[data-testid="stButton"] > button:contains("Stop Recording") {
            background-color: #FF6B6B;
            box-shadow: 0 4px 14px 0 rgba(255, 75, 75, 0.3);
        }
        div[data-testid="stButton"] > button:contains("Stop Recording"):hover {
            background-color: #FF8787;
            box-shadow: 0 6px 20px 0 rgba(255, 75, 75, 0.35);
        }
        
        /* Different style for the new AI button */
        div[data-testid="stButton"] > button:contains("Abstractive") {
            background-color: #0E1117;
            color: #79AFFF;
            border: 1px solid #79AFFF;
        }
        div[data-testid="stButton"] > button:contains("Abstractive"):hover {
            background-color: #262730;
            border: 1px solid #A3C6FF;
            color: #A3C6FF;
        }

        /* Text Area */
        div[data-testid="stTextArea"] textarea {
            background-color: #262730;
            color: #FAFAFA;
            border-radius: 8px;
            border: 1px solid #4A4C5A;
        }

        /* Tabs */
        div[data-testid="stTabs"] button[role="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            font-weight: 600;
            background-color: transparent;
            color: #A0A3B1;
            border: 1px solid transparent;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background-color: #262730;
            color: #79AFFF;
            border: 1px solid #4A4C5A;
            border-bottom: none;
        }
        div[data-testid="stTabs"] div[role="tabpanel"] {
            background-color: #262730;
            border: 1px solid #4A4C5A;
            border-top: none;
            border-radius: 0 0 8px 8px;
            padding: 24px;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background-color: #1E1E1E;
            border: 1px solid #4A4C5A;
            border-radius: 8px;
            padding: 16px;
        }
        div[data-testid="stMetric"] > div[data-testid="stMetricValue"] {
            color: #79AFFF;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

local_css()

@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("Could not load spaCy model 'en_core_web_sm'. Please run 'python -m spacy download en_core_web_sm'.")
        return None

@st.cache_resource
def load_summarizer_model():
    try:
        model_name = "sshleifer/distilbart-cnn-6-6"
        st.info(f"Loading Abstractive Summarizer model ('{model_name}')... This may take a few minutes on the first run.")
        summarizer = pipeline("summarization", model=model_name)
        st.success("Abstractive Summarizer model loaded.")
        return summarizer
    except Exception as e:
        st.error(f"Could not load summarization model: {e}")
        return None

nlp = load_spacy_model()
abstractive_summarizer = load_summarizer_model()

def analyze_text(text, nlp_model):
    if not nlp_model:
        st.error("NLP model is not loaded. Cannot perform analysis.")
        return

    doc = nlp_model(text)
    tab1, tab2, tab3 = st.tabs([
        "Sentiment & Topics", 
        "Keywords & Stats", 
        "Named Entities (NER)"
    ])

    with tab1:
        st.subheader("Sentiment Analysis (VADER)")
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(text)
        compound = sentiment_scores['compound']
        
        if compound >= 0.05:
            label, emoji = "Positive", "😊"
        elif compound <= -0.05:
            label, emoji = "Negative", "😠"
        else:
            label, emoji = "Neutral", "😐"
            
        st.metric(label="VADER Sentiment", value=f"{label} {emoji}", delta=f"Compound Score: {compound:.3f}")
        
        st.progress((compound + 1) / 2)
        
        with st.expander("See full VADER Score Breakdown"):
            st.json({
                "Positive": sentiment_scores['pos'],
                "Neutral": sentiment_scores['neu'],
                "Negative": sentiment_scores['neg'],
                "Compound Score": sentiment_scores['compound']
            })

        st.divider()
        
        st.subheader("Content Classifier")
        
        TOPIC_KEYWORDS = {
            'Action Item': {
                'action', 'task', 'follow-up', 'urgent', 'asap', 'immediately', 
                'todo', 'deadline', 'complete', 'assign', 'assigned', 'send', 
                'email', 'schedule', 'meeting', 'call', 'report', 'due'
            },
            'Movie/Show Review': {
                'movie', 'film', 'actor', 'actress', 'plot', 'ending', 'scene',
                'series', 'episode', 'directed', 'screenplay', 'netflix', 'hulu'
            },
            'Product Review': {
                'product', 'bought', 'buy', 'review', 'stars', 'rating', 
                'customer service', 'return', 'refund', 'shipping', 'quality'
            },
            'Academic Topic': {
                'math', 'calculus', 'algebra', 'geometry', 'biology', 'physics', 
                'chemistry', 'science', 'history', 'literature', 'econ', 'economics'
            },
            'Tech Topic': {
                'ml', 'os', 'ai', 'machine learning', 'operating system', 'artificial intelligence',
                'database', 'algorithm', 'python', 'javascript', 'code', 'programming', 'dev'
            }
        }
        
        lemmas = {token.lemma_.lower() for token in doc}
        found_topics = []
        trigger_words = []

        for topic, keywords in TOPIC_KEYWORDS.items():
            matches = lemmas.intersection(keywords)
            if matches:
                found_topics.append(topic)
                trigger_words.extend(list(matches))

        if found_topics:
            st.metric(label="Detected Topics", value=", ".join(found_topics))
            st.info(f"Trigger words found: {', '.join(list(set(trigger_words)))}")
        else:
            st.metric(label="Detected Topic", value="General Conversation", delta="No specific topics found")

    with tab2:
        st.subheader("Top Keywords (Bag-of-Words)")
        filtered_tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_stop and not token.is_punct and not token.is_space
        ]
        
        if filtered_tokens:
            keyword_freq = Counter(filtered_tokens)
            keywords_df = pd.DataFrame(
                keyword_freq.most_common(10), 
                columns=["Keyword", "Frequency"]
            )
            st.dataframe(keywords_df, use_container_width=True)
        else:
            st.info("No significant keywords found after filtering stop-words.")
        
        st.divider()

        st.subheader("Preprocessing Stats")
        tokens = [token.text for token in doc]
        
        col1, col2 = st.columns(2)
        col1.metric("Total Tokens (Words)", len(tokens))
        col2.metric("Tokens (Stop-words/Punctuation removed)", len(filtered_tokens))

    with tab3:
        st.subheader("Named Entities (NER)")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if entities:
            entities_df = pd.DataFrame(entities, columns=["Entity", "Type (Label)"])
            st.dataframe(entities_df, use_container_width=True)
            st.info("Visual representation of entities:")
            html = spacy.displacy.render(doc, style="ent")
            st.write(html, unsafe_allow_html=True)
            
        else:
            st.info("No named entities were found in the text.")


def summarize_extractive(text, nlp_model, num_sentences=3):
    """
    Performs extractive summarization on the text.
    Finds the most important sentences based on word frequency.
    """
    if not nlp_model:
        st.error("NLP model is not loaded. Cannot perform analysis.")
        return "Error: NLP model not loaded."

    doc = nlp_model(text)
    keywords = [
        token.lemma_.lower() for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    
    if not keywords:
        return "Could not generate a summary (e.g., text is too short or has no keywords)."
        
    word_freq = Counter(keywords)
    
    sentence_scores = {}
    for sent in doc.sents:
        for word in sent:
            word_lemma = word.lemma_.lower()
            if word_lemma in word_freq:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_freq[word_lemma]
                else:
                    sentence_scores[sent] += word_freq[word_lemma]
                    
    if not sentence_scores:
        return "Could not generate a summary (e.g., text is too short or has no keywords)."

    num_sentences = min(num_sentences, len(sentence_scores))
    if num_sentences == 0:
        return "Text is too short to summarize."

    summarized_sents = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    
    final_summary = ' '.join([sent.text for sent in summarized_sents])
    
    return final_summary

def summarize_abstractive(text, summarizer, min_len=30, max_len=150):
    if not summarizer:
        st.error("Abstractive summarizer model is not loaded.")
        return "Error: Model not loaded."
        
    if len(text.split()) < min_len:
        return "Text is too short for an abstractive summary. Please provide more content."
    
    try:
        summary_list = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary_list[0]['summary_text']
    except Exception as e:
        st.error(f"Error during abstractive summarization: {e}")
        return "Error: Could not generate summary."

st.title("SpeechSensAI")

mode = st.radio(
    "Choose your mode:",
    ("Record Audio & Analyze", "Analyze & Summarize Text"),
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

if mode == "Record Audio & Analyze":
    st.header("1. Record Audio & Analyze")
    st.write("Click the button to record your voice. Click again to stop.")

    audio = audiorecorder("Start Recording", "Stop Recording")

    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    if audio:
        audio_buffer = io.BytesIO()
        audio.export(audio_buffer, format="wav")
        audio_buffer.seek(0)
        r = sr.Recognizer()
        
        try:
            with sr.AudioFile(audio_buffer) as source:
                audio_data = r.record(source)
            
            with st.spinner("Transcribing audio..."):
                text = r.recognize_google(audio_data)
            
            st.success("Transcription complete!")
            st.session_state.transcribed_text = text 
                    
        except sr.UnknownValueError:
            st.error("Speech Recognition could not understand the audio. Please try again.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred during audio processing: {e}")
    transcribed_text_input = st.text_area(
        "Transcribed Text:", 
        st.session_state.transcribed_text, 
        height=150
    )

    st.divider()
    st.header("2. NLP Analysis Dashboard")
    st.write("Analyze or summarize the transcribed text.")
    
    if st.button("Analyze Transcribed Text"):
        if transcribed_text_input and nlp:
            with st.spinner("Running NLP analysis..."):
                analyze_text(transcribed_text_input, nlp)
        elif not nlp:
                st.error("Cannot analyze: spaCy model not loaded.")
        else:
            st.warning("Please transcribe some audio first.")
    
    st.subheader("Summarization")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate Extractive Summary (Fast)"):
            if transcribed_text_input and nlp:
                with st.spinner("Generating extractive summary..."):
                    summary = summarize_extractive(transcribed_text_input, nlp)
                    st.success("Extractive Summary (spaCy)")
                    st.write(summary)
            elif not nlp:
                st.error("Cannot summarize: spaCy model not loaded.")
            else:
                st.warning("Please transcribe some audio first.")
    
    with col2:
        if st.button("Generate Abstractive Summary (AI)"):
            if transcribed_text_input and abstractive_summarizer:
                with st.spinner("Generating abstractive summary... (This may take a moment)"):
                    summary = summarize_abstractive(transcribed_text_input, abstractive_summarizer)
                    st.success("Abstractive Summary (Transformers)")
                    st.write(summary)
            elif not abstractive_summarizer:
                st.error("Cannot summarize: Abstractive model not loaded.")
            else:
                st.warning("Please transcribe some audio first.")


elif mode == "Analyze & Summarize Text":
    st.header("Analyze & Summarize Text")
    st.write("Type or paste any text below to analyze or summarize it.")

    manual_text_input = st.text_area("Text Input:", height=200, key="manual_text")

    if st.button("Analyze Manual Text"):
        if manual_text_input and nlp:
            with st.spinner("Running NLP analysis..."):
                analyze_text(manual_text_input, nlp)
        elif not nlp:
            st.error("Cannot analyze: spaCy model not loaded.")
        else:
            st.warning("Please enter some text to analyze.")

    st.subheader("Summarization")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate Extractive Summary (Fast)"):
            if manual_text_input and nlp:
                with st.spinner("Generating extractive summary..."):
                    summary = summarize_extractive(manual_text_input, nlp)
                    st.success("Extractive Summary (spaCy)")
                    st.write(summary)
            elif not nlp:
                st.error("Cannot summarize: spaCy model not loaded.")
            else:
                st.warning("Please enter some text to summarize.")

    with col2:
        if st.button("Generate Abstractive Summary (AI)"):
            if manual_text_input and abstractive_summarizer:
                with st.spinner("Generating abstractive summary... (This may take a moment)"):
                    summary = summarize_abstractive(manual_text_input, abstractive_summarizer)
                    st.success("Abstractive Summary (Transformers)")
                    st.write(summary)
            elif not abstractive_summarizer:
                st.error("Cannot summarize: Abstractive model not loaded.")
            else:

                st.warning("Please enter some text to summarize.")
