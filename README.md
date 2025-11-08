# SpeechSense AI: An NLP Audio and Text Analysis Dashboard

**SpeechSense AI** is an interactive web application built with **Streamlit** for advanced **Natural Language Processing (NLP)**.  
It provides a comprehensive dashboard to transcribe, analyze, and summarize both live audio and text input.

This tool is designed to showcase and compare **classic NLP algorithms** (like extractive summarization) with **modern, transformer-based AI models** (like abstractive summarization) in a single, user-friendly interface.

---

## Core Features

The application operates in two distinct modes:
- **Record Audio & Analyze**
- **Analyze & Summarize Text**

### 1. Speech-to-Text

**Live Audio Transcription:**  
Utilizes the `audiorecorder` and `SpeechRecognition` libraries to capture and transcribe microphone input directly within the app.

---

### 2. Full NLP Analysis Dashboard

Once text is provided (either from transcription or manual input), the dashboard performs a deep analysis using **spaCy** and **VADER (vaderSentiment)**:

- **Sentiment Analysis:**  
  Classifies text as Positive, Negative, or Neutral using VADER, complete with a compound sentiment score.

- **Content Classifier:**  
  Automatically tags text with detected topics such as "Action Item," "Academic Topic," "Tech Topic," or "Product Review."

- **Named Entity Recognition (NER):**  
  Identifies and extracts entities like people (PER), organizations (ORG), and locations (GPE), including a color-coded visualization using spaCy’s **displacy**.

- **Preprocessing Stats:**  
  Displays token counts before and after the removal of stop words and punctuation.

- **Top Keywords:**  
  Implements a Bag-of-Words model to extract the most frequent and significant keywords.

---

### 3. Dual Summarization Methods

A key feature of **SpeechSense AI** is the comparison between two summarization techniques:

- **Extractive Summarization (Fast):**  
  A classic NLP algorithm (using spaCy) that identifies and extracts the most important sentences from the original text.

- **Abstractive Summarization (AI-based):**  
  A modern transformer-based model (using Hugging Face’s `transformers` and PyTorch) that generates a human-like summary of the text.

---

## Technologies Used

| Category | Technologies |
|-----------|---------------|
| **Core Framework** | Streamlit |
| **Speech Processing** | SpeechRecognition, audiorecorder, pydub |
| **NLP & Analysis** | spaCy, VADER (vaderSentiment), pandas |
| **AI & Deep Learning** | Transformers (Hugging Face), PyTorch |

---

## Setup & Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/SpeechSense-AI.git
cd SpeechSense-AI

