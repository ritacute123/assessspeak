import os
import uuid
import tempfile
import streamlit as st
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import google.generativeai as genai

# GEMINI API KEY
genai.configure(api_key="AIzaSyBas_7s1hD9cfAJuRHn-K4vrYZbqE-eXEE")

# Ensure media directory exists
os.makedirs("media", exist_ok=True)

PROMPT_TEMPLATE = """
You are a native speaker and expert linguist of the {language} language, specializing in pronunciation coaching.

Input:
1. An audio file of spoken {language}.
2. A phrase to compare with the audio.

Your task:
- Detect the phrase in the audio.
- Compare pronunciation to the reference.
- Identify errors in vowels, consonants, stress, intonation, linking.
- Recommend improvements.
- Rate overall pronunciation (0%–100%).

If audio doesn’t contain the input phrase, say: "The audio does not contain the phrase."

Output Format:
Phrase (Input): {word_phrase}
Phrase (Detected): [Detected phrase]

Comparison:
[Summary]

Problem Areas:
[Issues]

Recommendations:
[Tips]

Overall Pronunciation Rating:
[XX]%
"""

def convert_to_wav(file) -> str:
    """Converts mp3/m4a/wav to wav and returns the new file path"""
    audio = AudioSegment.from_file(file)
    guid = str(uuid.uuid4())
    new_path = f"media/{guid}.wav"
    audio.export(new_path, format="wav")
    return new_path

def generate_prompt(language, word_phrase):
    return PROMPT_TEMPLATE.format(language=language, word_phrase=word_phrase)

def evaluate(file_path, prompt, model_choice):
    file_ref = genai.upload_file(path=file_path)
    model = genai.GenerativeModel(model_choice)
    full_prompt = [prompt, file_ref]
    response = model.generate_content(full_prompt)
    return response.text, response.usage_metadata

# Streamlit UI
st.set_page_config(page_title="Multilingual Speaking Evaluation", layout="centered")
st.title("🗣️ Multilingual Speaking Evaluation")
st.markdown("Analyze your pronunciation in any language using Google Gemini.")

# Upload audio in any format
uploaded_audio = st.file_uploader(
    "📂 Upload your audio file (wav, mp3, m4a)", 
    type=["wav", "mp3", "m4a"]
)

language = st.text_input("🌍 Language (e.g., Arabic, Spanish, Mandarin)")
word_phrase = st.text_input("🗣️ Phrase to compare with the audio")
model_choice = st.selectbox("🤖 Select Gemini Model", [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite-preview-02-05"
])

# Main Analyze Button
if st.button("🔍 Analyze") and uploaded_audio and language and word_phrase:
    with st.spinner("Converting and analyzing your audio..."):
        try:
            converted_path = convert_to_wav(uploaded_audio)
            prompt = generate_prompt(language, word_phrase)
            result, usage = evaluate(converted_path, prompt, model_choice)
            st.success("✅ Evaluation Complete")
            st.markdown("### 📋 Feedback")
            st.text_area("AI Feedback", result, height=400)
            st.markdown("### 📊 Token Usage")
            st.write(f"**Input Tokens:** {usage.prompt_token_count}")
            st.write(f"**Total Tokens:** {usage.total_token_count}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
