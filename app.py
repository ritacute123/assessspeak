import os
import uuid
import numpy as np
import soundfile as sf
import streamlit as st
import google.generativeai as genai
import tempfile
from streamlit.components.v1 import html

# CONFIGURE GEMINI API
genai.configure(api_key="AIzaSyBas_7s1hD9cfAJuRHn-K4vrYZbqE-eXEE")

# AUDIO FOLDER
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

def generate_prompt(language, word_phrase):
    return PROMPT_TEMPLATE.format(language=language, word_phrase=word_phrase)

def evaluate(filename, prompt, model_choice):
    file_ref = genai.upload_file(path=filename)
    model = genai.GenerativeModel(model_choice)
    full_prompt = [prompt, file_ref]
    response = model.generate_content(full_prompt)
    return response.text, response.usage_metadata

def recorder_ui():
    st.markdown("### 🎤 Or record your voice below")
    html("""
    <script>
    let mediaRecorder;
    let audioChunks = [];

    async function startRecording() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const blob = new Blob(audioChunks);
            const arrayBuffer = await blob.arrayBuffer();
            const base64Audio = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
            const form = document.createElement("form");
            form.method = "post";
            form.action = "/_stcore/upload_file/";
            form.enctype = "multipart/form-data";
            const input = document.createElement("input");
            input.name = "file";
            input.type = "hidden";
            input.value = base64Audio;
            form.appendChild(input);
            document.body.appendChild(form);
            form.submit();
        };

        mediaRecorder.start();
    }

    function stopRecording() {
        mediaRecorder.stop();
    }
    </script>
    <button onclick="startRecording()">Click to Record</button>
    <button onclick="stopRecording()">Stop</button>
    """, height=100)

# App UI
st.set_page_config(page_title="Multilingual Speaking Evaluation", layout="centered")
st.title("🗣️ Multilingual Speaking Evaluation")
st.markdown("Analyze your pronunciation in any language using Google Gemini.")

uploaded_audio = st.file_uploader("📂 Upload your audio file (.wav only)", type=["wav"])

recorder_ui()

language = st.text_input("🌍 Language (e.g., Arabic, Spanish, Mandarin)")
word_phrase = st.text_input("🗣️ Phrase to compare with the audio")
model_choice = st.selectbox("🤖 Select Gemini Model", [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite-preview-02-05"
])

audio_source = None
if uploaded_audio:
    guid = str(uuid.uuid4())
    audio_source = f"media/{guid}.wav"
    audio_data, sample_rate = sf.read(uploaded_audio)
    sf.write(audio_source, audio_data, sample_rate)

if st.button("🔍 Analyze") and audio_source and language and word_phrase:
    with st.spinner("Analyzing pronunciation..."):
        prompt = generate_prompt(language, word_phrase)
        try:
            result, usage = evaluate(audio_source, prompt, model_choice)
            st.success("✅ Evaluation Complete")
            st.markdown("### 📋 Feedback")
            st.text_area("AI Feedback", result, height=400)
            st.markdown("### 📊 Token Usage")
            st.write(f"**Input Tokens:** {usage.prompt_token_count}")
            st.write(f"**Total Tokens:** {usage.total_token_count}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
