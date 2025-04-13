import os
import uuid
import time
import numpy as np
import soundfile as sf
import google.generativeai as genai
import gradio as gr

# Configure Gemini API
genai.configure(api_key="AIzaSyBas_7s1hD9cfAJuRHn-K4vrYZbqE-eXEE")
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

def upload_audio(audio):
    sample_rate, data = audio
    guid = str(uuid.uuid4())
    filename = f"media/{guid}.wav"
    
    if data.ndim == 2:  # stereo
        data = data.T
    elif data.ndim != 1:
        return "Unsupported audio format."
    
    sf.write(filename, data, sample_rate)
    ref = genai.upload_file(path=filename)
    return ref

def generate_prompt(language, word_phrase):
    return PROMPT_TEMPLATE.format(language=language, word_phrase=word_phrase)

def evaluate_audio(audio, language, phrase, model_choice):
    start = time.time()
    audio_file_id = upload_audio(audio)
    prompt = generate_prompt(language, phrase)
    
    model = genai.GenerativeModel(model_choice)
    response = model.generate_content([prompt, audio_file_id])
    
    elapsed = round(time.time() - start, 2)
    return (
        response.text,
        f"{elapsed} seconds",
        response.usage_metadata.prompt_token_count,
        response.usage_metadata.total_token_count,
        model_choice
    )

with gr.Blocks() as demo:
    with gr.Tab("Multilingual Pronunciation Evaluation"):
        audio_input = gr.Audio(sources=["microphone", "upload"], type="numpy", label="🎙️ Upload or record audio")
        language = gr.Textbox(label="🌍 Language", placeholder="e.g., Arabic, Spanish, Mandarin")
        phrase = gr.Textbox(label="🗣️ Phrase to compare with the audio")
        model_choice = gr.Radio(
            ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.0-flash-lite-preview-02-05"],
            label="🤖 Choose Gemini Model"
        )
        btn = gr.Button("🔍 Analyze")
        result = gr.Textbox(label="📋 Feedback")
        time_taken = gr.Textbox(label="⏱️ Time")
        tokens_used = gr.Textbox(label="📊 Input Tokens")
        total_tokens = gr.Textbox(label="📈 Total Tokens")
        selected_model = gr.Textbox(label="🤖 Model Used")

        btn.click(
            fn=evaluate_audio,
            inputs=[audio_input, language, phrase, model_choice],
            outputs=[result, time_taken, tokens_used, total_tokens, selected_model]
        )

if __name__ == "__main__":
    demo.launch()
