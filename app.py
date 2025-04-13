import google.generativeai as genai
import gradio as gr
import numpy as np
import soundfile as sf
import time
import uuid

# ✅ API key inserted directly (only do this in dev or trusted environments)
genai.configure(api_key="AIzaSyBas_7s1hD9cfAJuRHn-K4vrYZbqE-eXEE")

PROMPT_TEMPLATE = """
You are a native speaker and expert linguist of the {language} language, specializing in pronunciation coaching. Your task is to analyze an audio recording of spoken {language}, compare it with the reference phrase, and provide a detailed pronunciation assessment.

Input:
1. An audio file of spoken {language}.
2. A word, phrase, or sentence to compare with the audio.

Your task:
- Detect the phrase in the audio.
- Compare pronunciation to the reference.
- Identify errors in vowel sounds, consonant articulation, stress, intonation, linking, and missing words.
- Provide recommendations for improvement.
- Rate the overall pronunciation on a scale from 0% to 100%.

If the audio does not contain the input phrase, say: "The audio does not contain the phrase."

Your Output Format:
Phrase (Input): {word_phrase}
Phrase (Detected): [Detected phrase from audio]

Comparison:
[Similarities/differences]

Problem Areas:
[List and describe pronunciation issues]

Recommendations for Improvement:
[Personalized guidance per issue]

Overall Pronunciation Rating:
[XX]%
"""

def upload_audio(audio):
    sample_rate, data = audio
    data = np.array(data)
    guid_string = str(uuid.uuid4())
    filename = f"media/{guid_string}.wav"

    if data.ndim == 2:
        data = data.T
    elif data.ndim != 1:
        return "Unexpected audio data format"

    sf.write(filename, data, sample_rate)
    ref = genai.upload_file(path=filename)
    return ref


def create_prompt(language, word_phrase):
    return PROMPT_TEMPLATE.format(language=language, word_phrase=word_phrase)


def evaluate_audio_pronunciation(audio_file_id, prompt, model="gemini-2.0-flash"):
    prompt = [prompt, audio_file_id]
    model = genai.GenerativeModel(model)
    response = model.generate_content(contents=prompt)
    total_token_count = response.usage_metadata.total_token_count
    return response.text, response.usage_metadata.prompt_token_count, total_token_count


def orchestrate(audio, language, word_phrase, model):
    start_time = time.time()
    audio_file_id = upload_audio(audio)
    prompt = create_prompt(language, word_phrase)
    response, input_tokens, total_tokens = evaluate_audio_pronunciation(
        audio_file_id, prompt, model
    )
    end_time = time.time()
    return response, f"{end_time - start_time:.2f} seconds", input_tokens, total_tokens, model


ui_blocks = gr.Blocks()

input_audio = gr.Audio(
    sources=["microphone", "upload"],
    waveform_options=gr.WaveformOptions(
        waveform_color="#01C6FF",
        waveform_progress_color="#0066B4",
        skip_length=2,
        show_controls=False,
    ),
)

get_prompt_ui_block = gr.Interface(
    fn=orchestrate,
    inputs=[
        input_audio,
        gr.Textbox(label="Language (e.g., Arabic, Spanish, French, Japanese)", lines=1),
        gr.Textbox(label="Word or Phrase to Compare", lines=1),
        gr.Radio(
            ["gemini-1.5-flash-8b", "gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash"],
            info="Choose Gemini Model",
        ),
    ],
    outputs=[
        gr.Textbox(label="Response"),
        gr.Textbox(label="Evaluation Time"),
        gr.Textbox(label="Input Tokens"),
        gr.Textbox(label="Total Tokens"),
        gr.Textbox(label="Model Used"),
    ],
    allow_flagging="never"
)

with ui_blocks:
    gr.TabbedInterface(
        [get_prompt_ui_block],
        ["Multilingual Pronunciation Evaluation"]
    )

if __name__ == "__main__":
    ui_blocks.launch()
