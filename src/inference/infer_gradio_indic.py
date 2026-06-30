# ruff: noqa: E402
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files
import os

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch

try:
    import spaces
    USING_SPACES = True
except ImportError:
    USING_SPACES = False

def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func

from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)

DEFAULT_TTS_MODEL = "IndicF5"
tts_model_choice = DEFAULT_TTS_MODEL

# load models
vocoder = load_vocoder()

def load_indicf5():
    # Using transformers to load the model as suggested in README
    repo_id = "ai4bharat/IndicF5"
    print(f"Loading IndicF5 from {repo_id}...")
    # This will use the cached model if already downloaded
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model

indic_model = load_indicf5()

@gpu_decorator
def infer_indic(
    ref_audio_orig, ref_text, gen_text, remove_silence, speed=1, show_info=gr.Info
):
    show_info("Starting synthesis...")
    
    try:
        audio = indic_model(
            gen_text,
            ref_audio_path=ref_audio_orig,
            ref_text=ref_text
        )
        
        # Convert to numpy and handle sample rate (IndicF5 usually 24k)
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
            
        final_sample_rate = 24000
        final_wave = audio
        
        # Remove silence if requested
        if remove_silence:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                remove_silence_for_generated_wav(f.name)
                final_wave, _ = torchaudio.load(f.name)
            final_wave = final_wave.squeeze().cpu().numpy()

        return (final_sample_rate, final_wave)
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise gr.Error(f"Inference failed: {e}")

with gr.Blocks() as app_tts:
    gr.Markdown("# IndicF5: Polyglot TTS for Indian Languages")
    gr.Markdown("Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu")
    
    with gr.Row():
        with gr.Column():
            ref_audio_input = gr.Audio(label="Reference Audio (Voice to mimic)", type="filepath")
            ref_text_input = gr.Textbox(
                label="Reference Text",
                info="The exact text spoken in the reference audio.",
                lines=2,
            )
        with gr.Column():
            gen_text_input = gr.Textbox(label="Text to Generate", lines=10, placeholder="Enter text in any of the 11 supported Indian languages...")
    
    generate_btn = gr.Button("Synthesize Speech", variant="primary")
    
    with gr.Accordion("Advanced Settings", open=False):
        remove_silence_check = gr.Checkbox(
            label="Remove Silences",
            info="Experimental feature to trim pauses.",
            value=False,
        )

    audio_output = gr.Audio(label="Synthesized Audio")

    def basic_tts(
        ref_audio_input,
        ref_text_input,
        gen_text_input,
        remove_silence,
    ):
        if not ref_audio_input or not ref_text_input or not gen_text_input:
            raise gr.Error("Please provide all inputs: Reference Audio, Reference Text, and Text to Generate.")
            
        audio_out = infer_indic(
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
        )
        return audio_out

    generate_btn.click(
        basic_tts,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence_check,
        ],
        outputs=[audio_output],
    )

with gr.Blocks() as app:
    app_tts.render()

if __name__ == "__main__":
    print("Starting IndicF5 UI...")
    app.launch()
