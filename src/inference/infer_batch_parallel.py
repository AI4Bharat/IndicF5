import argparse
import codecs
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
import pandas as pd
from tqdm import tqdm

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
from f5_tts.model import DiT, UNetT


def run_batch_inference(prompt_paths, prompt_texts, texts, languages, categories, model_obj, vocoder, mel_spec_type, remove_silence, speed, output_dir):
    count = 0
    for ref_audio in prompt_paths:
        if not isinstance(ref_audio, str) or not os.path.isfile(ref_audio):
            print(f"Invalid ref_audio: {ref_audio}")
            count += 1
            print(count)
            # raise ValueError(f"Invalid ref_audio: {ref_audio}")

    for idx, (ref_audio, ref_text, text_gen, language, category) in tqdm(enumerate(zip(prompt_paths, prompt_texts, texts, languages, categories))):
        voices = {"main": {"ref_audio": ref_audio, "ref_text": ref_text}}
        for voice in voices:
            voices[voice]["ref_audio"], voices[voice]["ref_text"] = preprocess_ref_audio_text(
                voices[voice]["ref_audio"], voices[voice]["ref_text"]
            )
            print("Voice:", voice)
            print("Ref_audio:", voices[voice]["ref_audio"])
            print("Ref_text:", voices[voice]["ref_text"])

        generated_audio_segments = []
        reg1 = r"(?=\[\w+\])"
        chunks = re.split(reg1, text_gen)
        reg2 = r"\[(\w+)\]"
        for text in chunks:
            if not text.strip():
                continue
            match = re.match(reg2, text)
            if match:
                voice = match[1]
            else:
                print("No voice tag found, using main.")
                voice = "main"
            if voice not in voices:
                print(f"Voice {voice} not found, using main.")
                voice = "main"
            text = re.sub(reg2, "", text)
            gen_text = text.strip()
            ref_audio = voices[voice]["ref_audio"]
            ref_text = voices[voice]["ref_text"]
            print(f"Voice: {voice}")
            audio, final_sample_rate, spectragram = infer_process(
                ref_audio, ref_text, gen_text, model_obj, vocoder, mel_spec_type=mel_spec_type, speed=speed
            )
            generated_audio_segments.append(audio)

        if generated_audio_segments:
            final_wave = np.concatenate(generated_audio_segments)
            filename = f"{language.upper()}_{category.upper()}_{idx}.wav"
            outfile_dir = os.path.join(output_dir, language)
            os.makedirs(outfile_dir, exist_ok=True)
            wave_path = Path(outfile_dir) / filename
            with open(wave_path, "wb") as f:
                sf.write(f.name, final_wave, final_sample_rate)
                if remove_silence:
                    remove_silence_for_generated_wav(f.name)
                print(f"Generated audio saved to: {f.name}")


def main():
    parser = argparse.ArgumentParser(
        prog="python3 infer-cli.py",
        description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
        epilog="Specify options above to override one or more settings from config.",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="F5-TTS | E2-TTS",
    )
    parser.add_argument(
        "-p",
        "--ckpt_file",
        help="The Checkpoint .pt",
    )
    parser.add_argument(
        "-v",
        "--vocab_file",
        help="The vocab .txt",
    )

    parser.add_argument(
        "-f",
        "--generate_csv",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to output folder..",
    )
    parser.add_argument(
        "--remove_silence",
        help="Remove silence.",
    )
    parser.add_argument("--vocoder_name", type=str, default="vocos", choices=["vocos", "bigvgan"], help="vocoder name")
    parser.add_argument(
        "--load_vocoder_from_local",
        action="store_true",
        help="load vocoder from local. Default: ../checkpoints/charactr/vocos-mel-24khz",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Adjust the speed of the audio generation (default: 1.0)",
    )
    args = parser.parse_args()

    # Read texts and prompts to generate
    filepath = args.generate_csv
    df = pd.read_csv(filepath)
    prompt_paths = df['prompt_path'].tolist()
    prompt_texts = df['prompt_text'].tolist()
    texts = df['text'].tolist()
    languages = df['language'].tolist()
    categories = df['category'].tolist()

    # Model config
    model = args.model
    ckpt_file = args.ckpt_file
    vocab_file = args.vocab_file
    remove_silence = args.remove_silence
    speed = args.speed
    vocoder_name = args.vocoder_name
    mel_spec_type = args.vocoder_name
    if vocoder_name == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif vocoder_name == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=args.load_vocoder_from_local, local_path=vocoder_local_path)

    # load models
    model_cls = DiT
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    print(f"Using {model}...")
    ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type=mel_spec_type, vocab_file=vocab_file)

    # Batch inference
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    run_batch_inference(prompt_paths, prompt_texts, texts, languages, categories, ema_model, vocoder, mel_spec_type, remove_silence, speed, output_dir)


if __name__ == "__main__":
    main()
