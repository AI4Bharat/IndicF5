# **IndicF5: High-Quality Text-to-Speech for Indian Languages**

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/ai4bharat/IndicF5)


We release **IndicF5**, a **near-human polyglot** **Text-to-Speech (TTS)** model trained on **1417 hours** of high-quality speech from **[Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), [IndicTTS](https://www.iitm.ac.in/donlab/indictts/database), [LIMMITS](https://sites.google.com/view/limmits24/), and [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)**.  

IndicF5 supports **11 Indian languages**:  
**Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  

---

## 🚀 Installation
```bash
conda create -n indicf5 python=3.10 -y
conda activate indicf5
pip install git+https://github.com/ai4bharat/IndicF5.git
```


## 🎙 Usage

To generate speech, you need to provide **three inputs**:
1. **Text to synthesize** – The content you want the model to speak.
2. **A reference prompt audio** – An example speech clip that guides the model’s prosody and speaker characteristics.
3. **Text spoken in the reference prompt audio** – The transcript of the reference prompt audio.


```python
import os
import torch
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)

# Device selection
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Download model files from HuggingFace
repo_id = "ai4bharat/IndicF5"
vocab_path = hf_hub_download(repo_id, filename="checkpoints/vocab.txt")
ckpt_path = hf_hub_download(repo_id, filename="model.safetensors")

# Load vocoder
print("Loading vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

# Load model
print("Loading model...")
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    mel_spec_type="vocos",
    vocab_file=vocab_path,
    device=device,
)

# Load checkpoint weights (filter to ema_model keys and strip prefix)
state_dict = load_file(ckpt_path, device=device)
state_dict = {
    k.replace("ema_model._orig_mod.", ""): v
    for k, v in state_dict.items()
    if k.startswith("ema_model.")
}
model.load_state_dict(state_dict)
model.eval()

# Prepare inputs
ref_audio_path = "prompts/PAN_F_HAPPY_00001.wav"
ref_text = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
gen_text = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए."

# Preprocess reference audio
print("Preprocessing reference audio...")
ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text, device=device)

# Generate speech
print("Generating speech...")
audio, sample_rate, _ = infer_process(
    ref_audio,
    ref_text,
    gen_text,
    model,
    vocoder,
    mel_spec_type="vocos",
    device=device,
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
os.makedirs("samples", exist_ok=True)
sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
print("Audio saved to samples/namaste.wav")
```

## References

We would like to extend our gratitude to the authors of  **[F5-TTS](https://github.com/SWivid/F5-TTS)** for their invaluable contributions and inspiration to this work. Their efforts have played a crucial role in advancing  the field of text-to-speech synthesis.


## 📖 Citation
If you use **IndicF5** in your research or projects, please consider citing it:

### 🔹 BibTeX
```bibtex
@misc{AI4Bharat_IndicF5_2025,
  author       = {Praveen S V and Srija Anand and Soma Siddhartha and Mitesh M. Khapra},
  title        = {IndicF5: High-Quality Text-to-Speech for Indian Languages},
  year         = {2025},
  url          = {https://github.com/AI4Bharat/IndicF5},
}

