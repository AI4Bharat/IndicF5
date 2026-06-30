"""Inference wrapper for IndicF5 model with proper error handling"""

import os
import tempfile
import time
import gc
import contextlib
from typing import Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

from src.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IndicF5InferenceEngine:
    """Wrapper for IndicF5 model inference with proper error handling"""
    
    def __init__(self):
        self.config = get_config()
        self.device = torch.device(self.config.get_device())
        self.model = None
        self._compiled = False
        self._cpu_optimized = False
        self._warmed_up = False
        self._ref_preprocess_cache = {}
        self._max_ref_cache_size = 32
        # Apply global threadpool knobs early, so later CUDA->CPU switch is not constrained.
        self._configure_cpu_runtime()
        self._configure_runtime()
        logger.info(f"IndicF5 inference engine initialized on device: {self.device}")

    def _get_preprocessed_reference(self, ref_audio_path: str, ref_text: str):
        """Cache preprocessed reference audio/text to avoid repeated CPU-heavy preprocessing."""
        try:
            stat = os.stat(ref_audio_path)
            cache_key = (
                os.path.abspath(ref_audio_path),
                int(stat.st_mtime),
                stat.st_size,
                (ref_text or "").strip(),
            )
        except Exception:
            cache_key = (os.path.abspath(ref_audio_path), (ref_text or "").strip())

        cached = self._ref_preprocess_cache.get(cache_key)
        if cached is not None:
            cached_audio, cached_text = cached
            if cached_audio and os.path.exists(cached_audio):
                return cached_audio, cached_text

        from f5_tts.infer.utils_infer import preprocess_ref_audio_text

        pre_audio, pre_text = preprocess_ref_audio_text(ref_audio_path, ref_text)

        self._ref_preprocess_cache[cache_key] = (pre_audio, pre_text)
        if len(self._ref_preprocess_cache) > self._max_ref_cache_size:
            oldest_key = next(iter(self._ref_preprocess_cache))
            old_audio, _ = self._ref_preprocess_cache.pop(oldest_key)
            try:
                if old_audio and os.path.exists(old_audio):
                    os.remove(old_audio)
            except Exception:
                pass

        return pre_audio, pre_text

    def _configure_cpu_runtime(self) -> None:
        """Aggressive CPU runtime tuning for faster inference on low-end systems."""
        cpu_threads = max(1, int(os.cpu_count() or 1))

        # Thread env vars (best-effort)
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_threads)
        os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
        os.environ.setdefault("KMP_BLOCKTIME", "1")

        try:
            torch.set_num_threads(cpu_threads)
            torch.set_num_interop_threads(max(1, cpu_threads // 2))
        except Exception as e:
            logger.warning(f"[CPU] torch thread tuning failed: {str(e)}")

        if hasattr(torch.backends, "mkldnn"):
            try:
                torch.backends.mkldnn.enabled = True
            except Exception:
                pass

        # Control external BLAS/OpenMP threadpools when available
        try:
            from threadpoolctl import threadpool_limits

            threadpool_limits(limits=cpu_threads)
        except Exception as e:
            logger.debug(f"[CPU] threadpoolctl tuning skipped: {str(e)}")

        # Pin process to all cores and prefer higher scheduling priority
        try:
            if hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, set(range(cpu_threads)))
        except Exception as e:
            logger.debug(f"[CPU] affinity setup skipped: {str(e)}")

        try:
            import psutil

            process = psutil.Process(os.getpid())
            if hasattr(process, "cpu_affinity"):
                process.cpu_affinity(list(range(cpu_threads)))
            try:
                process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -10)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[CPU] psutil priority/affinity tuning skipped: {str(e)}")

        logger.info(
            "[CPU] aggressive runtime: threads=%d interop_threads=%d mkldnn=%s",
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            getattr(torch.backends, "mkldnn", None).enabled if hasattr(torch.backends, "mkldnn") else "n/a",
        )

    def _optimize_model_for_cpu(self) -> None:
        """Apply optional CPU model optimizations."""
        if self.model is None or self.device.type != "cpu" or self._cpu_optimized:
            return

        # Dynamic quantization (best-effort)
        try:
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            logger.info("[CPU] dynamic quantization enabled for Linear layers")
        except Exception as e:
            logger.debug(f"[CPU] dynamic quantization skipped: {str(e)}")

        # Intel Extension for PyTorch (optional)
        try:
            import intel_extension_for_pytorch as ipex

            self.model = ipex.optimize(self.model, dtype=torch.float32, inplace=True)
            logger.info("[CPU] Intel Extension for PyTorch optimization enabled")
        except Exception as e:
            logger.debug(f"[CPU] IPEX optimization skipped: {str(e)}")

        self._cpu_optimized = True

    def _configure_runtime(self) -> None:
        """Apply runtime performance settings."""
        if torch.cuda.is_available():
            if self.config.model.enable_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = self.config.model.cudnn_benchmark
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        if self.device.type == "cpu":
            try:
                self._configure_cpu_runtime()
            except Exception as e:
                logger.warning(f"[CPU] runtime tuning skipped: {str(e)}")

    def _autocast_context(self):
        """Return autocast context when enabled and supported."""
        if not self.config.model.use_mixed_precision:
            return contextlib.nullcontext()
        if self.device.type != "cuda":
            return contextlib.nullcontext()

        dtype = torch.float16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _gpu_mem_gb(self) -> Tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return allocated, reserved

    def _warmup(self) -> None:
        """Optional one-time warmup to reduce first-request latency."""
        if self._warmed_up or not self.config.model.warmup_on_load or self.model is None:
            return
        try:
            dummy_audio = None
            prompts_dir = self.config.root / "prompts"
            if prompts_dir.exists():
                for ext in ("*.wav", "*.mp3", "*.flac"):
                    found = list(prompts_dir.glob(ext))
                    if found:
                        dummy_audio = str(found[0])
                        break
            if not dummy_audio:
                self._warmed_up = True
                return
            logger.info("Running one-time warmup inference...")
            with torch.no_grad():
                _ = self.model(
                    text="warmup",
                    ref_audio_path=dummy_audio,
                    ref_text="warmup",
                )
            self._warmed_up = True
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup skipped: {str(e)}")
            self._warmed_up = True
    
    def load_model(self) -> bool:
        """
        Load the IndicF5 model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading model from {self.config.model.model_repo}...")
            
            # Import here to avoid issues if transformers not available
            from transformers import AutoModel
            
            self.model = AutoModel.from_pretrained(
                self.config.model.model_repo, 
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
            self._compiled = False
            self._cpu_optimized = False

            if (
                self.config.model.torch_compile
                and hasattr(torch, "compile")
                and self.device.type == "cuda"
            ):
                try:
                    self.model = torch.compile(self.model, mode=self.config.model.compile_mode)
                    self._compiled = True
                    logger.info(f"Applied torch.compile(mode={self.config.model.compile_mode})")
                except Exception as e:
                    logger.warning(f"torch.compile skipped: {str(e)}")

            self.model.eval()
            if self.device.type == "cpu":
                self._optimize_model_for_cpu()
            self._warmup()
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            return False
    
    def load_all_models(self) -> bool:
        """Load model (remote model already bundles inference stack)"""
        return self.load_model()
    
    def move_to_device(self, device: str) -> None:
        """
        Move models to specified device
        
        Args:
            device: Device string ('cuda' or 'cpu')
        """
        try:
            previous_device = self.device
            self.device = torch.device(device)
            self._configure_runtime()

            # torch.compile artifacts can retain CUDA assumptions.
            # Reloading model on CPU is safer for real CPU-only inference.
            should_reload = (
                self.model is not None
                and previous_device.type == "cuda"
                and self.device.type == "cpu"
                and self._compiled
            )

            if should_reload:
                logger.info("Reloading model for CPU mode to avoid CUDA-compiled graph usage")
                self.model = None
                self._compiled = False
                self._cpu_optimized = False
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not self.load_model():
                    raise RuntimeError("Failed to reload model for CPU mode")

            if self.model is not None:
                self.model = self.model.to(self.device)
                if hasattr(self.model, "ema_model") and self.model.ema_model is not None:
                    self.model.ema_model = self.model.ema_model.to(self.device)
                if hasattr(self.model, "vocoder") and self.model.vocoder is not None:
                    self.model.vocoder = self.model.vocoder.to(self.device)

            if self.device.type == "cpu":
                self._optimize_model_for_cpu()

            if self.device.type == "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Models moved to device: {device}")
        except Exception as e:
            logger.error(f"Failed to move models to device: {str(e)}", exc_info=True)
            raise
    
    def clear_cache(self) -> None:
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
    
    def _preprocess_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            audio, sr = torchaudio.load(audio_path)
            
            # Normalize loudness
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < self.config.audio.target_rms:
                audio = audio * self.config.audio.target_rms / rms
            
            # Resample if needed
            if sr != self.config.audio.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.audio.sample_rate)
                audio = resampler(audio)
            
            return audio.to(self.device), self.config.audio.sample_rate
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}", exc_info=True)
            raise
    
    def _prepare_text(self, ref_text: str, gen_text: str) -> str:
        """
        Prepare and validate text
        
        Args:
            ref_text: Reference text
            gen_text: Text to generate
            
        Returns:
            Normalized generation text
        """
        if not ref_text or not gen_text:
            raise ValueError("Both reference text and generation text are required")
        
        return gen_text.strip()
    
    @torch.inference_mode()
    def synthesize(
        self,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        speed: float = 1.0,
        nfe_steps: int | None = None,
        cfg_strength: float | None = None,
        remove_silence: bool = True,
        min_silence_duration_ms: int = 1000,
        silence_threshold_db: int = -50,
    ) -> Tuple[np.ndarray, int, dict]:
        """
        Synthesize speech from text
        
        Args:
            ref_audio_path: Path to reference audio
            ref_text: Reference text 
            gen_text: Text to generate
            speed: Speed multiplier
            remove_silence: Whether to remove silence gaps
            min_silence_duration_ms: Minimum silence duration in milliseconds (default: 1000ms)
            silence_threshold_db: Silence threshold in dB (default: -50dB)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            start_time = time.time()
            input_chars = len(gen_text.strip())
            gpu_alloc_before, gpu_reserved_before = self._gpu_mem_gb()
            
            # Check model is loaded
            if self.model is None:
                logger.warning("Models not loaded, loading now...")
                if not self.load_all_models():
                    raise RuntimeError("Failed to load models")
            
            logger.info(
                "[GEN] start device=%s nfe_steps=%s cfg_strength=%s speed=%.2f",
                self.device,
                nfe_steps if nfe_steps is not None else self.config.inference.nfe_steps,
                cfg_strength if cfg_strength is not None else self.config.inference.cfg_strength,
                float(speed),
            )
            
            # Prepare text
            logger.debug("Preparing text...")
            normalized_gen_text = self._prepare_text(ref_text, gen_text)
            effective_nfe_steps = int(nfe_steps if nfe_steps is not None else self.config.inference.nfe_steps)
            effective_cfg_strength = float(
                cfg_strength if cfg_strength is not None else self.config.inference.cfg_strength
            )
            if self.device.type == "cpu":
                effective_nfe_steps = max(4, min(24, effective_nfe_steps))
            else:
                effective_nfe_steps = max(8, min(64, effective_nfe_steps))
            effective_cfg_strength = max(0.5, min(5.0, effective_cfg_strength))

            if hasattr(self.model, "config"):
                try:
                    self.model.config.speed = float(speed)
                    if hasattr(self.model.config, "remove_sil"):
                        self.model.config.remove_sil = bool(remove_silence)
                except Exception as e:
                    logger.warning(f"Could not apply runtime model config: {str(e)}")

            infer_start = time.time()
            generated_wave = None

            if hasattr(self.model, "ema_model") and hasattr(self.model, "vocoder"):
                try:
                    from f5_tts.infer.utils_infer import infer_process

                    logger.info(
                        "[GEN] fast-path=ema_model nfe_steps=%d cfg_strength=%.2f speed=%.2f",
                        effective_nfe_steps,
                        effective_cfg_strength,
                        float(speed),
                    )

                    ref_audio, processed_ref_text = self._get_preprocessed_reference(ref_audio_path, ref_text)

                    with torch.no_grad(), self._autocast_context():
                        generated_wave, _, _ = infer_process(
                            ref_audio,
                            processed_ref_text,
                            normalized_gen_text,
                            self.model.ema_model,
                            self.model.vocoder,
                            nfe_step=effective_nfe_steps,
                            cfg_strength=effective_cfg_strength,
                            sway_sampling_coef=self.config.inference.sway_sampling_coef,
                            speed=max(0.5, float(speed)),
                            device=self.device,
                        )
                except Exception as e:
                    logger.warning(f"Fast-path failed, falling back to HF wrapper: {str(e)}")

            if generated_wave is None:
                logger.info("[GEN] fast-path unavailable; using HF wrapper")
                with torch.no_grad(), self._autocast_context():
                    generated_wave = self.model(
                        text=normalized_gen_text,
                        ref_audio_path=ref_audio_path,
                        ref_text=ref_text,
                    )

            infer_elapsed = time.time() - infer_start

            if isinstance(generated_wave, torch.Tensor):
                generated_wave = generated_wave.detach().cpu().numpy()
            generated_wave = np.asarray(generated_wave).squeeze()

            if generated_wave.dtype == np.int16:
                generated_wave = generated_wave.astype(np.float32) / 32768.0
            else:
                generated_wave = generated_wave.astype(np.float32)

            sr = self.config.audio.sample_rate
            
            # Remove silence if requested
            if remove_silence:
                logger.debug(f"Removing silence (min_duration={min_silence_duration_ms}ms, threshold={silence_threshold_db}dB)...")
                generated_wave = self._remove_silence(
                    generated_wave,
                    min_silence_len=min_silence_duration_ms,
                    silence_thresh=silence_threshold_db
                )
            
            latency = time.time() - start_time
            audio_sec = max(len(generated_wave) / sr, 1e-6)
            rtf = latency / audio_sec
            gpu_alloc_after, gpu_reserved_after = self._gpu_mem_gb()

            logger.info(
                "[GEN] chars=%d speed=%.2f device=%s time=%.2fs infer=%.2fs audio=%.2fs rtf=%.2f "
                "gpu_alloc=%.2fGB->%.2fGB gpu_reserved=%.2fGB->%.2fGB",
                input_chars,
                speed,
                self.device,
                latency,
                infer_elapsed,
                audio_sec,
                rtf,
                gpu_alloc_before,
                gpu_alloc_after,
                gpu_reserved_before,
                gpu_reserved_after,
            )
            
            metrics = {
                "total_time": float(latency),
                "infer_time": float(infer_elapsed),
                "audio_duration": float(audio_sec),
                "rtf": float(rtf),
                "device": str(self.device),
                "chars": int(input_chars),
            }
            return generated_wave, sr, metrics
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}", exc_info=True)
            raise
    
    def _remove_silence(self, audio: np.ndarray, min_silence_len: int = 1000, 
                       silence_thresh: int = -50, keep_silence: int = 500) -> np.ndarray:
        """
        Remove silence from audio
        
        Args:
            audio: Audio array
            min_silence_len: Minimum silence length in ms
            silence_thresh: Silence threshold in dB
            keep_silence: Keep silence length in ms
            
        Returns:
            Audio array with silence removed
        """
        try:
            from pydub import AudioSegment, silence
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.config.audio.sample_rate)
                temp_path = f.name
            
            # Process
            aseg = AudioSegment.from_file(temp_path)
            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence,
                seek_step=10
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                non_silent_wave += seg
            
            non_silent_wave.export(temp_path, format="wav")
            processed_audio, _ = torchaudio.load(temp_path)
            
            # Cleanup
            os.remove(temp_path)
            
            return processed_audio.squeeze().cpu().numpy()
            
        except ImportError:
            logger.warning("pydub not available, skipping silence removal")
            return audio
        except Exception as e:
            logger.warning(f"Error removing silence: {str(e)}, continuing without silence removal")
            return audio


# Singleton
_inference_engine = None


def get_inference_engine() -> IndicF5InferenceEngine:
    """Get or create the inference engine"""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = IndicF5InferenceEngine()
    return _inference_engine
