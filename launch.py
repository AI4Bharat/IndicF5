#!/usr/bin/env python3
"""
IndicF5 Neo: Fast and Scalable Text-to-Speech

Main entry point for the application.
Run: python launch.py
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import soundfile as sf
import torch
import gradio as gr

# Import our modules
from src.config import get_config
from src.utils.logger import LoggerSetup, get_logger
from src.utils.device_manager import DeviceManager
from src.inference.engine import get_inference_engine
from src.inference.subtitles import SubtitleGenerator
from src.data_management.profile_manager import VoiceProfileManager

# Setup logging
config = get_config()
LoggerSetup.setup(log_dir=config.paths.logs_dir)
logger = get_logger(__name__)

# Initialize managers
device_mgr = DeviceManager()
profile_mgr = VoiceProfileManager(profiles_dir=str(config.paths.profiles_dir))
inference_engine = get_inference_engine()

# Output directory
output_dir = config.paths.outputs_dir / "history"
output_dir.mkdir(parents=True, exist_ok=True)
subtitle_generator = SubtitleGenerator(output_dir=output_dir)

logger.info("=" * 50)
logger.info("IndicF5 Neo - Text-to-Speech Application")
logger.info("=" * 50)
logger.info(
    "Perf mode: device=%s mixed_precision=%s tf32=%s cudnn_benchmark=%s torch_compile=%s",
    config.get_device(),
    config.model.use_mixed_precision,
    config.model.enable_tf32,
    config.model.cudnn_benchmark,
    config.model.torch_compile,
)


class UIState:
    """Manage UI state"""
    def __init__(self):
        self.model_loaded = False


ui_state = UIState()


def get_history_files():
    """Get list of generated audio files"""
    try:
        files = sorted(output_dir.glob("*.wav"), key=os.path.getmtime, reverse=True)
        return [[str(f), f.name] for f in files[:50]]  # Limit to 50 most recent
    except Exception as e:
        logger.error(f"Error getting history files: {str(e)}")
        return []


def get_profile_table_rows():
    """Get profile table rows"""
    rows = []
    for name in profile_mgr.list_profiles():
        profile = profile_mgr.get_profile(name) or {}
        audio_path = profile.get("audio_path", "")
        audio_file = Path(audio_path).name if audio_path else ""
        last_modified = profile.get("updated_at") or profile.get("created_at") or ""
        rows.append([name, profile.get("ref_text", ""), audio_file, last_modified])
    return rows


def get_profile_action_button(profile_name: str):
    """Get create/update profile button state based on selected profile"""
    if profile_name and profile_name != "None":
        return gr.Button(value="🔄 Update Profile", variant="primary")
    return gr.Button(value="➕ Create New Profile", variant="primary")


def synthesize(
    profile_name,
    ref_audio,
    ref_text,
    gen_text,
    subtitle_lang,
    generate_srt,
    remove_silence,
    min_silence_duration_ms,
    silence_threshold_db,
    speed,
    nfe_steps,
    cfg_strength,
    device_type,
):
    """Synthesize speech from text"""
    try:
        req_start = time.time()
        logger.info(
            "[REQ] profile=%s device=%s remove_silence=%s min_silence=%dms threshold=%ddB speed=%.2f nfe_steps=%s cfg=%.2f",
            profile_name,
            device_type,
            remove_silence,
            min_silence_duration_ms,
            silence_threshold_db,
            float(speed),
            nfe_steps,
            float(cfg_strength),
        )
        
        # Change device if needed
        current_device = str(device_mgr.get_current_device()).replace(":0", "")
        if device_type == "cuda" and not torch.cuda.is_available():
            device_type = "cpu"
        if device_type != current_device:
            logger.info(f"Switching to device: {device_type}")
            device_mgr.set_device(device_type)
            inference_engine.move_to_device(device_type)
        
        # Load models if needed
        if not ui_state.model_loaded:
            logger.info("Loading models...")
            if not inference_engine.load_all_models():
                raise gr.Error("Failed to load models. Check logs for details.")
            ui_state.model_loaded = True
        
        # Get reference audio and text from profile if selected
        if profile_name and profile_name != "None":
            profile = profile_mgr.get_profile(profile_name)
            if profile:
                ref_audio = profile["audio_path"]
                ref_text = profile["ref_text"]
                logger.info(f"Using profile: {profile_name}")
        
        # Validate inputs
        if not ref_audio:
            raise gr.Error("Reference audio is required")
        if not ref_text:
            raise gr.Error("Reference text is required")
        if not gen_text:
            raise gr.Error("Text to generate is required")
        
        # Synthesize
        audio, sr, metrics = inference_engine.synthesize(
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            nfe_steps=nfe_steps,
            cfg_strength=cfg_strength,
            remove_silence=remove_silence,
            min_silence_duration_ms=int(min_silence_duration_ms),
            silence_threshold_db=int(silence_threshold_db),
        )
        
        # Save to history
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"gen_{timestamp}.wav"
        sf.write(str(output_path), audio, sr)
        logger.info(f"Audio saved to {output_path}")

        srt_file = None
        srt_method = "disabled"
        if generate_srt:
            try:
                srt_file, srt_method = subtitle_generator.generate_srt(
                    audio_path=output_path,
                    transcript=gen_text,
                    language=subtitle_lang,
                )
                logger.info("SRT saved to %s", srt_file)
            except Exception as e:
                logger.warning("SRT generation failed: %s", str(e))

        request_elapsed = time.time() - req_start
        status = (
            f"✓ Generated on {metrics.get('device', device_type)} | "
            f"infer {metrics.get('infer_time', 0.0):.2f}s | "
            f"total {metrics.get('total_time', 0.0):.2f}s | "
            f"rtf {metrics.get('rtf', 0.0):.2f} | "
            f"req {request_elapsed:.2f}s"
        )
        if generate_srt:
            status += f" | srt {srt_method}"
        logger.info("[REQ] completed in %.2fs", request_elapsed)
        return (sr, audio), status, get_history_files(), srt_file
        
    except gr.Error:
        raise
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}", exc_info=True)
        raise gr.Error(f"Synthesis failed: {str(e)}")


def save_or_update_profile(selected_profile, name, audio, ref_text):
    """Create a new profile or update the currently selected profile"""
    try:
        typed_name = (name or "").strip()
        selected = selected_profile if selected_profile and selected_profile != "None" else None

        if not audio:
            profiles = ["None"] + profile_mgr.list_profiles()
            return (
                gr.Dropdown(choices=profiles, value=selected_profile or "None"),
                gr.Dropdown(choices=profile_mgr.list_profiles(), value=selected),
                get_profile_table_rows(),
                "Audio file is required",
                get_profile_action_button(selected_profile or "None"),
                gr.Textbox(value=typed_name),
            )
        if not ref_text:
            profiles = ["None"] + profile_mgr.list_profiles()
            return (
                gr.Dropdown(choices=profiles, value=selected_profile or "None"),
                gr.Dropdown(choices=profile_mgr.list_profiles(), value=selected),
                get_profile_table_rows(),
                "Reference text is required",
                get_profile_action_button(selected_profile or "None"),
                gr.Textbox(value=typed_name),
            )

        if selected:
            if not profile_mgr.update_profile(selected, ref_text=ref_text, audio_path=audio):
                raise ValueError(f"Failed to update profile '{selected}'")
            active_name = selected
            message = f"Profile '{selected}' updated successfully!"
            logger.info(f"Profile updated: {selected}")
        else:
            if not typed_name:
                raise ValueError("Profile name is required for new profile")
            if typed_name in profile_mgr.list_profiles():
                raise ValueError(
                    f"Profile '{typed_name}' already exists. Select it from dropdown to update."
                )
            profile_mgr.save_profile(typed_name, audio, ref_text)
            active_name = typed_name
            message = f"Profile '{typed_name}' created successfully!"
            logger.info(f"Profile created: {typed_name}")

        profiles = ["None"] + profile_mgr.list_profiles()
        return (
            gr.Dropdown(choices=profiles, value=active_name),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=active_name),
            get_profile_table_rows(),
            message,
            get_profile_action_button(active_name),
            gr.Textbox(value=active_name),
        )
    except Exception as e:
        logger.error(f"Failed to save/update profile: {str(e)}")
        profiles = ["None"] + profile_mgr.list_profiles()
        selected = selected_profile if selected_profile and selected_profile != "None" else None
        return (
            gr.Dropdown(choices=profiles, value=selected_profile or "None"),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=selected),
            get_profile_table_rows(),
            f"Failed to save/update profile: {str(e)}",
            get_profile_action_button(selected_profile or "None"),
            gr.Textbox(value=(name or "")),
        )


def load_selected_profile(profile_name):
    """Load selected profile data into reference inputs"""
    try:
        if not profile_name or profile_name == "None":
            return (
                None,
                "",
                gr.Textbox(value=""),
                get_profile_action_button("None"),
                "Create mode: enter name, upload audio, and transcript",
            )

        profile = profile_mgr.get_profile(profile_name)
        if not profile:
            return (
                None,
                "",
                gr.Textbox(value=""),
                get_profile_action_button("None"),
                f"Profile '{profile_name}' not found",
            )

        return (
            profile.get("audio_path"),
            profile.get("ref_text", ""),
            gr.Textbox(value=profile_name),
            get_profile_action_button(profile_name),
            f"Loaded profile '{profile_name}'",
        )
    except Exception as e:
        logger.error(f"Failed to load selected profile: {str(e)}")
        return (
            None,
            "",
            gr.Textbox(value=""),
            get_profile_action_button("None"),
            f"Failed to load profile: {str(e)}",
        )


def delete_profile(name):
    """Delete a profile"""
    try:
        if not name:
            profiles = ["None"] + profile_mgr.list_profiles()
            return (
                gr.Dropdown(choices=profiles, value="None"),
                gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
                "Select a profile to delete",
                get_profile_table_rows(),
                None,
                "",
                gr.Textbox(value=""),
                get_profile_action_button("None"),
                "Create mode: enter name, upload audio, and transcript",
            )
        
        if profile_mgr.delete_profile(name):
            profiles = ["None"] + profile_mgr.list_profiles()
            logger.info(f"Profile deleted: {name}")
            return (
                gr.Dropdown(choices=profiles, value="None"),
                gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
                f"Profile '{name}' deleted",
                get_profile_table_rows(),
                None,
                "",
                gr.Textbox(value=""),
                get_profile_action_button("None"),
                "Create mode: enter name, upload audio, and transcript",
            )
        else:
            profiles = ["None"] + profile_mgr.list_profiles()
            return (
                gr.Dropdown(choices=profiles, value="None"),
                gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
                "Failed to delete profile",
                get_profile_table_rows(),
                None,
                "",
                gr.Textbox(value=""),
                get_profile_action_button("None"),
                "Create mode: enter name, upload audio, and transcript",
            )
    except Exception as e:
        logger.error(f"Error deleting profile: {str(e)}")
        profiles = ["None"] + profile_mgr.list_profiles()
        return (
            gr.Dropdown(choices=profiles, value="None"),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
            f"Error: {str(e)}",
            get_profile_table_rows(),
            None,
            "",
            gr.Textbox(value=""),
            get_profile_action_button("None"),
            "Create mode: enter name, upload audio, and transcript",
        )


def offload_model():
    """Offload model to CPU"""
    try:
        inference_engine.move_to_device('cpu')
        inference_engine.clear_cache()
        logger.info("Model offloaded to CPU")
        return "✓ Model offloaded to CPU"
    except Exception as e:
        logger.error(f"Error offloading model: {str(e)}")
        return f"Error: {str(e)}"


def refresh_profiles():
    """Refresh profile list"""
    profiles = ["None"] + profile_mgr.list_profiles()
    return (
        gr.Dropdown(choices=profiles, value="None"),
        gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
        get_profile_table_rows(),
        get_profile_action_button("None"),
        "Create mode: enter name, upload audio, and transcript",
    )


def manage_load_profile(profile_name):
    """Load selected profile into Manage Profiles editor"""
    try:
        profiles_with_none = ["None"] + profile_mgr.list_profiles()

        if not profile_name or profile_name == "None":
            create_status = "Create mode: enter name, upload audio, and transcript"
            return (
                gr.Dropdown(choices=profiles_with_none, value="None"),
                gr.Dropdown(choices=profiles_with_none, value="None"),
                gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
                gr.Textbox(value=""),
                None,
                "",
                get_profile_action_button("None"),
                create_status,
                get_profile_action_button("None"),
                gr.Textbox(value=""),
                None,
                "",
                create_status,
            )

        profile = profile_mgr.get_profile(profile_name)
        if not profile:
            return (
                gr.Dropdown(choices=profiles_with_none, value="None"),
                gr.Dropdown(choices=profiles_with_none, value="None"),
                gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
                gr.Textbox(value=""),
                None,
                "",
                get_profile_action_button("None"),
                f"Profile '{profile_name}' not found",
                get_profile_action_button("None"),
                gr.Textbox(value=""),
                None,
                "",
                f"Profile '{profile_name}' not found",
            )

        loaded_status = f"Loaded profile '{profile_name}'"
        return (
            gr.Dropdown(choices=profiles_with_none, value=profile_name),
            gr.Dropdown(choices=profiles_with_none, value=profile_name),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=profile_name),
            gr.Textbox(value=profile_name),
            profile.get("audio_path"),
            profile.get("ref_text", ""),
            get_profile_action_button(profile_name),
            loaded_status,
            get_profile_action_button(profile_name),
            gr.Textbox(value=profile_name),
            profile.get("audio_path"),
            profile.get("ref_text", ""),
            loaded_status,
        )
    except Exception as e:
        logger.error(f"Failed to load manage profile: {str(e)}")
        return (
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Dropdown(),
            gr.Textbox(),
            None,
            "",
            get_profile_action_button("None"),
            f"Failed to load profile: {str(e)}",
            get_profile_action_button("None"),
            gr.Textbox(),
            None,
            "",
            f"Failed to load profile: {str(e)}",
        )


def manage_save_or_update_profile(selected_profile, name, audio, ref_text):
    """Save or update profile from Manage Profiles editor"""
    try:
        typed_name = (name or "").strip()
        selected = selected_profile if selected_profile and selected_profile != "None" else None

        if not audio:
            raise ValueError("Audio file is required")
        if not ref_text:
            raise ValueError("Reference text is required")

        if selected:
            if not profile_mgr.update_profile(selected, ref_text=ref_text, audio_path=audio):
                raise ValueError(f"Failed to update profile '{selected}'")
            active_name = selected
            message = f"Profile '{selected}' updated successfully!"
            logger.info(f"Profile updated from manage tab: {selected}")
        else:
            if not typed_name:
                raise ValueError("Profile name is required for new profile")
            if typed_name in profile_mgr.list_profiles():
                raise ValueError(
                    f"Profile '{typed_name}' already exists. Select it from dropdown to update."
                )
            profile_mgr.save_profile(typed_name, audio, ref_text)
            active_name = typed_name
            message = f"Profile '{typed_name}' created successfully!"
            logger.info(f"Profile created from manage tab: {typed_name}")

        profile = profile_mgr.get_profile(active_name) or {}
        profiles_with_none = ["None"] + profile_mgr.list_profiles()
        return (
            gr.Dropdown(choices=profiles_with_none, value=active_name),
            gr.Dropdown(choices=profiles_with_none, value=active_name),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=active_name),
            get_profile_table_rows(),
            gr.Textbox(value=active_name),
            profile.get("audio_path"),
            profile.get("ref_text", ref_text),
            get_profile_action_button(active_name),
            message,
            get_profile_action_button(active_name),
            gr.Textbox(value=active_name),
            profile.get("audio_path"),
            profile.get("ref_text", ref_text),
            message,
        )
    except Exception as e:
        logger.error(f"Failed to save/update from manage tab: {str(e)}")
        profiles_with_none = ["None"] + profile_mgr.list_profiles()
        current_value = selected_profile if selected_profile in profiles_with_none else "None"
        selected_del = None if current_value == "None" else current_value
        return (
            gr.Dropdown(choices=profiles_with_none, value=current_value),
            gr.Dropdown(choices=profiles_with_none, value=current_value),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=selected_del),
            get_profile_table_rows(),
            gr.Textbox(value=(name or "")),
            audio,
            ref_text or "",
            get_profile_action_button(current_value),
            f"Failed to save/update profile: {str(e)}",
            get_profile_action_button(current_value),
            gr.Textbox(value=(name or "")),
            audio,
            ref_text or "",
            f"Failed to save/update profile: {str(e)}",
        )


def manage_delete_profile(name):
    """Delete profile from Manage Profiles editor"""
    try:
        if not name:
            raise ValueError("Select a profile to delete")

        if not profile_mgr.delete_profile(name):
            raise ValueError(f"Failed to delete profile '{name}'")

        profiles_with_none = ["None"] + profile_mgr.list_profiles()
        status = f"Profile '{name}' deleted"
        create_status = "Create mode: enter name, upload audio, and transcript"
        logger.info(f"Profile deleted from manage tab: {name}")
        return (
            gr.Dropdown(choices=profiles_with_none, value="None"),
            gr.Dropdown(choices=profiles_with_none, value="None"),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
            get_profile_table_rows(),
            gr.Textbox(value=""),
            None,
            "",
            get_profile_action_button("None"),
            status,
            get_profile_action_button("None"),
            gr.Textbox(value=""),
            None,
            "",
            create_status,
        )
    except Exception as e:
        logger.error(f"Error deleting from manage tab: {str(e)}")
        profiles_with_none = ["None"] + profile_mgr.list_profiles()
        return (
            gr.Dropdown(choices=profiles_with_none, value="None"),
            gr.Dropdown(choices=profiles_with_none, value="None"),
            gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
            get_profile_table_rows(),
            gr.Textbox(value=""),
            None,
            "",
            get_profile_action_button("None"),
            f"Error: {str(e)}",
            get_profile_action_button("None"),
            gr.Textbox(value=""),
            None,
            "",
            "Create mode: enter name, upload audio, and transcript",
        )


def manage_refresh_profiles():
    """Refresh all profile UI controls from Manage Profiles tab"""
    profiles_with_none = ["None"] + profile_mgr.list_profiles()
    status = "Refreshed profiles"
    create_status = "Create mode: enter name, upload audio, and transcript"
    return (
        gr.Dropdown(choices=profiles_with_none, value="None"),
        gr.Dropdown(choices=profiles_with_none, value="None"),
        gr.Dropdown(choices=profile_mgr.list_profiles(), value=None),
        get_profile_table_rows(),
        gr.Textbox(value=""),
        None,
        "",
        get_profile_action_button("None"),
        status,
        get_profile_action_button("None"),
        gr.Textbox(value=""),
        None,
        "",
        create_status,
    )


def load_audio_from_history(table_data, evt: gr.SelectData):
    """Load audio from history"""
    try:
        if not evt or not table_data:
            return None
        if hasattr(evt, "index") and isinstance(evt.index, (list, tuple)) and len(evt.index) > 0:
            row_idx = evt.index[0]
            if isinstance(row_idx, int) and 0 <= row_idx < len(table_data):
                selected = table_data[row_idx][0]
                if isinstance(selected, str) and os.path.exists(selected):
                    return selected
    except Exception as e:
        logger.warning(f"Error loading from history: {str(e)}")
    return None


def optimize_settings_for_device(device_type: str):
    """Apply practical performance defaults when device is changed"""
    if device_type == "cpu":
        return (
            gr.Slider(value=6),
            gr.Slider(value=1.0),
            gr.Slider(value=1.1),
            gr.Checkbox(value=False),
            "CPU turbo preset applied (faster, slightly lower quality)",
        )
    return (
        gr.Slider(value=12),
        gr.Slider(value=1.2),
        gr.Slider(value=1.15),
        gr.Checkbox(value=False),
        "CUDA turbo preset applied",
    )


def get_optimal_queue_concurrency() -> int:
    """Determine queue concurrency for cloud scaling (CPU/GPU aware)."""
    env_override = os.getenv("INDICF5_QUEUE_CONCURRENCY", "").strip()
    if env_override:
        try:
            value = int(env_override)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid INDICF5_QUEUE_CONCURRENCY='%s', using auto", env_override)

    device = config.get_device()
    if device == "cuda" and torch.cuda.is_available():
        gpu_default = os.getenv("INDICF5_GPU_CONCURRENCY", "2").strip()
        try:
            value = int(gpu_default)
            return max(1, min(8, value))
        except ValueError:
            return 2

    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count // 8 if cpu_count >= 8 else 1))


def get_server_max_threads(queue_concurrency: int) -> int:
    """Estimate server thread budget for API + preprocessing workloads."""
    env_override = os.getenv("INDICF5_MAX_THREADS", "").strip()
    if env_override:
        try:
            value = int(env_override)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Invalid INDICF5_MAX_THREADS='%s', using auto", env_override)
    return max(40, queue_concurrency * 16)


# Build Gradio Interface
with gr.Blocks(
    title="IndicF5 Neo"
) as app:
    
    gr.HTML("""
    <div class="container">
        <div class="header">
            <h1>🚀 IndicF5 Neo</h1>
            <p>Fast & Scalable Text-to-Speech for Indic Languages</p>
        </div>
    </div>
    """)
    
    with gr.Tabs():
        
        # ============ GENERATE TAB ============
        with gr.Tab("⚡ Generate Speech"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Selection")
                    profile_dropdown = gr.Dropdown(
                        choices=["None"] + profile_mgr.list_profiles(),
                        value="None",
                        label="Voice Profile"
                    )
                    refresh_btn = gr.Button("🔄 Refresh", scale=1)
                    
                    gr.Markdown("### Reference Audio")
                    with gr.Accordion("Upload Reference", open=True):
                        ref_audio_in = gr.Audio(
                            label="Audio File",
                            type="filepath"
                        )
                        ref_text_in = gr.Textbox(
                            label="Reference Text",
                            lines=2,
                            placeholder="Transcription of the reference audio..."
                        )
                        profile_name_in = gr.Textbox(
                            label="Profile Name",
                            placeholder="e.g., My Voice"
                        )
                        profile_action_btn = gr.Button("➕ Create New Profile", variant="primary")
                        profile_action_status = gr.Textbox(
                            label="Profile Status",
                            value="Create mode: enter name, upload audio, and transcript",
                            interactive=False,
                        )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Text Generation")
                    gen_text_in = gr.Textbox(
                        label="Text to Generate",
                        lines=6,
                        placeholder="Enter the text you want to synthesize..."
                    )
                    
                    with gr.Row():
                        speed_slider = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.2,
                            step=0.1,
                            label="Speed"
                        )
                        device_radio = gr.Radio(
                            choices=["cuda", "cpu"],
                            value="cuda" if torch.cuda.is_available() else "cpu",
                            label="Device",
                            interactive=True
                        )
                    
                    with gr.Row():
                        remove_silence_chk = gr.Checkbox(
                            label="Remove Long Gaps",
                            value=False
                        )
                        generate_srt_chk = gr.Checkbox(
                            label="Generate SRT",
                            value=True,
                        )
                        offload_btn = gr.Button("♻️ Offload", scale=1)

                    subtitle_lang = gr.Dropdown(
                        choices=["ta", "hi", "te", "kn", "ml", "bn"],
                        value="ta",
                        label="Subtitle Language",
                        info="Used for forced alignment model selection",
                    )

                    with gr.Row(visible=False) as gap_removal_row:
                        min_silence_slider = gr.Slider(
                            minimum=300,
                            maximum=3000,
                            value=1000,
                            step=100,
                            label="Gap Duration (ms)",
                            info="Minimum gap length to remove"
                        )
                        threshold_slider = gr.Slider(
                            minimum=-80,
                            maximum=-20,
                            value=-50,
                            step=5,
                            label="Gap Threshold (dB)",
                            info="Lower = detects quieter gaps"
                        )

                    with gr.Accordion("Advanced Performance", open=False):
                        with gr.Row():
                            nfe_steps_slider = gr.Slider(
                                minimum=4,
                                maximum=32,
                                value=16,
                                step=1,
                                label="NFE Steps (lower = faster)"
                            )
                            cfg_strength_slider = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                value=1.5,
                                step=0.1,
                                label="CFG Strength"
                            )

                    with gr.Row():
                        generate_btn = gr.Button("🎤 Synthesize", variant="primary", scale=2)
                        stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)
            
            with gr.Row():
                audio_out = gr.Audio(label="Generated Audio", type="numpy")
                status_out = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                srt_out = gr.File(label="Generated Subtitles (.srt)")
        
        
        # ============ PROFILES TAB ============
        with gr.Tab("👥 Manage Profiles"):
            gr.Markdown("### Profile Editor")
            with gr.Row():
                manage_profile_select = gr.Dropdown(
                    choices=["None"] + profile_mgr.list_profiles(),
                    value="None",
                    label="Select Profile",
                )
                manage_refresh_btn = gr.Button("🔄 Refresh", scale=1)
                manage_clear_btn = gr.Button("🧹 Clear", scale=1)

            with gr.Row():
                with gr.Column():
                    manage_profile_name = gr.Textbox(
                        label="Profile Name",
                        placeholder="e.g., My Voice",
                    )
                    manage_profile_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath",
                    )

                with gr.Column():
                    manage_profile_text = gr.Textbox(
                        label="Reference Text",
                        lines=3,
                        placeholder="Transcription of the reference audio...",
                    )
                    manage_save_btn = gr.Button("➕ Create New Profile", variant="primary")

            with gr.Row():
                manage_status = gr.Textbox(
                    label="Manage Status",
                    value="Create mode: enter name, upload audio, and transcript",
                    interactive=False,
                )

            gr.Markdown("### Existing Profiles")
            with gr.Row():
                profile_table = gr.Dataframe(
                    headers=["Name", "Reference Text", "Audio", "Last Modified"],
                    value=get_profile_table_rows(),
                    interactive=False,
                    label="Your Profiles"
                )
            
            gr.Markdown("### Delete Profile")
            with gr.Row():
                del_profile_name = gr.Dropdown(
                    choices=profile_mgr.list_profiles(),
                    label="Profile to Delete"
                )
                del_btn = gr.Button("🗑️ Delete", variant="stop")
        
        
        # ============ HISTORY TAB ============
        with gr.Tab("📜 History"):
            gr.Markdown("### Recent Generations")
            history_table = gr.Dataframe(
                headers=["Path", "Filename"],
                value=get_history_files(),
                interactive=False,
                label="Generated Audio Files"
            )
            
            history_audio = gr.Audio(label="Playback", type="filepath")
            refresh_hist_btn = gr.Button("🔄 Refresh History")
        
        
        # ============ SETTINGS TAB ============
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("### Device Information")
            
            device_info = f"""
            - **Current Device:** {device_mgr.get_device_string()}
            - **CUDA Available:** {torch.cuda.is_available()}
            - **PyTorch Version:** {torch.__version__}
            - **Python Version:** {sys.version.split()[0]}
            """
            gr.Markdown(device_info)
            
            if torch.cuda.is_available():
                mem_info = device_mgr.get_gpu_memory_info()
                mem_text = f"""
                ### GPU Memory
                - **Total:** {mem_info['total_memory'] / 1e9:.2f} GB
                - **Allocated:** {mem_info['allocated_memory'] / 1e9:.2f} GB
                - **Reserved:** {mem_info['reserved_memory'] / 1e9:.2f} GB
                - **Free:** {mem_info['free_memory'] / 1e9:.2f} GB
                """
                gr.Markdown(mem_text)
    # ============ EVENT HANDLERS ============
    
    # Toggle gap removal controls visibility
    def toggle_gap_controls(remove_silence_enabled):
        return gr.Row(visible=remove_silence_enabled)
    
    remove_silence_chk.change(
        fn=toggle_gap_controls,
        inputs=[remove_silence_chk],
        outputs=[gap_removal_row],
    )
    
    # Generate button
    generate_event = generate_btn.click(
        fn=synthesize,
        inputs=[
            profile_dropdown,
            ref_audio_in,
            ref_text_in,
            gen_text_in,
            subtitle_lang,
            generate_srt_chk,
            remove_silence_chk,
            min_silence_slider,
            threshold_slider,
            speed_slider,
            nfe_steps_slider,
            cfg_strength_slider,
            device_radio,
        ],
        outputs=[audio_out, status_out, history_table, srt_out],
        queue=True,
    )

    stop_btn.click(
        fn=lambda: "⏹ Generation interrupted",
        outputs=[status_out],
        cancels=[generate_event],
        queue=False,
    )
    
    # Profile management
    profile_action_btn.click(
        fn=save_or_update_profile,
        inputs=[profile_dropdown, profile_name_in, ref_audio_in, ref_text_in],
        outputs=[
            profile_dropdown,
            del_profile_name,
            profile_table,
            profile_action_status,
            profile_action_btn,
            profile_name_in,
        ],
    )

    profile_dropdown.change(
        fn=load_selected_profile,
        inputs=[profile_dropdown],
        outputs=[
            ref_audio_in,
            ref_text_in,
            profile_name_in,
            profile_action_btn,
            profile_action_status,
        ],
    )
    
    del_btn.click(
        fn=manage_delete_profile,
        inputs=[del_profile_name],
        outputs=[
            manage_profile_select,
            profile_dropdown,
            del_profile_name,
            profile_table,
            manage_profile_name,
            manage_profile_audio,
            manage_profile_text,
            manage_save_btn,
            manage_status,
            profile_action_btn,
            profile_name_in,
            ref_audio_in,
            ref_text_in,
            profile_action_status,
        ],
    )

    manage_profile_select.change(
        fn=manage_load_profile,
        inputs=[manage_profile_select],
        outputs=[
            manage_profile_select,
            profile_dropdown,
            del_profile_name,
            manage_profile_name,
            manage_profile_audio,
            manage_profile_text,
            manage_save_btn,
            manage_status,
            profile_action_btn,
            profile_name_in,
            ref_audio_in,
            ref_text_in,
            profile_action_status,
        ],
    )

    manage_save_btn.click(
        fn=manage_save_or_update_profile,
        inputs=[manage_profile_select, manage_profile_name, manage_profile_audio, manage_profile_text],
        outputs=[
            manage_profile_select,
            profile_dropdown,
            del_profile_name,
            profile_table,
            manage_profile_name,
            manage_profile_audio,
            manage_profile_text,
            manage_save_btn,
            manage_status,
            profile_action_btn,
            profile_name_in,
            ref_audio_in,
            ref_text_in,
            profile_action_status,
        ],
    )

    manage_clear_btn.click(
        fn=lambda: manage_load_profile("None"),
        outputs=[
            manage_profile_select,
            profile_dropdown,
            del_profile_name,
            manage_profile_name,
            manage_profile_audio,
            manage_profile_text,
            manage_save_btn,
            manage_status,
            profile_action_btn,
            profile_name_in,
            ref_audio_in,
            ref_text_in,
            profile_action_status,
        ],
    )

    manage_refresh_btn.click(
        fn=manage_refresh_profiles,
        outputs=[
            manage_profile_select,
            profile_dropdown,
            del_profile_name,
            profile_table,
            manage_profile_name,
            manage_profile_audio,
            manage_profile_text,
            manage_save_btn,
            manage_status,
            profile_action_btn,
            profile_name_in,
            ref_audio_in,
            ref_text_in,
            profile_action_status,
        ],
    )
    
    # Refresh buttons
    refresh_btn.click(
        fn=refresh_profiles,
        outputs=[
            profile_dropdown,
            del_profile_name,
            profile_table,
            profile_action_btn,
            profile_action_status,
        ],
    )
    
    refresh_hist_btn.click(
        fn=get_history_files,
        outputs=[history_table]
    )
    
    # Offload button
    offload_btn.click(
        fn=offload_model,
        outputs=[status_out]
    )

    device_radio.change(
        fn=optimize_settings_for_device,
        inputs=[device_radio],
        outputs=[
            nfe_steps_slider,
            cfg_strength_slider,
            speed_slider,
            remove_silence_chk,
            status_out,
        ],
    )
    
    # History audio playback
    history_table.select(
        fn=load_audio_from_history,
        inputs=[history_table],
        outputs=[history_audio]
    )


def main():
    """Main entry point"""
    try:
        logger.info(f"Starting IndicF5 Neo on {config.ui.host}:{config.ui.port}")
        
        # Pre-load models
        logger.info("Pre-loading models...")
        if inference_engine.load_all_models():
            ui_state.model_loaded = True
            logger.info("Models loaded successfully")
        else:
            logger.warning("Failed to pre-load models, will load on first use")

        queue_concurrency = get_optimal_queue_concurrency()
        server_max_threads = get_server_max_threads(queue_concurrency)

        try:
            app.queue(default_concurrency_limit=queue_concurrency, max_size=config.ui.max_queue_size)
            logger.info(
                "Queue enabled with max_size=%s concurrency=%s",
                config.ui.max_queue_size,
                queue_concurrency,
            )
        except TypeError:
            app.queue(concurrency_count=queue_concurrency, max_size=config.ui.max_queue_size)
            logger.info(
                "Queue enabled (legacy args) with max_size=%s concurrency=%s",
                config.ui.max_queue_size,
                queue_concurrency,
            )
        
        # Launch app
        launch_kwargs = dict(
            server_name=config.ui.host,
            server_port=config.ui.port,
            share=config.ui.share,
            debug=config.ui.debug,
            inbrowser=True,
            allowed_paths=[
                str(config.paths.profiles_dir.resolve()),
                str(config.paths.outputs_dir.resolve()),
            ],
            theme=gr.themes.Soft(),
            max_threads=server_max_threads,
            css="""
            .container { max-width: 1200px; margin: auto; }
            .header { text-align: center; margin-bottom: 20px; }
            .status-box { padding: 10px; border-radius: 5px; margin: 10px 0; }
            """,
        )
        logger.info("Server thread budget: %s", server_max_threads)
        try:
            app.launch(**launch_kwargs)
        except TypeError:
            launch_kwargs.pop("max_threads", None)
            app.launch(**launch_kwargs)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
