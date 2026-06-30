import os
import json
import glob
import shutil
import torch
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VoiceProfileManager:
    """Manage voice profiles for reference audio and voice characteristics"""
    
    METADATA_FILE = "metadata.json"
    AUDIO_FILENAME = "ref_audio"
    LATENT_FILENAME = "voice_latent.pt"
    
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.profiles_dir / self.METADATA_FILE
        self._load_metadata()
        logger.info(f"Voice profile manager initialized at {self.profiles_dir}")

    def _load_metadata(self) -> None:
        """Load metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.debug(f"Loaded metadata with {len(self.metadata)} profiles")
            else:
                self.metadata = {}
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}, starting fresh")
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save metadata to file"""
        try:
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            raise

    def save_profile(
        self,
        name: str,
        audio_path: str,
        ref_text: str,
        voice_latent: Optional[torch.Tensor] = None
    ) -> str:
        """
        Save a voice profile
        
        Args:
            name: Profile name
            audio_path: Path to reference audio file
            ref_text: Reference text for the audio
            voice_latent: Optional pre-computed voice latent
            
        Returns:
            Profile name if successful
            
        Raises:
            ValueError: If inputs are invalid
            FileNotFoundError: If audio file not found
        """
        try:
            # Validate inputs
            if not name or not isinstance(name, str):
                raise ValueError("Profile name must be a non-empty string")
            if not ref_text or not isinstance(ref_text, str):
                raise ValueError("Reference text must be a non-empty string")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Create profile directory
            profile_path = self.profiles_dir / name
            profile_path.mkdir(exist_ok=True)
            
            # Copy audio file
            audio_ext = os.path.splitext(audio_path)[1]
            if not audio_ext:
                audio_ext = ".wav"
            
            target_audio_path = profile_path / f"{self.AUDIO_FILENAME}{audio_ext}"
            shutil.copy2(audio_path, target_audio_path)
            
            # Save voice latent if provided
            latent_path = None
            if voice_latent is not None:
                latent_path = profile_path / self.LATENT_FILENAME
                torch.save(voice_latent, latent_path)
            
            # Update metadata
            self.metadata[name] = {
                "audio_path": str(target_audio_path),
                "ref_text": ref_text,
                "latent_path": str(latent_path) if latent_path else None,
                "created_at": datetime.now().isoformat()
            }
            self._save_metadata()
            
            logger.info(f"Profile '{name}' saved successfully")
            return name
            
        except Exception as e:
            logger.error(f"Failed to save profile '{name}': {str(e)}")
            raise

    def get_profile(self, name: str) -> Optional[Dict]:
        """Get a profile by name"""
        if name not in self.metadata:
            logger.warning(f"Profile '{name}' not found")
            return None
        return self.metadata.get(name)

    def list_profiles(self) -> List[str]:
        """List all profile names"""
        return list(self.metadata.keys())

    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile
        
        Args:
            name: Profile name to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self.metadata:
                raise ValueError(f"Profile '{name}' does not exist")
            
            # Delete profile directory
            profile_path = self.profiles_dir / name
            if profile_path.exists():
                shutil.rmtree(profile_path)
            
            # Update metadata
            del self.metadata[name]
            self._save_metadata()
            
            logger.info(f"Profile '{name}' deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete profile '{name}': {str(e)}")
            return False

    def update_profile(
        self,
        name: str,
        ref_text: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> bool:
        """
        Update an existing profile
        
        Args:
            name: Profile name
            ref_text: New reference text (optional)
            audio_path: New audio file path (optional)
            
        Returns:
            True if successful
        """
        try:
            if name not in self.metadata:
                raise ValueError(f"Profile '{name}' does not exist")
            
            profile = self.metadata[name]
            profile_path = self.profiles_dir / name
            
            # Update audio if provided
            if audio_path:
                if not os.path.exists(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                
                # Remove old audio
                old_audio = glob.glob(str(profile_path / f"{self.AUDIO_FILENAME}.*"))
                for f in old_audio:
                    os.remove(f)
                
                # Copy new audio
                audio_ext = os.path.splitext(audio_path)[1]
                target_audio_path = profile_path / f"{self.AUDIO_FILENAME}{audio_ext}"
                shutil.copy2(audio_path, target_audio_path)
                profile["audio_path"] = str(target_audio_path)
            
            # Update text if provided
            if ref_text is not None:
                profile["ref_text"] = ref_text
            
            profile["updated_at"] = datetime.now().isoformat()
            self._save_metadata()
            
            logger.info(f"Profile '{name}' updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update profile '{name}': {str(e)}")
            return False
