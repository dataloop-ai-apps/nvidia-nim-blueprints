from elevenlabs.client import ElevenLabs
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default voices configuration
DEFAULT_VOICE_1 = os.getenv("DEFAULT_VOICE_1", "EXAVITQu4vr4xnSDxMaL")
DEFAULT_VOICE_2 = os.getenv("DEFAULT_VOICE_2", "bIHbv24MWmeRgasZH58o")
DEFAULT_VOICE_MAPPING = {"speaker-1": DEFAULT_VOICE_1, "speaker-2": DEFAULT_VOICE_2}


class DialogueEntry(BaseModel):
    text: str
    speaker: Optional[str] = "speaker-1"
    voice_id: Optional[str] = None


class TTSConverter:
    def __init__(self, api_key: str = None):
        """Initialize the TTS converter with ElevenLabs API key"""
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.client = ElevenLabs(api_key=self.api_key)

    def _convert_text(self, text: str, voice_id: str) -> bytes:
        """Convert a single piece of text to speech"""
        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128",
                voice_settings={"stability": 0.5, "similarity_boost": 0.75},
            )
            return b"".join(chunk for chunk in audio_stream)
        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            raise

    def process_file(self, input_file: str, output_file: str, voice_mapping: dict = None):
        """Process a JSON file containing dialogue or monologue"""
        # Use default voice mapping if none provided
        voice_mapping = voice_mapping or DEFAULT_VOICE_MAPPING

        try:
            # Read and parse the input JSON file
            with open(os.path.join(os.path.dirname(__file__), input_file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract dialogue entries
            dialogue = data.get('dialogue', [])
            if not dialogue:
                # If no dialogue found, treat as monologue
                dialogue = [{"text": data.get("text", ""), "speaker": "speaker-1"}]

            # Convert all entries to audio and combine
            combined_audio = b""
            total_entries = len(dialogue)

            for i, entry in enumerate(dialogue, 1):
                # Create DialogueEntry instance for validation
                entry_model = DialogueEntry(**entry)

                # Determine which voice to use
                voice_id = (
                    entry_model.voice_id
                    if entry_model.voice_id
                    else voice_mapping.get(entry_model.speaker, DEFAULT_VOICE_1)
                )

                logger.info(f"Processing entry {i}/{total_entries}")
                audio_chunk = self._convert_text(entry_model.text, voice_id)
                combined_audio += audio_chunk

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save the combined audio to file
            with open(output_file, 'wb') as f:
                f.write(combined_audio)

            logger.info(f"Successfully created audio file: {output_file}")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise


def main(input_file: str, output_file: str = None):
    """
    Convert text to speech using ElevenLabs

    Args:
        input_file (str): Input JSON file containing dialogue or monologue
        output_file (str): Output MP3 file path (defaults to output/output.mp3)
    """
    if output_file is None:
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        output_file = os.path.join("output", "output.mp3")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    converter = TTSConverter(api_key=api_key)
    converter.process_file(input_file, output_file)


if __name__ == "__main__":
    # Example usage
    main("sample.json")
