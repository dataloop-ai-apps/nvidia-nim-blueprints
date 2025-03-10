import os
import logging

from typing import List, Dict
from elevenlabs.client import ElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default voices
DEFAULT_VOICE_1 = os.getenv("DEFAULT_VOICE_1", "iP95p4xoKVk53GoZ742B")
DEFAULT_VOICE_2 = os.getenv("DEFAULT_VOICE_2", "9BWtsMINqrJLrRacOk9x")


class SimpleTTSService:
    def __init__(self):
        self.eleven_labs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"), timeout=120)

    def convert_text_to_speech(self, text: str, voice_id: str = DEFAULT_VOICE_1) -> bytes:
        """Convert text to speech using ElevenLabs API"""
        try:
            audio_stream = self.eleven_labs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128",
                voice_settings={"stability": 0.5, "similarity_boost": 0.75, "style": 0.0},
            )
            return b"".join(chunk for chunk in audio_stream)
        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            raise

    def get_available_voices(self) -> List[Dict]:
        """Fetch available voices from ElevenLabs API"""
        try:
            response = self.eleven_labs_client.voices.get_all()
            return [
                {
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "description": voice.description if hasattr(voice, "description") else None,
                }
                for voice in response.voices
            ]
        except Exception as e:
            logger.error(f"Error fetching voices: {e}")
            return []


def main():
    # Example usage
    tts_service = SimpleTTSService()

    # Get available voices
    voices = tts_service.get_available_voices()
    print("Available voices:")
    for voice in voices:
        print(f"- {voice['name']} (ID: {voice['voice_id']})")

    # Example text to convert
    text = "Hello! This is a test of the ElevenLabs text-to-speech API."

    # Convert text to speech
    try:
        audio_data = tts_service.convert_text_to_speech(text)

        # Save the audio to a file
        with open("output.mp3", "wb") as f:
            f.write(audio_data)
        print("\nAudio saved to output.mp3")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
