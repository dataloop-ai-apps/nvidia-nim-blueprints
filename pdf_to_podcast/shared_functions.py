import os
import json
import dotenv
import logging
import dtlpy as dl

from pydantic import BaseModel
from typing import Optional
from elevenlabs.client import ElevenLabs

from monologue_prompts import FinancialSummaryPrompts
from podcast_prompts import PodcastPrompts

# Load environment variables from .env file
dotenv.load_dotenv('.env')

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")

# Default voices configuration
DEFAULT_VOICE_1 = os.getenv("DEFAULT_VOICE_1", "EXAVITQu4vr4xnSDxMaL")
DEFAULT_VOICE_2 = os.getenv("DEFAULT_VOICE_2", "bIHbv24MWmeRgasZH58o")
DEFAULT_VOICE_MAPPING = {"speaker-1": DEFAULT_VOICE_1, "speaker-2": DEFAULT_VOICE_2}
DEFAULT_SPEAKER_1_NAME = os.getenv("DEFAULT_SPEAKER_1_NAME", "Alice")
DEFAULT_SPEAKER_2_NAME = os.getenv("DEFAULT_SPEAKER_2_NAME", "Will")


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


class SharedServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def prepare_and_summarize_pdf(
        item: dl.Item,
        monologue: bool,
        progress: dl.Progress,
        context: dl.Context,
        focus: str = None,
        duration: int = 10,
    ):
        buffer = item.download(save_locally=False)
        pdf_text = buffer.read().decode('utf-8')

        if monologue is True:
            template = FinancialSummaryPrompts.get_template("monologue_summary_prompt")
        else:
            template = PodcastPrompts.get_template("podcast_summary_prompt")
        llm_prompt = template.render(text=pdf_text)

        new_name = f"{item.filename}_prompt"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]},  # role default is user
            prompt_key='1',
        )

        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
        new_item.metadata.get("user", {}).update(
            {
                "podcast": {
                    "pdf_id": item.metadata.user["original_item_id"],
                    "focus": focus,
                    "monologue": monologue,
                    "duration": duration,
                }
            }
        )
        new_item.update()

        logger.info(f"Successfully created prompt item for {item.filename} ID {item.id}")

        actions = ['monologue', 'dialogue']
        if monologue is True:
            progress.update(action=actions[0])
        else:
            progress.update(action=actions[1])

        return new_item

    @staticmethod
    def generate_audio(item: dl.Item, voice_mapping: dict = None, output_file: str = None):
        """
        Generate audio from the conversation JSON file

        Args:
            item (dl.Item): Dataloop item containing the conversation JSON
            output_file (str): Output MP3 file path (defaults to output/output.mp3)
            voice_mapping (dict): Optional mapping of speakers to voice IDs

        Returns:
            str: Path to the generated audio file
        """
        if output_file is None:
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            output_file = os.path.join("output", "output.mp3")

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        converter = TTSConverter(api_key=api_key)

        # Download and process the conversation JSON
        buffer = item.download(save_locally=True)
        try:
            conversation_json = json.loads(buffer.read().decode('utf-8'))

            # Create a temporary JSON file for the converter
            temp_file = os.path.join(os.path.dirname(__file__), "temp_conversation.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_json, f)

            # Process the file and generate audio
            converter.process_file(temp_file, output_file, voice_mapping)

            # Clean up temporary file
            os.remove(temp_file)

        except Exception as e:
            logger.error(f"Error processing conversation JSON: {e}")
            raise

        mp3_item = item.dataset.items.upload(output_file, remote_name=output_file, remote_path=item.dir, overwrite=True)
        logger.info(f"Successfully uploaded audio file: {mp3_item.id}")
        return mp3_item

    @staticmethod
    def _unescape_unicode_string(s: str) -> str:
        """
        Convert escaped Unicode sequences to actual Unicode characters.

        Args:
            s (str): String potentially containing escaped Unicode sequences

        Returns:
            str: String with Unicode sequences converted to actual characters

        Example:
            >>> unescape_unicode_string("Hello\\u2019s World")
            "Hello's World"
        """
        # This handles both raw strings (with extra backslashes) and regular strings
        return s.encode("utf-8").decode("unicode-escape")
