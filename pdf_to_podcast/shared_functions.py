import os
import json
import dotenv
import logging
import dtlpy as dl

from io import BytesIO
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict
from elevenlabs.client import ElevenLabs

from pdf_to_podcast.monologue_prompts import FinancialSummaryPrompts
from pdf_to_podcast.podcast_prompts import PodcastPrompts
from pdf_to_podcast.podcast_types import PodcastOutline

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
    def _collect_text_items(item: dl.Item) -> str:   
        """
        Collect the text from the PDF file

        Child items are found in the dataset by searching for the item id in the metadata

        Args:
            item (dl.Item): The oroginal PDF file to be processed

        Returns:
            str: The text from the PDF file
        """
        filters = dl.Filters()
        filters.add(field="metadata.user.original_item_id", values=item.id)
        filters.sort_by(field="name", value=dl.FiltersOrderByDirection.ASCENDING)
        items = item.dataset.items.list(filters=filters).all()
        
        pdf_text = ""
        for child_item in items:
            if "text" in child_item.mimetype:
                buffer = child_item.download(save_locally=False)
                pdf_text += buffer.read().decode('utf-8')
        return pdf_text

    @staticmethod
    def prepare_and_summarize_pdf(
        item: dl.Item,
        progress: dl.Progress,
        context: dl.Context,
        monologue: bool,
        focus: str = None,
        with_references: bool = False,
        duration: int = None,
    ):
        """
        Prepare the PDF file into a prompt item with the text to be processed

        Args:
            item (dl.Item): The PDF file to be processed
            monologue (bool): Whether to generate a monologue or a podcast
            progress (dl.Progress): The progress object to update the user
            context (dl.Context): The context object to access the item
            focus (str): The focus of the summary
            with_references (bool): Whether to include references in the summary
            duration (int): The duration of a dialogue podcast

        Returns:
            dl.Item: The prompt item with the text to be processed
        """

        pdf_text = SharedServiceRunner._collect_text_items(item)
        # upload text to dataloop item
        text_filename = Path(item.name).stem + "_text.txt"
        with open(text_filename, "w", encoding='utf-8') as f: 
            f.write(pdf_text)
        text_item = item.dataset.items.upload(local_path=text_filename, 
                                              remote_name=text_filename, 
                                              remote_path=item.dir, # same dir as pdf 
                                              overwrite=True, 
                                              item_metadata={"user": {"podcast": {"original_item_name": item.name}}})

        if monologue is True:
            template = FinancialSummaryPrompts.get_template("monologue_summary_prompt")
        else:
            template = PodcastPrompts.get_template("podcast_summary_prompt")
        llm_prompt = template.render(text=pdf_text)

        new_name = f"{Path(item.name).stem}_prompt1_summary.json"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )

        new_item_metadata = item.metadata.get("user", {})
        new_item_metadata.update(
            {
                "podcast": {
                    "pdf_id": item.id,
                    "pdf_name": item.name,
                    "focus": focus,
                    "monologue": monologue,
                    "with_references": with_references,
                }
            }
        )
        if duration is not None:
            new_item_metadata['podcast']['duration'] = duration
        new_item = item.dataset.items.upload(prompt_item, 
                                             remote_name=new_name, 
                                             remote_path=item.dir, 
                                             overwrite=True, 
                                             item_metadata={"user": new_item_metadata})
        
        logger.info(f"Successfully created prompt item for {item.name} in new item {new_item.id}")

        actions = ['monologue', 'dialogue']
        progress.update(action=actions[0] if monologue is True else actions[1])

        return new_item

    @staticmethod
    def create_final_json(item: dl.Item, progress: dl.Progress, context: dl.Context) -> dl.Item:
        """
        Check the final conversation JSON and make sure all strings are unescaped

        Args:
            item (dl.Item): Dataloop item containing the dialogue

        Returns:
            new_item (dl.Item): Dataloop item containing the structured conversation JSON
        """
        logger.info("Formatting final conversation")

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        pdf_name = podcast_metadata.get("pdf_name", None)

        conversation_json_str = SharedServiceRunner._get_last_message(item=item)
        if conversation_json_str is None:
            raise ValueError(f"No conversation JSON found in the conversation segment item {item.id}.")

        # Convert string to JSON if needed
        if isinstance(conversation_json_str, str):
            conversation_json = json.loads(conversation_json_str)
        else:
            conversation_json = conversation_json_str

        # Ensure all strings are unescaped
        if "dialogue" in conversation_json:
            for entry in conversation_json["dialogue"]:
                if "text" in entry:
                    entry["text"] = SharedServiceRunner._unescape_unicode_string(entry["text"])
                    
        # upload the final conversation
        new_name = f"{Path(pdf_name).stem}_final_transcript.json"
        json_path = Path.cwd() / new_name
        with open(json_path, "w", encoding='utf-8') as f:
            json_file = json.dumps(conversation_json, indent=2)
            f.write(json_file)

        new_item = item.dataset.items.upload(
            local_path=str(json_path),
            remote_name=new_name,
            remote_path=item.dir,
            overwrite=True,
        )
        return new_item


    @staticmethod
    def generate_audio(item: dl.Item, progress, context, voice_mapping: dict = None, output_file: str = None):
        """
        Generate audio from the conversation JSON file

        Args:
            item (dl.Item): Dataloop JSON item containing the conversation 

        Returns:
            str: Path to the generated audio file
        """
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        pdf_name = podcast_metadata.get("pdf_name", None)
        monologue = podcast_metadata.get("monologue", None)

        if monologue is True:
            output_file_name = Path(pdf_name).stem + "_monologue.mp3"
        else:
            output_file_name = Path(pdf_name).stem + "_podcast.mp3"

        if output_file is None:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, output_file_name)

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        converter = TTSConverter(api_key=api_key)

        # Download and process the conversation JSON from the last response
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        conversation_json = last_message.get("content", [])[0].get("text", None)
        if conversation_json is None:
            raise ValueError("No conversation JSON found in the prompt item.")

        try:
            conversation_json = json.loads(conversation_json)

            # Create a temporary JSON file for the converter
            temp_file = os.path.join(os.path.dirname(__file__), "temp_conversation.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_json, f)

            # Process the file and generate audio
            converter.process_file(temp_file, output_file, voice_mapping)

            # Clean up temporary file
            os.remove(temp_file)

        except Exception as e:
            logger.error(f"Error processing conversation JSON: {e} from item {item.id}")
            raise

        mp3_item = item.dataset.items.upload(output_file, 
                                             remote_name=output_file_name, 
                                             remote_path=item.dir, 
                                             overwrite=True, 
                                             item_metadata=item.metadata)
        logger.info(f"Successfully uploaded audio file: {mp3_item.id}")
        return mp3_item

    @staticmethod
    def _get_last_message(item: dl.Item) -> str:
        """
        Get the last message from the item.

        Args:
            item (dl.Item): Dataloop item containing the last message

        Returns:
            str: The last message from the item

        Converts the item to a prompt item and gets the last message.
        """
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        try:
            last_message = messages[-1]
            text = last_message.get("content", [])[0].get("text", None)
        except Exception as e:
            logger.error(f"Error getting last message from item {item.id}: {e}")
            text = None
        return text

    @staticmethod
    def _get_summary_text(summary_item_id: str) -> str:
        """
        Get the summary text from the summary item.
        """
        summary_item = dl.items.get(item_id=summary_item_id)
        if summary_item is None:
            raise ValueError(f"Summary item not found for id: {summary_item_id}")
        if "text" not in summary_item.mimetype:
            raise ValueError(f"Summary item is not a text file for id: {summary_item_id}")
        text = summary_item.download(save_locally=False).decode('utf-8')
        return text

    @staticmethod
    def _get_outline_dict(outline_item: dl.Item) -> PodcastOutline:
        """
        Get the PodcastOutline from an outline item.
        """
        prompt_outline_item = dl.PromptItem.from_item(outline_item)
        messages = prompt_outline_item.to_messages()
        last_message = messages[-1]
        outline_dict = last_message.get("content", [])[0].get("text", None)
        if outline_dict is None:
            raise ValueError(f"No outline found in item {outline_item.id} metadata.")
        if isinstance(outline_dict, str):
            outline_dict = json.loads(outline_dict)
        outline = PodcastOutline.model_validate_json(outline_dict)
        return outline


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
