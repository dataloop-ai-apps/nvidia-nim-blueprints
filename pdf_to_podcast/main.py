import dotenv
import logging
import dtlpy as dl

from pydantic import BaseModel
from typing import Optional
from elevenlabs.client import ElevenLabs

from pdf_to_podcast.shared_functions import SharedServiceRunner
from pdf_to_podcast.monologue_flow import MonologueServiceRunner
from pdf_to_podcast.dialogue_flow import DialogueServiceRunner

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class ServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def prepare_and_summarize_pdf(
        item: dl.Item, monologue: bool, progress: dl.Progress, context: dl.Context, focus: str = None, duration: int = 10
    ):
        """
        Prepare the PDF text for the summary

        Args:
            item (dl.Item): Dataloop item containing the original PDF file
            monologue (bool): Whether to generate a monologue or a podcast
            progress (dl.Progress): Progress object to update the user
            context (dl.Context): Context object to access the item
            focus (str): Focus of the summary
            duration (int): Duration of the summary

        Returns:
            str: Prompt for the summary
        """
        return SharedServiceRunner.prepare_and_summarize_pdf(item, monologue, progress, context, focus, duration)

    @staticmethod
    def generate_audio(item: dl.Item, voice_mapping: dict = None, output_file: str = None):
        return SharedServiceRunner.generate_audio(item, voice_mapping, output_file)

    # Monologue flow methods
    @staticmethod
    def monologue_generate_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return MonologueServiceRunner.generate_outline(item, progress, context)

    @staticmethod
    def monologue_generate_monologue(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return MonologueServiceRunner.generate_monologue(item, progress, context)

    @staticmethod
    def monologue_create_convo_json(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return MonologueServiceRunner.create_convo_json(item, progress, context)

    # Dialogue flow methods
    @staticmethod
    def dialogue_generate_raw_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return DialogueServiceRunner.generate_raw_outline(item, progress, context)

    @staticmethod
    def dialogue_generate_structured_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return DialogueServiceRunner.generate_structured_outline(item, progress, context)

    @staticmethod
    def dialogue_process_segments(item: dl.Item, outline, progress: dl.Progress, context: dl.Context):
        return DialogueServiceRunner.process_segments(item, outline, progress, context)

    @staticmethod
    def dialogue_generate_dialogue(segments, outline):
        return DialogueServiceRunner.generate_dialogue(segments, outline)

    @staticmethod
    def dialogue_combine_dialogues(segment_dialogues, outline):
        return DialogueServiceRunner.combine_dialogues(segment_dialogues, outline)

    @staticmethod
    def dialogue_create_convo_json(item: dl.Item, dialogue: str):
        return DialogueServiceRunner.create_convo_json(item, dialogue)

    @staticmethod
    def dialogue_create_final_conversation(dir_item: dl.Item, dialogue: str):
        return DialogueServiceRunner.create_final_conversation(dir_item, dialogue)


if __name__ == "__main__":
    dl.setenv("prod")
    item = dl.items.get(item_id="6758139d45821d442ba1f6e1")
    progress = dl.Progress()
    context = dl.Context()

    processed_item = ServiceRunner.prepare_and_summarize_pdf(item, True, progress, context)

    try:
        output_file = ServiceRunner.generate_audio(processed_item)
        logger.info(f"Successfully generated audio file: {output_file}")
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise
