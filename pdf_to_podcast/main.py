import dotenv
import logging
import dtlpy as dl

from typing import Optional

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
            item: dl.Item,
            progress: dl.Progress,
            context: dl.Context,
            monologue: bool,
            focus: Optional[str] = None,
            with_references: Optional[bool] = False,
            duration: Optional[int] = None,
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
        return SharedServiceRunner.prepare_and_summarize_pdf(
            item, progress, context, monologue, focus, with_references, duration
        )

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
    def dialogue_process_segments(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return DialogueServiceRunner.process_segments(item, progress, context)

    @staticmethod
    def dialogue_generate_dialogue(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return DialogueServiceRunner.generate_dialogue(item, progress, context)

    @staticmethod
    def dialogue_combine_dialogues(item: dl.Item, model: dl.Model, progress: dl.Progress, context: dl.Context):
        return DialogueServiceRunner.combine_dialogues(item, model, progress, context)

    # @staticmethod
    # def dialogue_check_dialogue(items: List[dl.Item], progress: dl.Progress, context: dl.Context):
    #     return DialogueServiceRunner.check_dialogue(items, progress, context)

    @staticmethod
    def dialogue_create_convo_json(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return DialogueServiceRunner.create_convo_json(item, progress, context)

    @staticmethod
    def create_final_json(item: dl.Item, progress: dl.Progress, context: dl.Context):
        return SharedServiceRunner.create_final_json(item, progress, context)


if __name__ == "__main__":
    env = "prod"
    dl.setenv(env)

    monologue = False
    focus = None
    with_references = False
    duration = None

    item_id = "67d05c15c3d298140be63374"

    progress = dl.Progress()
    context = dl.Context()

    model_dialogue = dl.models.get(model_id="67ed3672f41fe3426dd2c3e0") # 405b reasoning

    # item should be a pdf file
    # item = dl.items.get(item_id=item_id)
    # # go through each function and test whether it works with a real item
    # processed_item = ServiceRunner.prepare_and_summarize_pdf(
    #     item, progress, context, monologue, focus, with_references, duration
    # )
    # print(f"1: Successfully processed item: {processed_item.name} ({processed_item.id})")
    # print(f"Link here: {processed_item.platform_url}")
    # # processed_item = dl.items.get(item_id="67f7bd4a17f2118afa100545") # TODO DEBUG DELETE

    # # wait for llama prediction on UI...
    # input("Please get llama reasoning prediction via UI. Once it's finished, press Enter to continue...")
    # # outline = dl.items.get(item_id="67f765bb9d79af79958872d9") # TODO DEBUG DELETE

    if monologue is True:
        outline = ServiceRunner.monologue_generate_outline(processed_item, progress, context)
        print(f"2/5: Successfully prepared outline: {outline.name} ({outline.id})")
        print(f"Link here: {outline.platform_url}")

        # wait for llama prediction on UI...
        input("Please get llama reasoning prediction via UI. Once it's finished, press Enter to continue...")
        # outline = dl.items.get(item_id="67f534d829b36bf6a83e1195") # TODO DEBUG DELETE

        # Generate monologue
        monologue = ServiceRunner.monologue_generate_monologue(outline, progress, context)
        print(f"3/5: Successfully prepared monologue: {monologue.name} ({monologue.id})")
        print(f"Link here: {monologue.platform_url}")

        # wait for llama prediction on UI...
        input("Please get llama reasoning prediction via UI. Once it's finished, press Enter to continue...")
        # monologue = dl.items.get(item_id="67f514a74ccc17715c9e81ad") # TODO DEBUG DELETE

        # Create convo json
        convo_json = ServiceRunner.monologue_create_convo_json(monologue, progress, context)
        print(f"4/5: Successfully prepared final conversation: {convo_json.name} ({convo_json.id})")
        print(f"Link here: {convo_json.platform_url}")


    else:
        # # Generate raw outline
        # outline = ServiceRunner.dialogue_generate_raw_outline(processed_item, progress, context)
        # print(f"2/9: Successfully prepared raw outline: {outline.name} ({outline.id})")
        # print(f"Link here: {outline.platform_url}")

        # # wait for llama prediction on UI...
        # input("Please get llama reasoning prediction via UI. Once it's finished, press Enter to continue...")

        # # Generate structured outline
        # structured_outline = ServiceRunner.dialogue_generate_structured_outline(outline, progress, context)
        # print(f"3/9: Successfully prepared structured outline: {structured_outline.name} ({structured_outline.id})")
        # print(f"Link here: {structured_outline.platform_url}")

        # # wait for llama prediction on UI...
        # input("Please get llama json podcast prediction via UI. Once it's finished, press Enter to continue...")
        structured_outline = dl.items.get(item_id="67fcc603489a0facf79f7f9e") # TODO DEBUG DELETE

        # # Process segments
        # segments = ServiceRunner.dialogue_process_segments(structured_outline, progress, context)
        # print(f"4/9: Successfully processed segments: [{', '.join([segment.name for segment in segments])}]")
        # print(f"Link here: {segment.platform_url for segment in segments}")

        # # wait for llama prediction on UI...
        # input("Please get llama iteration prediction via UI. Once it's finished, press Enter to continue...")
        # # get segment items
        # filters = dl.Filters()
        # filters.add(field="dir", values=f"/segments/{structured_outline.metadata['user']['podcast']['pdf_name']}")
        # filters.sort_by(field="filename")
        # segments = structured_outline.dataset.items.list(filters=filters)
        # segment_items = list(segments.all())

        # # Generate dialogue
        # for segment in segments:
        #     dialogue = ServiceRunner.dialogue_generate_dialogue(segment, progress, context)
        #     print(f"5/9: Successfully prepared dialogue: {dialogue.name} ({dialogue.id})")
        #     print(f"Link here: {dialogue.platform_url}")

        # # wait for llama prediction on UI...
        # input("Please get llama reasoning prediction via UI. Once it's finished, press Enter to continue...")

        # Iteratively combine
        combined_dialogue = ServiceRunner.dialogue_combine_dialogues(structured_outline, model_dialogue, progress, context)
            
        print(f"6/9: Successfully combined dialogues: {combined_dialogue.name} ({combined_dialogue.id})")
        print(f"Link here: {combined_dialogue.platform_url}")
        # combined_dialogue = dl.items.get(item_id="67fe03596d3d49b4dacb2569") # TODO DEBUG DELETE

        # Create convo json
        convo_json = ServiceRunner.dialogue_create_convo_json(combined_dialogue, progress, context)
        print(f"7/9: Successfully prepared convo json: {convo_json.name} ({convo_json.id})")
        print(f"Link here: {convo_json.platform_url}")

    # wait for llama prediction on UI...
    input("Please get llama json convo prediction via UI. Once it's finished, press Enter to continue...")
    # convo_json = dl.items.get(item_id="67f7ce449d79afd4298926de") # TODO DEBUG DELETE
    # Generate audio
    try:
        # Create final conversation
        final_transcript = ServiceRunner.create_final_json(convo_json, progress, context)
        print(f"end-1: Successfully prepared final conversation: {final_transcript.name} ({final_transcript.id})")
        print(f"Link here: {final_transcript.platform_url}")

        output_podcast = SharedServiceRunner.generate_audio(final_transcript, progress, context)
        print(f"End: Successfully prepared audio file: {output_podcast.name} ({output_podcast.id})")
        print(f"Link here: {output_podcast.platform_url}")
    except Exception as e:
        print(f"Error generating audio: {e}")
        raise
