import json
import logging
import dtlpy as dl

from pathlib import Path
from pdf_to_podcast.podcast_types import (
    Conversation,
)  # Podcast conversation data structures
from pdf_to_podcast.monologue_prompts import FinancialSummaryPrompts
from pdf_to_podcast.shared_functions import SharedServiceRunner

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class MonologueServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def generate_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate an outline from the pdf text summary

        Args:
            item (dl.Item): Dataloop item containing the podcast summary
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        # get the podcast metadata from the item
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        focus = podcast_metadata["focus"]
        pdf_name = podcast_metadata["pdf_name"]
        # get the summary from the last prompt annotation
        summary = SharedServiceRunner._get_summary_from_id(item.id)

        logger.info("Preparing to generate outline")

        # create summary file
        summary_filename = f"{Path(pdf_name).stem}_summary.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(summary)

        summary_item = item.dataset.items.upload(
            local_path=summary_filename,
            remote_name=summary_filename,
            remote_path=SharedServiceRunner._get_hidden_dir(item=item),
            overwrite=True,
            item_metadata={"user": item.metadata["user"]},
        )
        logger.info(f"Saved PDF summary as text item {summary_item.id}")

        # retreieve documents for context
        if podcast_metadata.get("with_references", None) is None:
            documents = [f"Document: {item.filename}\n{summary}"]
        else:
            documents = [f"Document: {item.filename}\n{summary}"]
            # TODO support multiple documents as context
            # # get all documents
            # reference_docs = MonologueServiceRunner._get_references(item)
            # documents = [f"Document: {item.filename}\n{summary}" for doc in reference_docs]

        template = FinancialSummaryPrompts.get_template(
            "monologue_multi_doc_synthesis_prompt"
        )
        llm_prompt = template.render(
            focus_instructions=focus if focus is not None else None,
            documents="\n\n".join(documents),
        )

        new_name = f"{Path(pdf_name).stem}_prompt2_summary_to_outline"
        new_metadata = item.metadata
        new_metadata["user"]["podcast"]["summary_item_id"] = summary_item.id

        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            item_metadata=new_metadata,
        )
        return new_item

    @staticmethod
    def generate_monologue(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate a monologue from the outline

        Args:
            item (dl.Item): Dataloop item containing the outline
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        # prep prompt item to get the outline
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        focus = podcast_metadata["focus"]
        pdf_name = podcast_metadata["pdf_name"]
        speaker_1_name = podcast_metadata["speaker_1_name"]
        # get the summary from the summary item
        summary = SharedServiceRunner._get_summary_from_id(
            podcast_metadata["summary_item_id"]
        )

        logger.info("Preparing to generate monologue")

        # get the outline from the last respoinse
        outline = SharedServiceRunner._get_last_message(item)
        if outline is None:
            raise ValueError("No outline found in the prompt item.")

        documents = [
            f"Document: {item.filename}\n{summary}"
        ]  # TODO support multiple documents as context

        template = FinancialSummaryPrompts.get_template("monologue_transcript_prompt")
        llm_prompt = template.render(
            raw_outline=outline,
            documents=documents,
            focus=(
                focus if focus else "key financial metrics and performance indicators"
            ),
            speaker_1_name=speaker_1_name,
        )

        # add the prompt to the prompt item
        new_name = f"{Path(pdf_name).stem}_prompt3_outline_to_monologue"
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            item_metadata=item.metadata,
        )
        return new_item

    @staticmethod
    def create_convo_json(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Create a final conversation from the monologue in JSON format

        Args:
            item (dl.Item): Dataloop item containing the monologue
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        pdf_name = podcast_metadata["pdf_name"]
        speaker_1_name = podcast_metadata["speaker_1_name"]

        # prep prompt item to get the monologue
        monologue = SharedServiceRunner._get_last_message(item)
        if monologue is None:
            raise ValueError(
                "No monologue found in the prompt item. Try running the previous step again."
            )

        logger.info("Preparing to generate final monologue transcript JSON")

        # create the final conversation in JSON format
        schema = Conversation.model_json_schema()
        template = FinancialSummaryPrompts.get_template("monologue_dialogue_prompt")
        llm_prompt = template.render(
            speaker_1_name=speaker_1_name,
            text=monologue,
            schema=json.dumps(schema, indent=2),
        )

        new_name = f"{Path(pdf_name).stem}_prompt4_monologue_to_convo_json"
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            item_metadata=item.metadata,
        )
        return new_item
