import json
import logging
import os
import tempfile
import dtlpy as dl

from pathlib import Path
from pdf_to_podcast.podcast_types import Conversation, PodcastMetadata
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
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        podcast_metadata.validate_stage("summary")
        summary = SharedServiceRunner._get_last_message(item)

        logger.info("Preparing to generate outline")

        # save summary to temp file (item ID avoids collisions)
        remote_summary_name = f"{Path(podcast_metadata.pdf_name).stem}_summary.txt"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{item.id}_summary.txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(summary)
            local_summary_path = f.name
        try:
            summary_item = item.dataset.items.upload(
                local_path=local_summary_path,
                remote_name=remote_summary_name,
                remote_path=SharedServiceRunner._get_hidden_dir(item=item),
                overwrite=True,
                item_metadata={"user": item.metadata["user"]},
            )
        finally:
            os.remove(local_summary_path)
        logger.info(f"Saved PDF summary as text item {summary_item.id}")

        documents = [f"Document: {item.filename}\n{summary}"]

        template = FinancialSummaryPrompts.get_template(
            "monologue_multi_doc_synthesis_prompt"
        )
        llm_prompt = template.render(
            focus_instructions=podcast_metadata.focus,
            documents="\n\n".join(documents),
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt2_summary_to_outline"
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "outline",
            "summary_item_id": summary_item.id,
        })
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            item_metadata=new_meta.to_item_metadata(),
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
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        podcast_metadata.validate_stage("outline")
        summary = SharedServiceRunner._get_summary_from_id(
            podcast_metadata.summary_item_id
        )

        logger.info("Preparing to generate monologue")

        outline = SharedServiceRunner._get_last_message(item)
        if outline is None:
            raise ValueError("No outline found in the prompt item.")

        documents = [f"Document: {item.filename}\n{summary}"]

        template = FinancialSummaryPrompts.get_template("monologue_transcript_prompt")
        llm_prompt = template.render(
            raw_outline=outline,
            documents=documents,
            focus=(
                podcast_metadata.focus
                if podcast_metadata.focus
                else "key financial metrics and performance indicators"
            ),
            speaker_1_name=podcast_metadata.speaker_1_name,
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt3_outline_to_monologue"
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "monologue",
        })
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            item_metadata=new_meta.to_item_metadata(),
        )
        return new_item

    @staticmethod
    def create_convo_json(
        item: dl.Item, model: dl.Model, progress: dl.Progress, context: dl.Context
    ):
        """
        Create a final conversation from the monologue in JSON format

        Args:
            item (dl.Item): Dataloop item containing the monologue
            model (dl.Model): Dataloop model entity for setting max_tokens
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        podcast_metadata.validate_stage("monologue")

        monologue = SharedServiceRunner._get_last_message(item)
        if monologue is None:
            raise ValueError(
                "No monologue found in the prompt item. Try running the previous step again."
            )

        logger.info("Preparing to generate final monologue transcript JSON")

        schema = Conversation.model_json_schema()
        SharedServiceRunner._set_model_configuration(model, max_tokens=2048)

        template = FinancialSummaryPrompts.get_template("monologue_dialogue_prompt")
        llm_prompt = template.render(
            speaker_1_name=podcast_metadata.speaker_1_name,
            text=monologue,
            schema=json.dumps(schema, indent=2),
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt4_monologue_to_convo_json"
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "convo_json",
        })
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            item_metadata=new_meta.to_item_metadata(),
        )
        return new_item
