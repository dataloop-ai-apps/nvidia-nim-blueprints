import json
import logging
import tempfile
import os
import dtlpy as dl

from pathlib import Path
from typing import List, Optional
from pdf_to_podcast.shared_functions import (
    SharedServiceRunner,
    SEGMENT_DIR_PROMPT4,
    SEGMENT_DIR_PROMPT5,
)
from pdf_to_podcast.podcast_prompts import PodcastPrompts
from pdf_to_podcast.podcast_types import (
    PodcastOutline,
    Conversation,
    PodcastSegment,
    PodcastMetadata,
)

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class DialogueServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def generate_raw_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate initial raw outline from summarized PDFs.

        Args:
            item (dl.Item): the original PDF item
            progress (dl.Progress): Dataloop progress object from pipelines
            context (dl.Context): Dataloop context object from pipelines

        Returns:
            dl.Item: Dataloop prompt item containing the initial raw outline in the hidden directory
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        podcast_metadata.validate_stage("summary")
        working_dir = SharedServiceRunner._get_hidden_dir(item=item)

        summary = SharedServiceRunner._get_last_message(item)
        if summary is None:
            raise ValueError(
                "No text summary found in the prompt item. Try running the previous step again."
            )

        logger.info("Preparing to generate initial outline")

        # save and upload summary file (temp file with item ID to avoid collisions)
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
                remote_path=working_dir,
                overwrite=True,
                item_metadata={"user": item.metadata["user"]},
            )
        finally:
            os.remove(local_summary_path)

        logger.info(f"Saved PDF summary as text item {summary_item.id}")

        documents = [f"Document: {podcast_metadata.pdf_name}\n{summary}"]

        template = PodcastPrompts.get_template("podcast_multi_pdf_outline_prompt")
        llm_prompt = template.render(
            total_duration=podcast_metadata.duration,
            focus_instructions=podcast_metadata.focus,
            documents="\n\n".join(documents),
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt2_raw_outline"
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "raw_outline",
            "summary_item_id": summary_item.id,
        })
        extra_user = {}
        original_item_id = item.metadata.get("user", {}).get("original_item_id")
        if original_item_id:
            extra_user["original_item_id"] = original_item_id

        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=working_dir,
            overwrite=True,
            item_metadata=new_meta.to_item_metadata(**extra_user),
        )
        return new_item

    @staticmethod
    def generate_structured_outline(
        item: dl.Item, model: dl.Model, progress: dl.Progress, context: dl.Context
    ):
        """
        Convert raw outline text to structured PodcastOutline format.

        Uses JSON schema validation to ensure the outline follows the required structure
        and only references valid PDF filenames.

        Args:
            item (dl.Item): Dataloop item containing the raw outline in the hidden directory
            model (dl.Model): Dataloop model entity for setting max_tokens
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): Item for prompting to generate structured outline following the PodcastOutline schema

        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        podcast_metadata.validate_stage("raw_outline")

        raw_outline = SharedServiceRunner._get_last_message(item)
        if raw_outline is None:
            raise ValueError(f"No outline found in item {item.id}.")

        logger.info("Preparing to generate structured outline")

        valid_filenames = [podcast_metadata.pdf_name]
        if podcast_metadata.references is not None:
            valid_filenames.extend(podcast_metadata.references)
        schema = PodcastOutline.model_json_schema()
        schema["$defs"]["PodcastSegment"]["properties"]["references"]["items"] = {
            "type": "string",
            "enum": valid_filenames,
        }

        SharedServiceRunner._set_model_configuration(model, max_tokens=2048)

        template = PodcastPrompts.get_template(
            "podcast_multi_pdf_structured_outline_prompt"
        )
        llm_prompt = template.render(
            outline=raw_outline,
            schema=json.dumps(schema, indent=2),
            valid_filenames=valid_filenames,
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt3_structured_outline"
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "structured_outline",
        })
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=item.dir,
            overwrite=True,
            item_metadata=new_meta.to_item_metadata(),
        )
        return new_item

    @staticmethod
    def _process_segment(
        item: dl.Item,
        segment: PodcastSegment,
        idx: int,
        total_segments: int,
        summary: str,
    ) -> dl.Item:
        """
        Process a single outline segment to generate initial content.

        The parent item is the structured outline item.

        Args:
            item (dl.Item): Dataloop item containing the outline segment
            segment (PodcastOutline.Segment): Segment from the outline to process
            idx (int): Index of the segment
            total_segments (int): Total number of segments
            summary (str): Summary of the podcast

        Returns:
            dl.Item: Dataloop item containing the generated content
        """
        logger.info(
            f"Preparing to generate initial content for segment {idx + 1}/{total_segments}"
        )

        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)

        text_content = f"Document: {podcast_metadata.pdf_name}\n{summary}"

        template = PodcastPrompts.get_template("podcast_prompt_with_references")
        llm_prompt = template.render(
            duration=segment.duration,
            topic=segment.section,
            angles="\n".join([topic.title for topic in segment.topics]),
            text=text_content,
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt4_segment_{idx:02d}"
        new_dir = SharedServiceRunner._get_segment_dir(
            item=item, pdf_name=podcast_metadata.pdf_name, stage=SEGMENT_DIR_PROMPT4
        )
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "segment",
            "outline_item_id": item.id,
            "segment_topic": segment.section,
            "segment_idx": idx,
            "total_segments": total_segments,
            "topics": [topic.title for topic in segment.topics],
        })

        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=new_dir,
            item_metadata=new_meta.to_item_metadata(parentItemId=item.id),
        )
        return new_item

    @staticmethod
    def process_segments(
        item: dl.Item, progress: dl.Progress, context: dl.Context
    ) -> List[dl.Item]:
        """
        Process all outline segments in parallel to generate initial content.

        Args:
            item (dl.Item): Dataloop item containing the outline
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            List[dl.Item]: List of segment items for compatibility with the workflow
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        podcast_metadata.validate_stage("structured_outline")
        summary = SharedServiceRunner._get_summary_from_id(
            podcast_metadata.summary_item_id
        )

        outline = SharedServiceRunner._get_outline_dict(outline_item=item)
        total_segments = len(outline.segments)
        logger.info(f"Preparing to process {total_segments} segments")

        # Clean up any stale segment items from previous runs
        for stage in [SEGMENT_DIR_PROMPT4, SEGMENT_DIR_PROMPT5]:
            cleanup_dir = SharedServiceRunner._get_segment_dir(
                item=item, pdf_name=podcast_metadata.pdf_name, stage=stage
            )
            cleanup_filters = dl.Filters()
            cleanup_filters.add(field="hidden", values=True)
            cleanup_filters.add(field="dir", values=cleanup_dir)
            old_items = list(item.dataset.items.list(filters=cleanup_filters).all())
            for old_item in old_items:
                logger.info(f"Cleaning up stale item {old_item.id} from {cleanup_dir}")
                old_item.delete()

        segment_items: List[dl.Item] = []
        for idx, segment in enumerate(outline.segments):
            logger.info(
                f"Processing segment {idx + 1}/{total_segments}: {segment.section}"
            )
            segment_item = DialogueServiceRunner._process_segment(
                item, segment, idx, total_segments, summary
            )
            segment_items.append(segment_item)

        return segment_items

    @staticmethod
    def generate_dialogue(
        item: dl.Item, progress: dl.Progress, context: dl.Context
    ) -> dl.Item:
        """
        Generate dialogue for each segment.

        Args:
            item (dl.Item): Dataloop item containing the outline segment to be converted to dialogue
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            dl.Item: Dataloop item containing the generated dialogue
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        podcast_metadata.validate_stage("segment")

        if podcast_metadata.segment_idx is None or podcast_metadata.total_segments is None:
            raise ValueError(
                f"No segment index or total segments found in item {item.id}. "
                f"Check that segments were properly processed."
            )

        if podcast_metadata.outline_item_id is None:
            raise ValueError(f"No outline item id found in item {item.id}.")
        outline = SharedServiceRunner._get_outline_dict(
            outline_item=dl.items.get(item_id=podcast_metadata.outline_item_id)
        )

        segment_text = SharedServiceRunner._get_last_message(item=item)
        if segment_text is None:
            raise ValueError(f"No segment text found in item {item.id}.")

        segment_idx = podcast_metadata.segment_idx
        logger.info(
            f"Converting segment {segment_idx + 1}/{podcast_metadata.total_segments} to dialogue"
        )

        topics_text = "\n".join(
            [
                f"- {topic.title}\n"
                + "\n".join([f"  * {point.description}" for point in topic.points])
                for topic in outline.segments[segment_idx].topics
            ]
        )

        template = PodcastPrompts.get_template("podcast_transcript_to_dialogue_prompt")
        llm_prompt = template.render(
            text=segment_text,
            duration=outline.segments[segment_idx].duration,
            descriptions=topics_text,
            speaker_1_name=podcast_metadata.speaker_1_name,
            speaker_2_name=podcast_metadata.speaker_2_name,
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt5_segment_{segment_idx:02d}_dialogue"
        new_dir = SharedServiceRunner._get_segment_dir(
            item=dl.items.get(item_id=podcast_metadata.outline_item_id),
            pdf_name=podcast_metadata.pdf_name,
            stage=SEGMENT_DIR_PROMPT5,
        )
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "dialogue",
        })
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=new_dir,
            item_metadata=new_meta.to_item_metadata(),
        )
        return new_item

    @staticmethod
    def combine_dialogues(
        item: dl.Item, model: dl.Model, progress: dl.Progress, context: dl.Context
    ) -> dl.Item:
        """
        Combine all dialogue segments into one cohesive conversation.

        Function should only receive a list that is at least 2 items long, and dialogue item is passed after the first iteration.
        List is sorted by segment index in ascending order.

        Args:
            item (dl.Item): last segment to be processed by LLM
            model (dl.Model): Dataloop model entity
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            new_item (dl.Item): Dataloop item containing the combined dialogue
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)

        # Resolve the outline — try outline_item_id first, fall back to querying by name
        outline_item_id = podcast_metadata.outline_item_id
        if outline_item_id is not None:
            outline_item = dl.items.get(item_id=outline_item_id)
        else:
            # Wait node may return a parent item without outline_item_id.
            # Query for the structured outline in the working directory.
            working_dir = SharedServiceRunner._get_hidden_dir(item=item)
            outline_filters = dl.Filters()
            outline_filters.add(field="hidden", values=True)
            outline_filters.add(field="dir", values=working_dir)
            outline_filters.add(
                field="name",
                values=f"*prompt3_structured_outline*",
            )
            candidates = list(item.dataset.items.list(filters=outline_filters).all())
            if not candidates:
                raise ValueError(
                    f"No outline item id found in item {item.id} metadata, "
                    f"and could not find a structured outline item in {working_dir}."
                )
            outline_item = candidates[0]
            logger.info(
                f"Resolved structured outline via query: {outline_item.id} "
                f"(item {item.id} had no outline_item_id in metadata)"
            )

        outline = SharedServiceRunner._get_outline_dict(outline_item=outline_item)
        prompt5_dir = SharedServiceRunner._get_segment_dir(
            item=outline_item,
            pdf_name=podcast_metadata.pdf_name,
            stage=SEGMENT_DIR_PROMPT5,
        )

        filters = dl.Filters()
        filters.add(field="hidden", values=True)
        filters.add(field="dir", values=prompt5_dir)
        filters.sort_by(field="filename")
        segment_items = list(outline_item.dataset.items.list(filters=filters).all())
        if len(segment_items) < 2:
            raise ValueError(
                f"Insufficient dialogue segments found in {prompt5_dir} "
                f"(found {len(segment_items)}, need at least 2)."
            )

        logger.info(f"Combining {len(segment_items)} dialogue segments")

        dialogue_dict = {}
        for idx, segment_item in enumerate(segment_items):
            dialogue_dict[idx] = SharedServiceRunner._get_last_message(
                item=segment_item
            )

        segment_pairs = list(
            zip(segment_items[:-1], segment_items[1:], range(len(segment_items)))
        )
        current_dialogue = dialogue_dict[0]
        total_combines = len(segment_pairs) - 1

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt6_combined_dialogue"
        working_dir = SharedServiceRunner._get_hidden_dir(item=outline_item)
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "combined_dialogue",
            "outline_item_id": outline_item.id,
        })

        for idx in range(1, len(segment_pairs)):
            logger.info(f"Combining segment pair {idx}/{total_combines}")
            if idx != 1:
                new_item = dl.items.get(item_id=new_item.id)
                current_dialogue = SharedServiceRunner._get_last_message(item=new_item)
            next_section = dialogue_dict[idx]
            current_section = outline.segments[idx].section

            template = PodcastPrompts.get_template("podcast_combine_dialogues_prompt")
            llm_prompt = template.render(
                outline=outline.model_dump_json(),
                dialogue_transcript=current_dialogue,
                next_section=next_section,
                current_section=current_section,
            )

            new_item = SharedServiceRunner._create_and_upload_prompt_item(
                dataset=outline_item.dataset,
                item_name=new_name,
                prompt=llm_prompt,
                remote_dir=working_dir,
                overwrite=True,
                item_metadata=new_meta.to_item_metadata(),
            )

            ex = model.predict(item_ids=[new_item.id])
            logger.info(f"Model predict execution started: {ex.id}")
            ex = dl.executions.wait(execution=ex, timeout=300)
            if ex.latest_status["status"] not in ["success"]:
                raise ValueError(
                    f"Combine step {idx}/{total_combines} failed. "
                    f"Execution id: {ex.id}, status: {ex.latest_status['status']}"
                )
            progress.update(progress=int((idx / total_combines) * 100))
        return new_item

    @staticmethod
    def create_convo_json(
        item: dl.Item, model: dl.Model, progress: dl.Progress, context: dl.Context
    ) -> dl.Item:
        """
        Convert the dialogue into structured Conversation format.

        Args:
            item (dl.Item): Dataloop item containing the dialogue
            model (dl.Model): Dataloop model entity for setting max_tokens
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            dl.Item: Dataloop item containing the structured conversation

        Formats the dialogue into a structured conversation format with proper speaker
        attribution and timing information.
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)

        dialogue = SharedServiceRunner._get_last_message(item=item)
        if dialogue is None:
            raise ValueError(f"No dialogue found in item {item.id}.")
        dialogue += " Do not include titles in unescaped quotes."

        logger.info("Formatting final conversation")

        schema = Conversation.model_json_schema()
        SharedServiceRunner._set_model_configuration(model, max_tokens=2048)

        template = PodcastPrompts.get_template("podcast_dialogue_prompt")
        llm_prompt = template.render(
            speaker_1_name=podcast_metadata.speaker_1_name,
            speaker_2_name=podcast_metadata.speaker_2_name,
            text=dialogue,
            schema=json.dumps(schema, indent=2),
        )

        new_name = f"{Path(podcast_metadata.pdf_name).stem}_prompt7_convo_json"
        new_meta = podcast_metadata.model_copy(update={
            "pipeline_stage": "convo_json",
        })
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            overwrite=True,
            item_metadata=new_meta.to_item_metadata(),
        )
        return new_item


if __name__ == "__main__":
    item = dl.items.get(item_id="68064d5289d1cf34433fb28a")
    progress = dl.Progress()
    context = dl.Context()
    DialogueServiceRunner.combine_dialogues(
        item=item,
        model=dl.models.get(model_id="67ed3672f41fe3426dd2c3e0"),
        progress=progress,
        context=context,
    )
