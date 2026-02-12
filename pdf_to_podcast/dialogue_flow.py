import json
import logging
import dtlpy as dl

from pathlib import Path
from typing import List, Tuple
from pdf_to_podcast.shared_functions import SharedServiceRunner
from pdf_to_podcast.podcast_prompts import PodcastPrompts
from pdf_to_podcast.podcast_types import PodcastOutline, Conversation, PodcastSegment

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
        # get the podcast metadata from the item
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        focus = podcast_metadata.get("focus")
        duration = podcast_metadata.get("duration")
        pdf_name = podcast_metadata.get("pdf_name")
        working_dir = SharedServiceRunner._get_hidden_dir(item=item)

        # get the summary from the last prompt annotation
        summary = SharedServiceRunner._get_last_message(item)
        if summary is None:
            raise ValueError(
                "No text summary found in the prompt item. Try running the previous step again."
            )

        logger.info("Preparing to generate initial outline")

        # save and upload summary file
        summary_filename = f"{Path(pdf_name).stem}_summary.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(summary)
        summary_item = item.dataset.items.upload(
            local_path=summary_filename,
            remote_name=summary_filename,
            remote_path=working_dir,
            overwrite=True,
            item_metadata={"user": item.metadata["user"]},
        )

        logger.info(f"Saved PDF summary as text item {summary_item.id}")

        # generate the outline prompt
        documents = [f"Document: {pdf_name}\n{summary}"]
        # TODO support multiple pdfs as context
        # add a section of the metadata that includes the filename, pdf id, and summary text item id for each pdf
        # then load each of the summary texts, and compile into them into a new json item to load all the relevant pdfs + metadata
        # This is the original code:
        # for pdf in summarized_pdfs:
        #     doc_str = f"""
        #     <document>
        #     <type>{"Target Document" if pdf.type == "target" else "Context Document"}</type>
        #     <path>{pdf.filename}</path>
        #     <summary>
        #     {pdf.summary}
        #     </summary>
        #     </document>"""
        #     documents.append(doc_str)

        template = PodcastPrompts.get_template("podcast_multi_pdf_outline_prompt")
        llm_prompt = template.render(
            total_duration=duration,
            focus_instructions=focus,
            documents="\n\n".join(documents),
        )

        # create new prompt item for the raw outline
        new_name = f"{Path(pdf_name).stem}_prompt2_raw_outline"
        new_metadata = item.metadata.get("user", {})
        new_metadata["podcast"] = new_metadata.get("podcast", {})
        new_metadata["podcast"]["summary_item_id"] = summary_item.id

        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=working_dir,
            overwrite=True,
            item_metadata={"user": new_metadata},
        )
        return new_item

    @staticmethod
    def generate_structured_outline(
        item: dl.Item, progress: dl.Progress, context: dl.Context
    ):
        """
        Convert raw outline text to structured PodcastOutline format.

        Uses JSON schema validation to ensure the outline follows the required structure
        and only references valid PDF filenames.

        Args:
            item (dl.Item): Dataloop item containing the raw outline in the hidden directory
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object
            prompt_focus (str): Focus instructions guide for the prompt

        Returns:
            item (dl.Item): Item for prompting to generate structured outline following the PodcastOutline schema

        """
        # get the podcast metadata from the item
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        pdf_name = podcast_metadata.get("pdf_name")
        references = podcast_metadata.get("references")

        raw_outline = SharedServiceRunner._get_last_message(item)
        if raw_outline is None:
            raise ValueError(f"No outline found in item {item.id}.")

        logger.info("Preparing to generate structured outline")

        # Force the model to only reference valid filenames
        valid_filenames = [pdf_name]  # TODO
        if references is not None:
            valid_filenames.extend(references)
        schema = PodcastOutline.model_json_schema()
        schema["$defs"]["PodcastSegment"]["properties"]["references"]["items"] = {
            "type": "string",
            "enum": valid_filenames,
        }

        template = PodcastPrompts.get_template(
            "podcast_multi_pdf_structured_outline_prompt"
        )
        llm_prompt = template.render(
            outline=raw_outline,
            schema=json.dumps(schema, indent=2),
            valid_filenames=valid_filenames,
        )

        # create new prompt item for the structured outline
        new_name = f"{Path(pdf_name).stem}_prompt3_structured_outline"
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=item.dir,
            overwrite=True,
            item_metadata=item.metadata,
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
        pdf_name = podcast_metadata.get("pdf_name")

        # Get the PDF content
        # TODO support multiple documents from references
        text_content = [f"Document: {pdf_name}\n{summary}"]

        # Choose template based on whether we have references
        template_name = (
            "podcast_prompt_with_references"
            if text_content
            else "podcast_prompt_no_references"
        )
        template = PodcastPrompts.get_template(template_name)

        # Prepare prompt parameters
        llm_prompt_params = {
            "duration": segment.duration,
            "topic": segment.section,
            "angles": "\n".join([topic.title for topic in segment.topics]),
        }

        # Add text content if we have references
        if text_content != []:
            llm_prompt_params["text"] = "\n\n".join(text_content)

        llm_prompt = template.render(**llm_prompt_params)

        # Create a new prompt item
        new_name = f"{Path(pdf_name).stem}_prompt4_segment_{idx:02d}"
        new_dir = f"{SharedServiceRunner._get_hidden_dir(item=item)}/{pdf_name}/prompt4"
        new_metadata = podcast_metadata.copy()
        new_metadata.update(
            {
                "outline_item_id": item.id,
                "segment_topic": segment.section,
                "segment_idx": idx,
                "total_segments": total_segments,
                "topics": [topic.title for topic in segment.topics],
                # "references": [reference.filename for reference in segment.references], # TODO
            }
        )

        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=new_dir,
            item_metadata={"user": {"parentItemId": item.id, "podcast": new_metadata}},
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
        summary = SharedServiceRunner._get_summary_from_id(
            podcast_metadata.get("summary_item_id")
        )

        # get the outline from item
        outline = SharedServiceRunner._get_outline_dict(outline_item=item)
        total_segments = len(outline.segments)
        logger.info(f"Preparing to process {total_segments} segments")

        # Create items for processing each segment
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
        # check item is the structured outline

        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        pdf_name = podcast_metadata.get("pdf_name")
        speaker_1_name = podcast_metadata.get("speaker_1_name")
        speaker_2_name = podcast_metadata.get("speaker_2_name")
        segment_idx = podcast_metadata.get("segment_idx")
        total_segments = podcast_metadata.get("total_segments")
        if segment_idx is None or total_segments is None:
            raise ValueError(
                f"No segment index or total segments found in item {item.id}. Check that segments were properly processed."
            )

        outline_item_id = podcast_metadata.get("outline_item_id")
        if outline_item_id is None:
            raise ValueError(f"No outline item id found in item {item.id}.")
        outline = SharedServiceRunner._get_outline_dict(
            outline_item=dl.items.get(item_id=outline_item_id)
        )

        segment_text = SharedServiceRunner._get_last_message(item=item)
        if segment_text is None:
            raise ValueError(f"No segment text found in item {item.id}.")

        logger.info(
            f"Converting segment {segment_idx + 1}/{total_segments} to dialogue"
        )

        # Format topics for prompt
        topics_text = "\n".join(
            [
                f"- {topic.title}\n"
                + "\n".join([f"  * {point.description}" for point in topic.points])
                for topic in outline.segments[segment_idx].topics
            ]
        )

        # Generate dialogue using template
        template = PodcastPrompts.get_template("podcast_transcript_to_dialogue_prompt")
        llm_prompt = template.render(
            text=segment_text,
            duration=outline.segments[segment_idx].duration,
            descriptions=topics_text,
            speaker_1_name=speaker_1_name,
            speaker_2_name=speaker_2_name,
        )

        # Create new prompt item for the dialogue
        new_name = f"{Path(pdf_name).stem}_prompt5_segment_{segment_idx:02d}_dialogue"
        new_dir = SharedServiceRunner._get_hidden_dir(item=item).replace(
            "prompt4", "prompt5"
        )
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=new_dir,
            item_metadata=item.metadata,
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
        # load podcast params and outline
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        pdf_name = podcast_metadata.get("pdf_name")
        # item is the original structured outline prompt item
        outline_item_id = podcast_metadata.get("outline_item_id")
        if outline_item_id is None:
            raise ValueError(f"No outline item id found in item {item.id}.")
        outline_item = dl.items.get(item_id=outline_item_id)
        outline = SharedServiceRunner._get_outline_dict(outline_item=outline_item)
        working_dir = SharedServiceRunner._get_hidden_dir(item=outline_item)

        # get all segment items
        filters = dl.Filters()
        filters.add(field='hidden', values=True)
        filters.add(
            field="dir",
            values=f"{working_dir}/{pdf_name}/prompt5",
        )
        filters.sort_by(field="filename")
        segment_items = list(outline_item.dataset.items.list(filters=filters).all())
        if len(segment_items) < 2:
            raise ValueError(
                "Insufficient segments for a podcast. At least 2 segments are required to combine dialogues."
            )

        logger.info("Combining dialogue segments")

        # create a dictionary that indexes the dialogue from each enumerated segment
        dialogue_dict = {}
        for idx, segment_item in enumerate(segment_items):
            dialogue_dict[idx] = SharedServiceRunner._get_last_message(
                item=segment_item
            )

        # create a list that pairs each segment item with the next one, and the segment index
        segment_pairs = list(
            zip(segment_items[:-1], segment_items[1:], range(len(segment_items)))
        )
        current_dialogue = dialogue_dict[0]

        # create a new prompt item for the combined dialogue
        new_name = f"{Path(pdf_name).stem}_prompt6_combined_dialogue"
        for idx in range(1, len(segment_pairs)):
            if idx != 1:
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
                item_metadata=item.metadata,
            )

            ex = model.predict(item_ids=[new_item.id])
            logger.info(f"Model predict execution started: {ex.id}")
            ex = dl.executions.wait(execution=ex, timeout=300)
            if ex.latest_status["status"] not in ["success"]:
                raise ValueError(f"Execution failed. ex id: {ex.id}")
        return new_item

    @staticmethod
    def create_convo_json(
        item: dl.Item, progress: dl.Progress, context: dl.Context
    ) -> dl.Item:
        """
        Convert the dialogue into structured Conversation format.

        Args:
            item (dl.Item): Dataloop item containing the dialogue
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            dl.Item: Dataloop item containing the structured conversation

        Formats the dialogue into a structured conversation format with proper speaker
        attribution and timing information.
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        pdf_name = podcast_metadata.get("pdf_name")
        speaker_1_name = podcast_metadata.get("speaker_1_name")
        speaker_2_name = podcast_metadata.get("speaker_2_name")

        dialogue = SharedServiceRunner._get_last_message(item=item)
        if dialogue is None:
            raise ValueError(f"No dialogue found in item {item.id}.")
        dialogue += "Do not include titles in unscaped quotes."

        logger.info("Formatting final conversation")

        schema = Conversation.model_json_schema()
        template = PodcastPrompts.get_template("podcast_dialogue_prompt")
        llm_prompt = template.render(
            speaker_1_name=speaker_1_name,
            speaker_2_name=speaker_2_name,
            text=dialogue,
            schema=json.dumps(schema, indent=2),
        )

        new_name = f"{Path(pdf_name).stem}_prompt7_convo_json"
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=item),
            overwrite=True,
            item_metadata=item.metadata,
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
