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

DEFAULT_SPEAKER_1_NAME = "Alice"
DEFAULT_SPEAKER_2_NAME = "Will"

class DialogueServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def generate_raw_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate initial raw outline from summarized PDFs.

        Args:
            item (dl.Item): Dataloop item containing the podcast summary
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        # get the podcast metadata from the item
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        if podcast_metadata is None:
            raise ValueError(
                "No podcast metadata found in the prompt item. Try running the previous step again."
            )
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        pdf_name = podcast_metadata.get("pdf_name", None)

        # get the summary from the last prompt annotation
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        summary = last_message.get("content", [])[0].get("text", None)
        if summary is None:
            raise ValueError(
                "No text summary found in the prompt item. Try running the previous step again."
            )

        logger.info("Preparing to generate initial outline")

        # create summary file
        summary_filename = f"{Path(pdf_name).stem}_summary.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(summary)

        summary_item = item.dataset.items.upload(
            local_path=summary_filename,
            remote_name=summary_filename,
            remote_path=item.dir,
            overwrite=True,
            item_metadata={"user": item.metadata["user"]},
        )

        logger.info(f"Saved PDF summary as text item {summary_item.id}")

        # generate the outline
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
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={
                "content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]
            }  # role default is user
        )
        new_metadata = item.metadata.get("user", {})
        new_metadata["podcast"] = new_metadata.get("podcast", {})
        new_metadata["podcast"]["summary_item_id"] = summary_item.id
        new_item = item.dataset.items.upload(
            prompt_item,
            remote_name=new_name,
            remote_path=item.dir,
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
            item (dl.Item): Dataloop item containing the raw outline
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object
            prompt_focus (str): Focus instructions guide for the prompt

        Returns:
            item (dl.Item): Item for prompting to generate structured outline following the PodcastOutline schema

        """
        # get the podcast metadata from the item
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        if podcast_metadata is None:
            raise ValueError(
                "No podcast metadata found in the prompt item. Try running the previous step again."
            )
        pdf_name = podcast_metadata.get("pdf_name", None)
        references = podcast_metadata.get("references", None)  # TODO

        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        raw_outline = last_message.get("content", [])[0].get("text", None)
        if raw_outline is None:
            raise ValueError(f"No outline found in item {item.id}.")

        logger.info("Preparing to generate structured outline")

        # Force the model to only reference valid filenames
        valid_filenames = [pdf_name]
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
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={
                "content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]
            }  # role default is user
        )
        new_item = item.dataset.items.upload(
            prompt_item,
            remote_name=new_name,
            remote_path=item.dir,
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
        focus: str,
        duration: int,
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
            focus (str): Focus of the podcast
            duration (int): Duration of the podcast
            summary (str): Summary of the podcast

        Returns:
            dl.Item: Dataloop item containing the generated content
        """
        logger.info(
            f"Preparing to generate initial content for segment {idx + 1}/{total_segments}"
        )

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        if podcast_metadata is None:
            raise ValueError(
                "No podcast metadata found in the prompt item. Try running the previous step again."
            )
        pdf_name = podcast_metadata.get("pdf_name", None)

        # Get the PDF content
        # TODO support multiple documents
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
        new_name = f"{Path(pdf_name).stem}_prompt4_segment_{idx}"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={
                "content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]
            }  # role default is user
        )

        # Update metadata with segment information
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
        new_name = f"{Path(pdf_name).stem}_prompt4_segment_{idx:02d}"
        if item.dir == "/":
            new_dir = f"/segments/{pdf_name}/prompt4"
        else:
            new_dir = f"{item.dir}/segments/{pdf_name}/prompt4"
        new_item = item.dataset.items.upload(
            prompt_item,
            remote_name=new_name,
            remote_path=new_dir,
            overwrite=True,
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
        podcast_metadata = item.metadata.get("user", {}).get("podcast", {})
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        summary_item_id = podcast_metadata.get("summary_item_id", None)
        summary = SharedServiceRunner._get_summary_text(summary_item_id=summary_item_id)

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
                item, segment, idx, total_segments, focus, duration, summary
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

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        pdf_name = podcast_metadata.get("pdf_name", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", DEFAULT_SPEAKER_1_NAME)
        speaker_2_name = podcast_metadata.get("speaker_2_name", DEFAULT_SPEAKER_2_NAME)
        segment_idx = podcast_metadata.get("segment_idx", None)
        total_segments = podcast_metadata.get("total_segments", None)
        if segment_idx is None or total_segments is None:
            raise ValueError(
                f"No segment index or total segments found in item {item.id}. Check that segments were properly processed."
            )

        outline_item_id = podcast_metadata.get("outline_item_id", None)
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
        new_dir = item.dir.replace("prompt4", "prompt5")
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={
                "content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]
            }  # role default is user
        )

        # Upload the new prompt item
        new_item = item.dataset.items.upload(
            prompt_item,
            remote_name=new_name,
            remote_path=new_dir,
            overwrite=True,
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
            item (dl.Item): Dataloop item containing the outline segment to be converted to dialogue
            model (dl.Model): Dataloop model entity
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            new_item (dl.Item): Dataloop item containing the combined dialogue
        """
        # get all segment items
        filters = dl.Filters()
        filters.add(
            field="dir",
            values=f"/segments/{item.metadata['user']['podcast']['pdf_name']}/prompt5",
        )
        filters.sort_by(field="filename")
        segment_items = list(item.dataset.items.list(filters=filters).all())
        if len(segment_items) < 2:
            raise ValueError(
                "Insufficient segments for a podcast. At least 2 segments are required to combine dialogues."
            )

        # load podcast params and outline
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        pdf_name = podcast_metadata.get("pdf_name", None)
        # item is the original structured outline prompt item
        outline = SharedServiceRunner._get_outline_dict(outline_item=item)

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
        new_name = f"{Path(pdf_name).stem}_prompt6_combined_dialogue.json"
        for idx in range(1, len(segment_pairs)):
            if idx != 1:
                current_dialogue = SharedServiceRunner._get_last_message(item=new_item)
                print(f"Current dialogue: {current_dialogue}")  # TODO DEBUG DELETE
                print(f"New item: {new_item.name} ({new_item.id})")  # TODO DEBUG DELETE
            next_section = dialogue_dict[idx]
            current_section = outline.segments[idx].section

            template = PodcastPrompts.get_template("podcast_combine_dialogues_prompt")
            llm_prompt = template.render(
                outline=outline.model_dump_json(),
                dialogue_transcript=current_dialogue,
                next_section=next_section,
                current_section=current_section,
            )

            prompt_item = dl.PromptItem(name=new_name)
            prompt_item.add(
                message={
                    "content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]
                }  # role default is user
            )
            new_item = item.dataset.items.upload(
                prompt_item,
                remote_name=new_name,
                remote_path=item.dir,
                overwrite=True,
                item_metadata=item.metadata,
            )

            ex = model.predict(item_ids=[new_item.id])
            logger.info(f"Model predict execution started: {ex.id}")
            ex = dl.executions.wait(execution=ex, timeout=300)
            if ex.latest_status["status"] not in ["success"]:
                raise ValueError(f"Execution failed. ex id: {ex.id}")
        return new_item

    # @staticmethod
    # def check_dialogue(items: List[dl.Item], progress: dl.Progress, context: dl.Context) -> List[dl.Item]:
    #     """
    #     Check the dialogue for the first two items
    #     """
    #     actions = ['continue', 'iterate']
    #     # check that the list only has one item left
    #     if len(items) < 2:
    #         progress.update(action=actions[0])
    #     else:
    #         progress.update(action=actions[1])

    #     return items

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
        logger.info("Formatting final conversation")

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        pdf_name = podcast_metadata.get("pdf_name", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", "Alice")
        speaker_2_name = podcast_metadata.get("speaker_2_name", "Will")

        dialogue = SharedServiceRunner._get_last_message(item=item)
        if dialogue is None:
            raise ValueError(f"No dialogue found in item {item.id}.")
        dialogue += "Do not include titles in unscaped quotes."

        schema = Conversation.model_json_schema()
        template = PodcastPrompts.get_template("podcast_dialogue_prompt")
        llm_prompt = template.render(
            speaker_1_name=speaker_1_name,
            speaker_2_name=speaker_2_name,
            text=dialogue,
            schema=json.dumps(schema, indent=2),
        )

        new_name = f"{Path(pdf_name).stem}_prompt7_convo_json"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={
                "content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]
            }  # role default is user
        )

        new_item = item.dataset.items.upload(
            prompt_item,
            remote_name=new_name,
            remote_path=item.dir,
            overwrite=True,
            item_metadata=item.metadata,
        )

        return new_item
