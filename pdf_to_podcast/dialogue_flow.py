import json
import logging
import dtlpy as dl

from pathlib import Path
from typing import Dict, Any, List
from pdf_to_podcast.shared_functions import SharedServiceRunner
from pdf_to_podcast.podcast_prompts import PodcastPrompts
from pdf_to_podcast.podcast_types import PodcastOutline, Conversation

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class DialogueServiceRunner(dl.BaseServiceRunner):
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
            raise ValueError("No podcast metadata found in the prompt item. Try running the previous step again.")
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        pdf_name = podcast_metadata.get("pdf_name", None)

        # get the summary from the last prompt annotation
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        summary = last_message.get("content", [])[0].get("text", None)
        if summary is None:
            raise ValueError("No text summary found in the prompt item. Try running the previous step again.")

        logger.info("Preparing to generate initial outline")

        # create summary file
        summary_filename = f"{Path(pdf_name).stem}_summary.txt"
        with open(summary_filename, "w", encoding='utf-8') as f:
            f.write(summary)

        summary_item = item.dataset.items.upload(
            local_path=summary_filename,
            remote_name=summary_filename,
            remote_path=item.dir,
            overwrite=True,
            item_metadata={"user": item.metadata['user']},
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
            total_duration=duration, focus_instructions=focus, documents="\n\n".join(documents)
        )

        # create new prompt item for the raw outline
        new_name = f"{Path(pdf_name).stem}_prompt2_raw_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_metadata = item.metadata
        new_metadata["user"]["podcast"]["summary_item_id"] = summary_item.id
        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=new_metadata
        )
        return new_item

    @staticmethod
    def generate_structured_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
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
            raise ValueError("No podcast metadata found in the prompt item. Try running the previous step again.")
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
        valid_filenames = [pdf_name] + references  # TODO support multiple pdfs as valid files to reference
        schema = PodcastOutline.model_json_schema()
        schema["$defs"]["PodcastSegment"]["properties"]["references"]["items"] = {
            "type": "string",
            "enum": valid_filenames,
        }

        template = PodcastPrompts.get_template("podcast_multi_pdf_structured_outline_prompt")
        llm_prompt = template.render(
            outline=raw_outline, schema=json.dumps(schema, indent=2), valid_filenames=valid_filenames
        )

        # create new prompt item for the structured outline
        new_name = f"{Path(pdf_name).stem}_prompt3_structured_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
        return new_item

    @staticmethod
    def _process_segment(
        item: dl.Item, segment: Any, idx: int, total_segments: int, focus: str, duration: int, summary: str
    ) -> dl.Item:
        """
        Process a single outline segment to generate initial content.

        Args:
            item (dl.Item): Dataloop item containing the outline segment
            segment (Any): Segment from the outline to process
            idx (int): Index of the segment
            total_segments (int): Total number of segments
            focus (str): Focus of the podcast
            duration (int): Duration of the podcast
            summary (str): Summary of the podcast

        Returns:
            dl.Item: Dataloop item containing the generated content
        """
        logger.info(f"Preparing to generate initial content for segment {idx + 1}/{total_segments}")

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        if podcast_metadata is None:
            raise ValueError("No podcast metadata found in the prompt item. Try running the previous step again.")
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        pdf_name = podcast_metadata.get("pdf_name", None)

        # Get the PDF content
        text_content = []
        # TODO support multiple documents
        text_content = [f"Document: {item.name}\n{summary}"]

        # Choose template based on whether we have references
        template_name = "podcast_prompt_with_references" if text_content else "podcast_prompt_no_references"
        template = PodcastPrompts.get_template(template_name)

        # Prepare prompt parameters
        llm_prompt_params = {
            "duration": segment.duration,
            "topic": segment.section,
            "angles": "\n".join([topic.title for topic in segment.topics]),
        }

        # Add text content if we have references
        if text_content:
            llm_prompt_params["text"] = "\n\n".join(text_content)

        llm_prompt = template.render(**llm_prompt_params)

        # Create a new prompt item
        new_name = f"{Path(item.name).stem}_prompt4_segment_{idx}"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )

        # Update metadata with segment information
        new_metadata = podcast_metadata.copy()
        new_metadata.update(
            {
                "focus": focus,
                "duration": duration,
                "summary": summary,
                "segment": segment.section,
                "segment_idx": idx,
                "total_segments": total_segments,
                "topics": [topic.title for topic in segment.topics],
                "references": [reference.filename for reference in segment.references],
            }
        )
        new_name = f"{Path(pdf_name).stem}_prompt4_segment_{idx}"
        new_dir = f"{item.dir}/.dataloop/{pdf_name}"
        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=new_dir, overwrite=True, item_metadata=new_metadata
        )
        return new_item

    @staticmethod
    def process_segments(item: dl.Item, progress: dl.Progress, context: dl.Context) -> List[dl.Item]:
        """
        Process all outline segments in parallel to generate initial content.

        Args:
            item (dl.Item): Dataloop item containing the outline
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            List[dl.Item]: List of Dataloop items containing the prompt items for each segment
        """
        podcast_metadata = item.metadata.get("user", {}).get("podcast", {})
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        summary = DialogueServiceRunner._get_summary_text(summary_item_id=podcast_metadata.get("summary_item_id", None))

        # Create the outline item
        outline = DialogueServiceRunner._get_outline_dict(outline_item=item)

        logger.info(f"Preparing to process {len(outline.segments)} segments")

        # Create items for processing each segment
        segment_items: List[Any] = []
        for idx, segment in enumerate(outline.segments):
            logger.info(f"Processing segment {idx + 1}/{len(outline.segments)}: {segment.section}")

            segment_item = DialogueServiceRunner._process_segment(
                item, segment, idx, len(outline.segments), focus, duration, summary
            )
            segment_items.append(segment_item)

        return segment_items

    @staticmethod
    def _generate_dialogue_segment(item: dl.Item, outline: PodcastOutline, segment: Any, idx: int) -> dl.Item:
        """
        Generate dialogue for a single segment.

        Args:
            segment (Any): Segment from the outline
            idx (int): Index of the segment

        Returns:
            dl.Item: Dataloop item containing the generated dialogue

        Formats segment topics and uses a template to convert content into a dialogue
        format between two speakers.
        """
        # Format topics for prompt
        item_metadata = item.metadata
        podcast_metadata = item_metadata.get("user", {}).get("podcast", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", "Alice")
        speaker_2_name = podcast_metadata.get("speaker_2_name", "Will")

        segment_text = DialogueServiceRunner._get_last_message(item=item)
        if segment_text is None:
            raise ValueError(f"No segment text found in item {item.id}.")

        # Format topics for prompt
        topics_text = "\n".join(
            [
                f"- {topic.title}\n" + "\n".join([f"  * {point.description}" for point in topic.points])
                for topic in outline.segments[idx].topics
            ]
        )

        # Generate dialogue using template
        template = PodcastPrompts.get_template("podcast_transcript_to_dialogue_prompt")
        llm_prompt = template.render(
            text=segment_text,
            duration=segment.duration,
            descriptions=topics_text,
            speaker_1_name=speaker_1_name,
            speaker_2_name=speaker_2_name,
        )

        # Create new prompt item for the dialogue
        new_name = f"{Path(item.name).stem}_prompt5_dialogue"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )

        # Update metadata with segment information
        new_metadata = item.metadata.copy()
        new_metadata.update(
            {
                "segment": segment.section,
                "segment_idx": idx, 
                "total_segments": len(outline.segments),
            }
        )

        # Upload the new prompt item
        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=new_metadata
        )
        return new_item

    @staticmethod
    def generate_dialogue(item: dl.Item, progress: dl.Progress, context: dl.Context) -> List[dl.Item]:
        """
        Generate dialogue for each segment.

        Args:
            item (dl.Item): Dataloop item containing the outline segment to be converted to dialogue
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object


        Returns:
            List[dl.Item]: List of Dataloop items containing the generated dialogue

        Creates tasks for generating dialogue for each segment and executes them in parallel.
        """
        logger.info("Generating segment dialogue")
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        outline_item_id = podcast_metadata.get("outline_item_id", None)
        outline = DialogueServiceRunner._get_outline_dict(outline_item=dl.items.get(item_id=outline_item_id))
        segment_text = DialogueServiceRunner._get_last_message(item=item)
        if segment_text is None:
            raise ValueError(f"No segment text found in item {item.id}.")

        # Update status
        logger.info(
            f"Converting segment {podcast_metadata.get('segment_idx', 0) + 1}/{podcast_metadata.get('total_segments', 0)} to dialogue"
        )

        segment_items = []
        for idx, segment in enumerate(outline.segments):
            segment_items.append(DialogueServiceRunner._generate_dialogue_segment(item, outline, segment, idx))

        return segment_items

    @staticmethod
    def combine_dialogues(item: dl.Item, progress: dl.Progress, context: dl.Context) -> dl.Item:
        """
        Combine all dialogue segments into one cohesive conversation.

        Args:
            item (dl.Item): Dataloop item containing the original structure outline

        Returns:
            new_item (dl.Item): Dataloop item containing the combined dialogue
        """
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        pdf_name = podcast_metadata.get("pdf_name", None)
        
        logger.info("Combining dialogue segments")

        # search for all the segment dialogues
        segments_dir = f"{item.dir}/.dataloop/{pdf_name}"
        filters = dl.Filters()
        filters.add(field='dir', values=segments_dir)
        filters.add(field='metadata.system.mimetype', values='*json*')
        segment_items = item.dataset.items.list(filters=filters).all()

        # combine the segment dialogues
        combined_dialogue = ""
        for segment_item in segment_items:
            segment_text = DialogueServiceRunner._get_last_message(item=segment_item)
            if segment_text is None:
                raise ValueError(f"No segment text found in item {segment_item.id}.")
            combined_dialogue += segment_text

        # create a new prompt item for the combined dialogue
        new_name = f"{Path(item.name).stem}_prompt6_combined_dialogue"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": combined_dialogue}]}  # role default is user
        )
        new_item = item.dataset.items.upload(
            prompt_item,
            remote_name=new_name,
            remote_path=item.dir,
            overwrite=True,
        )
        return new_item

    @staticmethod
    def create_convo_json(item: dl.Item, progress: dl.Progress, context: dl.Context) -> dl.Item:
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
        speaker_1_name = podcast_metadata.get("speaker_1_name", "Alice")
        speaker_2_name = podcast_metadata.get("speaker_2_name", "Will")

        dialogue = DialogueServiceRunner._get_last_message(item=item)
        if dialogue is None:
            raise ValueError(f"No dialogue found in item {item.id}.")

        schema = {
            "type": "object",
            "properties": {
                "scratchpad": {"type": "string"},
                "dialogue": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "speaker": {"type": "string", "enum": [speaker_1_name, speaker_2_name]},
                        },
                        "required": ["text", "speaker"],
                    },
                },
            },
            "required": ["scratchpad", "dialogue"],
        }
        template = PodcastPrompts.get_template("podcast_dialogue_prompt")
        llm_prompt = template.render(
            speaker_1_name=speaker_1_name,
            speaker_2_name=speaker_2_name,
            text=dialogue,
            schema=json.dumps(schema, indent=2),
        )

        prompt_item = dl.PromptItem(name=f"{Path(item.name).stem}_prompt_json")
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]},  # role default is user
        )

        new_name = f"{Path(item.name).stem}_prompt7_convo_json"
        new_item = item.dataset.items.upload(prompt_item, 
                                             remote_name=new_name, 
                                             remote_path=item.dir, 
                                             overwrite=True, 
                                             item_metadata=item.metadata)

        return new_item

    @staticmethod
    def create_final_conversation(item: dl.Item, progress: dl.Progress, context: dl.Context) -> dl.Item:
        """
        Create a final conversation from the dialogue.

        Args:
            item (dl.Item): Dataloop item containing the dialogue

        Returns:
            dl.Item: Dataloop item containing the structured conversation JSON
        """
        logger.info("Formatting final conversation")

        conversation_json_str = DialogueServiceRunner._get_last_message(item=item)
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

        # return DialogueServiceRunner.create_convo_json(dir_item, dialogue)
        return item
