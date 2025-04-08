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
        logger.info("Generating initial outline")

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

        # create summary file
        summary_filename = f"{Path(pdf_name).stem}_summary.txt"
        with open(summary_filename, "w", encoding='utf-8') as f:
            f.write(summary)

        summary_item = item.dataset.items.upload(local_path=summary_filename,
                                                 remote_name=summary_filename,
                                                 remote_path=item.dir,
                                                 overwrite=True,
                                                 item_metadata={"user": item.metadata['user']})

        # generate the outline
        documents = [f"Document: {pdf_name}\n{summary}"]
        # TODO support multiple pdfs as context
        # basically add a section of the metadata that includes the filename, pdf id, and summary text item id for each pdf
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
        new_name = f"{Path(item.name).stem}_prompt2_raw_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_metadata = item.metadata
        new_metadata["user"]["podcast"]["summary_item_id"] = summary_item.id
        new_item = item.dataset.items.upload(prompt_item,
                                             remote_name=new_name,
                                             remote_path=item.dir,
                                             overwrite=True,
                                             item_metadata=new_metadata)
        return new_item
    
    @staticmethod
    def generate_structured_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Convert raw outline text to structured PodcastOutline format.

        Args:
            item (dl.Item): Dataloop item containing the raw outline
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object
            prompt_focus (str): Focus instructions guide for the prompt

        Returns:
            PodcastOutline: Structured outline following the PodcastOutline schema

        Uses JSON schema validation to ensure the outline follows the required structure
        and only references valid PDF filenames.
        """
        logging.info("Converting raw outline to structured format")

        prompt_item = dl.PromptItem.from_json(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        raw_outline = last_message.get("content", [])[0].get("text", None)
        if raw_outline is None:
            raise ValueError("No raw outline found in the prompt item.")

        # Force the model to only reference valid filenames
        valid_filenames = [item.filename]
        schema = PodcastOutline.model_json_schema()
        schema["$defs"]["PodcastSegment"]["properties"]["references"]["items"] = {
            "type": "string",
            "enum": valid_filenames,
        }

        schema = PodcastOutline.model_json_schema()
        template = PodcastPrompts.get_template("podcast_multi_pdf_structured_outline_prompt")
        llm_prompt = template.render(
            outline=raw_outline, schema=json.dumps(schema, indent=2), valid_filenames=valid_filenames
        )

        # create new prompt item for the structured outline
        new_name = f"{Path(item.name).stem}_prompt3_structured_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_item = item.dataset.items.upload(prompt_item, 
                                             remote_name=new_name, 
                                             remote_path=item.dir, 
                                             overwrite=True)
        return new_item

    @staticmethod
    def _process_segment(
        item: dl.Item, segment: Any, idx: int, total_segments: int, focus: str, duration: int, summary: str
    ) -> dl.Item:
        """
        Process a single outline segment to generate initial content.

        Args:
            segment (Any): Segment from the outline to process
            idx (int): Index of the segment
            request (TranscriptionRequest): Original transcription request
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses

        Returns:
            dl.Item: Dataloop item containing the generated content

        Generates initial content for a segment, incorporating referenced PDF content
        if available. Uses different templates based on whether references exist.
        """
        prompt_item = dl.PromptItem.from_json(item)

        pdf_text = prompt_item.prompts[0].elements[0].value
        text_content = [f"Document: {item.filename}\n{pdf_text}"]

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

        prompt = dl.Prompt(key="3")  # "3_segment_prompt")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)
        prompt_item.prompts.append(prompt)

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        podcast_metadata.update(
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
        new_name = f"{Path(item.name).stem}_prompt4_segment_{idx}"
        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=podcast_metadata
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

        Creates tasks for processing each segment and executes them in parallel using
        asyncio.gather.
        """
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        summary = podcast_metadata.get("summary", None)

        # create the outline item
        prompt_item = dl.PromptItem.from_json(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        outline_dict = last_message.get("content", [])[0].get("text", None)

        if outline_dict is None:
            raise ValueError("No outline found in the prompt item.")
        if isinstance(outline_dict, str):
            outline_dict = json.loads(outline_dict)

        outline = PodcastOutline.model_validate_json(outline_dict)

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
    def _generate_dialogue_segment(item: dl.Item, segment: Any, idx: int) -> Dict[str, str]:
        """
        Generate dialogue for a single segment.

        Args:
            segment (Any): Segment from the outline
            idx (int): Index of the segment
            segment_text (str): Generated content for the segment
            request (TranscriptionRequest): Original transcription request
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses

        Returns:
            Dict[str, str]: Dictionary containing section name and generated dialogue

        Formats segment topics and uses a template to convert content into a dialogue
        format between two speakers.
        """
        # Format topics for prompt
        item_metadata = item.metadata
        podcast_metadata = item_metadata.get("user", {}).get("podcast", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", "Alice")
        speaker_2_name = podcast_metadata.get("speaker_2_name", "Will")

        prompt_item = dl.PromptItem.from_json(item)

        # Get the segment text from the last prompt
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        segment_text = last_message.get("content", [])[0].get("text", None)
        if segment_text is None:
            raise ValueError("No segment text found in the prompt item.")

        # Format topics for prompt
        topics_text = "\n".join(
            [
                f"- {topic.title}\n" + "\n".join([f"  * {point.description}" for point in topic.points])
                for topic in segment.topics
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

        # Upload the new prompt item
        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=item.metadata
        )

        return new_item

    @staticmethod
    def generate_dialogue(item: dl.Item) -> List[Dict[str, str]]:
        """
        Generate dialogue for each segment.

        Args:
            item (dl.Item): Dataloop item containing the outline
            outline (PodcastOutline): Structured outline

        Returns:
            List[Dict[str, str]]: List of dictionaries containing section names and dialogues

        Creates tasks for generating dialogue for each segment and executes them in parallel.
        """
        logger.info("Generating dialogue")
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)

        prompt_item = dl.PromptItem.from_json(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        segment_text = last_message.get("content", [])[0].get("text", None)
        if segment_text is None:
            raise ValueError("No segment text found in the prompt item.")

        # Update status
        logger.info(
            f"Converting segment {podcast_metadata.get('segment_idx', 0) + 1}/{podcast_metadata.get('total_segments', 0)} to dialogue"
        )

        dialogue_segments = []
        for idx, segment in enumerate(outline.segments):
            dialogue_segments.append(DialogueServiceRunner._generate_dialogue_segment(segment, idx, segment_text))

        return dialogue_segments

        # return item

    @staticmethod
    def combine_dialogues(item: dl.Item) -> str:
        """
        Iteratively combine dialogue segments into a cohesive conversation.

        Args:
            item (dl.Item): Dataloop item containing the outline

        Returns:
            dl.Item: Dataloop item containing the combined dialogue
        """
        logger.info("Combining dialogue segments")

        prompt_item = dl.PromptItem.from_json(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        segment_text = last_message.get("content", [])[0].get("text", None)
        if segment_text is None:
            raise ValueError("No segment text found in the prompt item.")

        # search for all the segment dialogues
        filters = dl.Filters()
        filters.add(field='dir', values=item.dir)
        filters.add(field='metadata.system.mimetype', values='*json*')
        items = item.dataset.items.list(filters=filters).all()

        # combine the segment dialogues
        combined_dialogue = ""
        for item in items:
            segment_text = item.metadata.get("user", {}).get("podcast", {}).get("segment_text", None)
            if segment_text is None:
                raise ValueError("No segment text found in the prompt item.")
            combined_dialogue += segment_text
        dialogue_item = item.dataset.items.upload(
            combined_dialogue,
            remote_name=f"{item.filename}_prompt6_combined_dialogue",
            remote_path=item.dir,
            overwrite=True,
        )
        return dialogue_item

    @staticmethod
    def create_convo_json(item: dl.Item, dialogue: str) -> dl.Item:
        """
        Convert the dialogue into structured Conversation format.

        Args:
            item (dl.Item): Dataloop item containing the dialogue
            dialogue (str): Combined dialogue text
            llm_manager (LLMManager): Manager for LLM interactions

        Returns:
            dl.Item: Dataloop item containing the structured conversation

        Formats the dialogue into a structured conversation format with proper speaker
        attribution and timing information.
        """
        logging.info("Formatting final conversation")

        item_metadata = item.metadata
        podcast_metadata = item_metadata.get("user", {}).get("podcast", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", "Alice")
        speaker_2_name = podcast_metadata.get("speaker_2_name", "Will")

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
                            "speaker": {"type": "string", "enum": ["speaker-1", "speaker-2"]},
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
            prompt_key='1',
        )

        new_name = f"{Path(item.name).stem}_prompt7_convo_json"
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)

        return new_item

    @staticmethod
    def create_final_conversation(item: dl.Item, dialogue: str) -> dl.Item:
        """
        Create a final conversation from the dialogue.
        """
        filters = dl.Filters()
        filters.add(field='dir', values=item.dir)
        filters.add(field='metadata.system.mimetype', values='*json*')
        items = item.dataset.items.list(filters=filters).all()

        for item in items:
            # Process the conversation JSON
            prompt_item = dl.PromptItem.from_json(item)
            messages = prompt_item.to_messages()
            last_message = messages[-1]
            conversation_json_str = last_message.get("content", [])[0].get("text", None)

            if conversation_json_str is None:
                raise ValueError(f"No conversation JSON found in the conversation segment. Check item {item.id}.")

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
