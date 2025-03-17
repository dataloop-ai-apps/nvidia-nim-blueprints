import os
import json
import logging
import dtlpy as dl

from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from pdf_to_podcast.shared_functions import SharedServiceRunner
from pdf_to_podcast.podcast_prompts import PodcastPrompts, PodcastOutline

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class DialogueServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def generate_raw_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate initial raw outline from summarized PDFs.
        """
        logger.info("Generating initial outline")

        # get the podcast metadata from the item
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        summary = podcast_metadata.get("summary", None)

        # get the summary from the last prompt annotation
        prompt_item = dl.PromptItem.from_json(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        summary = last_message.get("content", [])[0].get("text", None)
        if summary is None:
            raise ValueError("No summary found in the prompt item.")

        documents = [f"Document: {item.filename}\n{summary}"]
        # TODO support multiple pdfs as context
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
        new_name = f"{item.filename}_prompt1_raw_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
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
        new_name = f"{item.filename}_prompt2_structured_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
        return new_item

    @staticmethod
    def _process_segment(item: dl.Item, segment: Any, idx: int) -> tuple[str, str]:
        """
        Process a single outline segment to generate initial content.

        Args:
            segment (Any): Segment from the outline to process
            idx (int): Index of the segment
            request (TranscriptionRequest): Original transcription request
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses

        Returns:
            tuple[str, str]: Tuple of (segment_id, generated_content)

        Generates initial content for a segment, incorporating referenced PDF content
        if available. Uses different templates based on whether references exist.
        """
        prompt_item = dl.PromptItem.from_json(item)
        item_metadata = item.metadata
        podcast_metadata = item_metadata.get("user", {}).get("podcast", None)
        focus = podcast_metadata.get("focus", None)
        duration = podcast_metadata.get("duration", 10)
        summary = podcast_metadata.get("summary", None)

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

        return prompt_item

    @staticmethod
    def process_segments(item: dl.Item, progress: dl.Progress, context: dl.Context) -> Dict[str, str]:
        """
        Process all outline segments in parallel to generate initial content.

        Args:
            outline (PodcastOutline): Structured outline to process
            request (TranscriptionRequest): Original transcription request
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses
            job_id (str): ID for tracking job progress
            job_manager (JobStatusManager): Manages job status updates
            logger (logging.Logger): Logger for tracking progress

        Returns:
            Dict[str, str]: Dictionary mapping segment IDs to their generated content

        Creates tasks for processing each segment and executes them in parallel using
        asyncio.gather.
        """
        # create the outline item
        prompt_item = dl.PromptItem.from_json(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        outline_dict = last_message.get("content", [])[0].get("text", None)
        if outline_dict is None:
            raise ValueError("No outline found in the prompt item.")

        outline = PodcastOutline.model_validate_json(outline_dict)

        # Create tasks for processing each segment
        segment_tasks: List[Any] = []
        for idx, segment in enumerate(outline.segments):
            logger.info(f"Processing segment {idx + 1}/{len(outline.segments)}: {segment.section}")

            task = DialogueServiceRunner._process_segment(item, segment, idx)
            segment_tasks.append(task)

        # Process all segments in parallel
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {}
            for idx, segment in enumerate(outline.segments):
                task = DialogueServiceRunner._process_segment(item, segment, idx)
                futures[idx] = executor.submit(task)

            for idx, future in futures.items():
                results.append(future.result())

        # Convert results to dictionary
        return dict(results)

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

        prompt = dl.Prompt(key="4")  # "4_dialogue_segment")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)
        prompt_item.prompts.append(prompt)

        new_name = f"{item.filename}_prompt4_dialogue"
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
        # prompt item to produce the section name and the dialogue
        return new_item

    @staticmethod
    def generate_dialogue(segments: Dict[str, str], outline: PodcastOutline) -> List[Dict[str, str]]:
        """
        Generate dialogue for all segments in parallel.

        Args:
            segments (Dict[str, str]): Dictionary of segment IDs and their content
            outline (PodcastOutline): Structured outline
            request (TranscriptionRequest): Original transcription request
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses
            job_id (str): ID for tracking job progress
            job_manager (JobStatusManager): Manages job status updates
            logger (logging.Logger): Logger for tracking progress

        Returns:
            List[Dict[str, str]]: List of dictionaries containing section names and dialogues

        Creates tasks for generating dialogue for each segment and executes them in parallel.
        """
        logger.info("Generating dialogue")

        # Create tasks for generating dialogue for each segment
        dialogue_segments = {}
        with ThreadPoolExecutor() as executor:
            futures = {}
            for idx, segment in enumerate(outline.segments):
                segment_name = f"segment_transcript_{idx}"
                seg_response = segments.get(segment_name)

                if not seg_response:
                    logger.warning(f"Segment {segment_name} not found in segment transcripts")
                    continue

                # Update prompt tracker with segment response
                segment_text = seg_response

                # Update status
                logger.info(f"Converting segment {idx + 1}/{len(outline.segments)} to dialogue")

                future = executor.submit(DialogueServiceRunner.generate_dialogue_segment, segment, idx, segment_text)
                futures[idx] = future

            # Process all dialogues in parallel and preserve order
            for idx, future in futures.items():
                dialogue_segments[idx] = future.result()
        # Convert dictionary to ordered list
        dialogue_segments = [dialogue_segments[i] for i in sorted(dialogue_segments.keys())]

        return dialogue_segments

    @staticmethod
    def combine_dialogues(segment_dialogues: List[Dict[str, str]], outline: PodcastOutline) -> str:
        """
        Iteratively combine dialogue segments into a cohesive conversation.

        Args:
            segment_dialogues (List[Dict[str, str]]): List of segment dialogues
            outline (PodcastOutline): Structured outline
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses
            job_id (str): ID for tracking job progress
            job_manager (JobStatusManager): Manages job status updates
            logger (logging.Logger): Logger for tracking progress

        Returns:
            str: Combined dialogue text

        Iteratively combines dialogue segments, ensuring smooth transitions between sections.
        """
        logger.info("Combining dialogue segments")

        # Start with the first segment's dialogue
        current_dialogue = segment_dialogues[0]["dialogue"]

        # Iteratively combine with subsequent segments
        for idx in range(1, len(segment_dialogues)):
            next_section = segment_dialogues[idx]["dialogue"]
            current_section = segment_dialogues[idx]["section"]

            template = PodcastPrompts.get_template("podcast_combine_dialogues_prompt")
            llm_prompt = template.render(  # TODO streaming sections to the prompt
                outline=outline.model_dump_json(),
                dialogue_transcript=current_dialogue,
                next_section=next_section,
                current_section=current_section,
            )

        return current_dialogue

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

        prompt_item = dl.PromptItem(name=f"{item.filename}_prompt_json")
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]},  # role default is user
            prompt_key='1',
        )

        new_name = f"{item.filename}_prompt_convo_json"
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)

        return new_item

    @staticmethod
    def create_final_conversation(dir_item: dl.Item, dialogue: str) -> dl.Item:
        """
        Create a final conversation from the dialogue.
        """
        filters = dl.Filters()
        filters.add(field='dir', values=dir_item.dir)
        filters.add(field='metadata.system.mimetype', values='*json*')
        items = dir_item.dataset.items.list(filters=filters).all()

        # TODO fix
        conversation_json = items[0].metadata.get("user", {}).get("podcast", {}).get("conversation", None)

        # Ensure all strings are unescaped
        if "dialogues" in conversation_json:
            for entry in conversation_json["dialogues"]:
                if "text" in entry:
                    entry["text"] = SharedServiceRunner._unescape_unicode_string(entry["text"])

        return DialogueServiceRunner.create_convo_json(dir_item, dialogue)
