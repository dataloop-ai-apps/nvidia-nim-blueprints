import logging
import dtlpy as dl

from pdf_to_podcast.monologue_prompts import FinancialSummaryPrompts
from pdf_to_podcast.podcast_prompts import PodcastPrompts

logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class ServiceRunner(dl.BaseServiceRunner):
    def prepare_and_summarize_pdf(
        item: dl.Item, monologue: bool, progress: dl.Progress, context: dl.Context, guide: str = None
    ):
        item_metadata = item.metadata
        summary = item_metadata.get("user", {}).get("summary", None)
        if monologue is True:
            template = FinancialSummaryPrompts.get_template("monologue_summary_prompt")
        else:
            template = PodcastPrompts.get_template("podcast_summary_prompt")
        llm_prompt = template.render(text=summary)

        new_name = f"{item.filename}_prompt"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]},  # role default is user
            prompt_key='1',
        )

        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
        logger.info(f"Successfully created and uploaded summary prompt item for {item.filename} ID {item.id}")
        return new_item


class MonologueService(dl.BaseServiceRunner):
    def monologue_generate_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate an outline from the pdf text summary

        Args:
            item (dl.Item): Dataloop item containing the podcast summary
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """

        # retrieve the summary from the last prompt message
        # item = dl.items.get(item_id="6758139d45821d442ba1f6e1") # DEBUG
        # buffer = item.download(save_locally=False)
        # try:
        #     text = buffer.read().decode('utf-8')
        # except Exception as e:
        #     logger.error(f"Error decoding the item: {e}. Please check the item is a prompt item and the prompt contains text.")
        #     raise e

        # prep prompt item to get the summary
        prompt_item = dl.PromptItem.from_json(item)

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        guide = podcast_metadata.get("guide", None)

        messages = prompt_item.to_messages()
        last_message = messages[-1]
        summary = last_message.get("content", [])[0].get("text", None)

        if summary is None:
            raise ValueError("No summary found in the prompt item.")

        # generate the outline
        documents = [f"Document: {item.filename}\n{summary}"]

        template = FinancialSummaryPrompts.get_template("monologue_multi_doc_synthesis_prompt")
        llm_prompt = template.render(
            focus_instructions=guide if guide is not None else None, documents="\n\n".join(documents)
        )

        prompt = dl.Prompt(key="2")  # "2_summary_to_outline")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)

        prompt_item.prompts.append(prompt)

        return prompt_item

    def monologue_generate_monologue(
        item: dl.Item, progress: dl.Progress, context: dl.Context, prompt_guide: str = None
    ):
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
        prompt_item = dl.PromptItem.from_json(item)

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        guide = podcast_metadata.get("guide", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", None)

        # get the outline from the last prompt
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        outline = last_message.get("content", [])[0].get("text", None)
        if outline is None:
            raise ValueError("No outline found in the prompt item.")

        # get the summary docs context, from first prompt
        summary = messages[0].get("content", [])[0].get("text", None)
        if summary is None:
            try:
                summary = podcast_metadata.get("summary")
            except Exception as e:
                raise ValueError("No summary found in the prompt item.")

        documents = [f"Document: {item.filename}\n{summary}"]

        # generate the monologue
        template = FinancialSummaryPrompts.get_template("monologue_transcript_prompt")
        llm_prompt = template.render(
            raw_outline=outline,
            documents=documents,
            focus=guide if guide else "key financial metrics and performance indicators",
            speaker_1_name=speaker_1_name,
        )

        # add the prompt to the prompt item
        prompt = dl.Prompt(key="3")  # "3_outline_to_monologue")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)

        prompt_item.prompts.append(prompt)
        return prompt_item

    def monologue_create_convo_json(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Create a final conversation from the monologue in JSON format

        Args:
            item (dl.Item): Dataloop item containing the monologue
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """

        # prep prompt item to get the monologue
        prompt_item = dl.PromptItem.from_json(item)

        # get the monologue from the last prompt
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        monologue = last_message.get("content", [])[0].get("text", None)
        if monologue is None:
            raise ValueError("No monologue found in the prompt item.")

        # create the final conversation in JSON format
        template = FinancialSummaryPrompts.get_template("monologue_convo_json_prompt")
        llm_prompt = template.render(monologue=monologue)

        prompt = dl.Prompt(key="4")  # "4_monologue_to_convo_json")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)

        prompt_item.prompts.append(prompt)

        return prompt_item


class DialogueFlow(dl.BaseServiceRunner):
    def podcast_generate_raw_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate initial raw outline from summarized PDFs.
        """
        # TODO add support for context pdfs
        # documents = []
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

        prompt_item = dl.PromptItem.from_json(item)

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        guide = podcast_metadata.get("guide", None)
        duration = podcast_metadata.get("duration", "10 minutes")

        template = PodcastPrompts.get_template("podcast_multi_pdf_outline_prompt")
        prompt = template.render(
            total_duration=podcast_duration,
            focus_instructions=request.guide if request.guide else None,
            documents="\n\n".join(documents),
        )

    async def podcast_generate_structured_outline(
        raw_outline: str,
        request: TranscriptionRequest,
        llm_manager: LLMManager,
        prompt_tracker: PromptTracker,
        job_id: str,
        job_manager: JobStatusManager,
        logger: logging.Logger,
    ) -> PodcastOutline:
        """
        Convert raw outline text to structured PodcastOutline format.

        Args:
            raw_outline (str): Raw outline text to structure
            request (TranscriptionRequest): Original transcription request
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses
            job_id (str): ID for tracking job progress
            job_manager (JobStatusManager): Manages job status updates
            logger (logging.Logger): Logger for tracking progress

        Returns:
            PodcastOutline: Structured outline following the PodcastOutline schema

        Uses JSON schema validation to ensure the outline follows the required structure
        and only references valid PDF filenames.
        """
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Converting raw outline to structured format")

        # Force the model to only reference valid filenames
        valid_filenames = [pdf.filename for pdf in request.pdf_metadata]
        schema = PodcastOutline.model_json_schema()
        schema["$defs"]["PodcastSegment"]["properties"]["references"]["items"] = {
            "type": "string",
            "enum": valid_filenames,
        }

        schema = PodcastOutline.model_json_schema()
        template = PodcastPrompts.get_template("podcast_multi_pdf_structured_outline_prompt")
        prompt = template.render(
            outline=raw_outline,
            schema=json.dumps(schema, indent=2),
            valid_filenames=[pdf.filename for pdf in request.pdf_metadata],
        )
        outline: Dict = await llm_manager.query_async(
            "json", [{"role": "user", "content": prompt}], "outline", json_schema=schema
        )
        prompt_tracker.track("outline", prompt, llm_manager.model_configs["json"].name, json.dumps(outline))
        return PodcastOutline.model_validate(outline)

    async def podcast_process_segment(
        segment: Any, idx: int, request: TranscriptionRequest, llm_manager: LLMManager, prompt_tracker: PromptTracker
    ) -> tuple[str, str]:
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
        # Get reference content if it exists
        text_content = []
        if segment.references:
            for ref in segment.references:
                # Find matching PDF metadata by filename
                pdf = next((pdf for pdf in request.pdf_metadata if pdf.filename == ref), None)
                if pdf:
                    text_content.append(pdf.markdown)

        # Choose template based on whether we have references
        template_name = "podcast_prompt_with_references" if text_content else "podcast_prompt_no_references"
        template = PodcastPrompts.get_template(template_name)

        # Prepare prompt parameters
        prompt_params = {
            "duration": segment.duration,
            "topic": segment.section,
            "angles": "\n".join([topic.title for topic in segment.topics]),
        }

        # Add text content if we have references
        if text_content:
            prompt_params["text"] = "\n\n".join(text_content)

        prompt = template.render(**prompt_params)

        response: AIMessage = await llm_manager.query_async(
            "iteration", [{"role": "user", "content": prompt}], f"segment_{idx}"
        )

        prompt_tracker.track(
            f"segment_transcript_{idx}", prompt, llm_manager.model_configs["iteration"].name, response.content
        )

        return f"segment_transcript_{idx}", response.content

    async def podcast_process_segments(
        outline: PodcastOutline,
        request: TranscriptionRequest,
        llm_manager: LLMManager,
        prompt_tracker: PromptTracker,
        job_id: str,
        job_manager: JobStatusManager,
        logger: logging.Logger,
    ) -> Dict[str, str]:
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
        # Create tasks for processing each segment
        segment_tasks: List[Coroutine] = []
        for idx, segment in enumerate(outline.segments):
            job_manager.update_status(
                job_id, JobStatus.PROCESSING, f"Processing segment {idx + 1}/{len(outline.segments)}: {segment.section}"
            )

            task = podcast_process_segment(segment, idx, request, llm_manager, prompt_tracker)
            segment_tasks.append(task)

        # Process all segments in parallel
        results = await asyncio.gather(*segment_tasks)

        # Convert results to dictionary
        return dict(results)

    async def podcast_generate_dialogue_segment(
        segment: Any,
        idx: int,
        segment_text: str,
        request: TranscriptionRequest,
        llm_manager: LLMManager,
        prompt_tracker: PromptTracker,
    ) -> Dict[str, str]:
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
        topics_text = "\n".join(
            [
                f"- {topic.title}\n" + "\n".join([f"  * {point.description}" for point in topic.points])
                for topic in segment.topics
            ]
        )

        # Generate dialogue using template
        template = PodcastPrompts.get_template("podcast_transcript_to_dialogue_prompt")
        prompt = template.render(
            text=segment_text,
            duration=segment.duration,
            descriptions=topics_text,
            speaker_1_name=request.speaker_1_name,
            speaker_2_name=request.speaker_2_name,
        )

        # Query LLM for dialogue
        dialogue_response = await llm_manager.query_async(
            "reasoning", [{"role": "user", "content": prompt}], f"segment_dialogue_{idx}"
        )

        # Track prompt and response
        prompt_tracker.track(
            f"segment_dialogue_{idx}", prompt, llm_manager.model_configs["reasoning"].name, dialogue_response.content
        )

        return {"section": segment.section, "dialogue": dialogue_response.content}

    async def podcast_generate_dialogue(
        segments: Dict[str, str],
        outline: PodcastOutline,
        request: TranscriptionRequest,
        llm_manager: LLMManager,
        prompt_tracker: PromptTracker,
        job_id: str,
        job_manager: JobStatusManager,
        logger: logging.Logger,
    ) -> List[Dict[str, str]]:
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
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Generating dialogue")

        # Create tasks for generating dialogue for each segment
        dialogue_tasks = []
        for idx, segment in enumerate(outline.segments):
            segment_name = f"segment_transcript_{idx}"
            seg_response = segments.get(segment_name)

            if not seg_response:
                logger.warning(f"Segment {segment_name} not found in segment transcripts")
                continue

            # Update prompt tracker with segment response
            segment_text = seg_response
            prompt_tracker.update_result(segment_name, segment_text)

            # Update status
            job_manager.update_status(
                job_id, JobStatus.PROCESSING, f"Converting segment {idx + 1}/{len(outline.segments)} to dialogue"
            )

            task = podcast_generate_dialogue_segment(segment, idx, segment_text, request, llm_manager, prompt_tracker)
            dialogue_tasks.append(task)

        # Process all dialogues in parallel
        dialogues = await asyncio.gather(*dialogue_tasks)

        return list(dialogues)

    async def podcast_combine_dialogues(
        segment_dialogues: List[Dict[str, str]],
        outline: PodcastOutline,
        llm_manager: LLMManager,
        prompt_tracker: PromptTracker,
        job_id: str,
        job_manager: JobStatusManager,
        logger: logging.Logger,
    ) -> str:
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
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Combining dialogue segments")

        # Start with the first segment's dialogue
        current_dialogue = segment_dialogues[0]["dialogue"]
        prompt_tracker.update_result("segment_dialogue_0", current_dialogue)

        # Iteratively combine with subsequent segments
        for idx in range(1, len(segment_dialogues)):
            job_manager.update_status(
                job_id,
                JobStatus.PROCESSING,
                f"Combining segment {idx + 1}/{len(segment_dialogues)} with existing dialogue",
            )

            next_section = segment_dialogues[idx]["dialogue"]
            prompt_tracker.update_result(f"segment_dialogue_{idx}", next_section)
            current_section = segment_dialogues[idx]["section"]

            template = PodcastPrompts.get_template("podcast_combine_dialogues_prompt")
            prompt = template.render(
                outline=outline.model_dump_json(),
                dialogue_transcript=current_dialogue,
                next_section=next_section,
                current_section=current_section,
            )

            combined: AIMessage = await llm_manager.query_async(
                "iteration", [{"role": "user", "content": prompt}], f"combine_dialogues_{idx}"
            )

            prompt_tracker.track(
                f"combine_dialogues_{idx}", prompt, llm_manager.model_configs["iteration"].name, combined.content
            )

            current_dialogue = combined.content

        return current_dialogue

    async def podcast_create_final_conversation(
        dialogue: str,
        request: TranscriptionRequest,
        llm_manager: LLMManager,
        prompt_tracker: PromptTracker,
        job_id: str,
        job_manager: JobStatusManager,
        logger: logging.Logger,
    ) -> Conversation:
        """
        Convert the dialogue into structured Conversation format.

        Args:
            dialogue (str): Combined dialogue text
            request (TranscriptionRequest): Original transcription request
            llm_manager (LLMManager): Manager for LLM interactions
            prompt_tracker (PromptTracker): Tracks prompts and responses
            job_id (str): ID for tracking job progress
            job_manager (JobStatusManager): Manages job status updates
            logger (logging.Logger): Logger for tracking progress

        Returns:
            Conversation: Structured conversation following the Conversation schema

        Formats the dialogue into a structured conversation format with proper speaker
        attribution and timing information.
        """
        job_manager.update_status(job_id, JobStatus.PROCESSING, "Formatting final conversation")

        schema = Conversation.model_json_schema()
        template = PodcastPrompts.get_template("podcast_dialogue_prompt")
        prompt = template.render(
            speaker_1_name=request.speaker_1_name,
            speaker_2_name=request.speaker_2_name,
            text=dialogue,
            schema=json.dumps(schema, indent=2),
        )

        # We accumulate response as it comes in then cast
        conversation_json: Dict = await llm_manager.stream_async(
            "json", [{"role": "user", "content": prompt}], "create_final_conversation", json_schema=schema
        )

        # Ensure all strings are unescaped
        if "dialogues" in conversation_json:
            for entry in conversation_json["dialogues"]:
                if "text" in entry:
                    entry["text"] = unescape_unicode_string(entry["text"])

        prompt_tracker.track(
            "create_final_conversation", prompt, llm_manager.model_configs["json"].name, json.dumps(conversation_json)
        )

        return Conversation.model_validate(conversation_json)


if __name__ == "__main__":
    dl.setenv("prod")
    item = dl.items.get(item_id="6758139d45821d442ba1f6e1")
    ServiceRunner.prepare_and_summarize_pdf(item, True, None, None)
    MonologueService.monologue_generate_outline(item, None, None)
    MonologueService.monologue_generate_monologue(item, None, None)
    MonologueService.monologue_create_convo_json(item, None, None)
