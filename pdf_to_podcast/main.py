import json
import logging
import dtlpy as dl
import os
from elevenlabs.client import ElevenLabs
from pydantic import BaseModel
from typing import Dict, Any, List, Coroutine, Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from pdf_to_podcast.monologue_prompts import FinancialSummaryPrompts
from pdf_to_podcast.podcast_prompts import PodcastPrompts, PodcastOutline

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")

# Default voices configuration
DEFAULT_VOICE_1 = os.getenv("DEFAULT_VOICE_1", "EXAVITQu4vr4xnSDxMaL")
DEFAULT_VOICE_2 = os.getenv("DEFAULT_VOICE_2", "bIHbv24MWmeRgasZH58o")
DEFAULT_VOICE_MAPPING = {"speaker-1": DEFAULT_VOICE_1, "speaker-2": DEFAULT_VOICE_2}


class DialogueEntry(BaseModel):
    text: str
    speaker: Optional[str] = "speaker-1"
    voice_id: Optional[str] = None


class TTSConverter:
    def __init__(self, api_key: str = None):
        """Initialize the TTS converter with ElevenLabs API key"""
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key is required")

        self.client = ElevenLabs(api_key=self.api_key)

    def _convert_text(self, text: str, voice_id: str) -> bytes:
        """Convert a single piece of text to speech"""
        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_monolingual_v1",
                output_format="mp3_44100_128",
                voice_settings={"stability": 0.5, "similarity_boost": 0.75},
            )
            return b"".join(chunk for chunk in audio_stream)
        except Exception as e:
            logger.error(f"Error converting text to speech: {e}")
            raise

    def process_file(self, input_file: str, output_file: str, voice_mapping: dict = None):
        """Process a JSON file containing dialogue or monologue"""
        # Use default voice mapping if none provided
        voice_mapping = voice_mapping or DEFAULT_VOICE_MAPPING

        try:
            # Read and parse the input JSON file
            with open(os.path.join(os.path.dirname(__file__), input_file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract dialogue entries
            dialogue = data.get('dialogue', [])
            if not dialogue:
                # If no dialogue found, treat as monologue
                dialogue = [{"text": data.get("text", ""), "speaker": "speaker-1"}]

            # Convert all entries to audio and combine
            combined_audio = b""
            total_entries = len(dialogue)

            for i, entry in enumerate(dialogue, 1):
                # Create DialogueEntry instance for validation
                entry_model = DialogueEntry(**entry)

                # Determine which voice to use
                voice_id = (
                    entry_model.voice_id
                    if entry_model.voice_id
                    else voice_mapping.get(entry_model.speaker, DEFAULT_VOICE_1)
                )

                logger.info(f"Processing entry {i}/{total_entries}")
                audio_chunk = self._convert_text(entry_model.text, voice_id)
                combined_audio += audio_chunk

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save the combined audio to file
            with open(output_file, 'wb') as f:
                f.write(combined_audio)

            logger.info(f"Successfully created audio file: {output_file}")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise


class ServiceRunner(dl.BaseServiceRunner):
    @staticmethod
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

    @staticmethod
    def generate_audio(item: dl.Item, voice_mapping: dict = None):
        """
        Generate audio from the conversation JSON file

        Args:
            item (dl.Item): Dataloop item containing the conversation JSON
            output_file (str): Output MP3 file path (defaults to output/output.mp3)
            voice_mapping (dict): Optional mapping of speakers to voice IDs

        Returns:
            str: Path to the generated audio file
        """
        if output_file is None:
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            output_file = os.path.join("output", "output.mp3")

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        converter = TTSConverter(api_key=api_key)

        # Download and process the conversation JSON
        buffer = item.download(save_locally=True)
        try:
            conversation_json = json.loads(buffer.read().decode('utf-8'))

            # Create a temporary JSON file for the converter
            temp_file = os.path.join(os.path.dirname(__file__), "temp_conversation.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_json, f)

            # Process the file and generate audio
            converter.process_file(temp_file, output_file, voice_mapping)

            # Clean up temporary file
            os.remove(temp_file)

        except Exception as e:
            logger.error(f"Error processing conversation JSON: {e}")
            raise

        mp3_item = item.dataset.items.upload(output_file, remote_name=output_file, remote_path=item.dir, overwrite=True)
        logger.info(f"Successfully uploaded audio file: {mp3_item.id}")
        return mp3_item

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        duration = podcast_metadata.get("duration", 10)
        summary = podcast_metadata.get("summary", None)

        documents = [f"Document: {item.filename}\n{summary}"]

        template = PodcastPrompts.get_template("podcast_multi_pdf_outline_prompt")
        llm_prompt = template.render(
            total_duration=duration,
            focus_instructions=guide if guide is not None else None,
            documents="\n\n".join(documents),
        )

        prompt = dl.Prompt(key="1")  # "1_raw_outline")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)

        prompt_item.prompts.append(prompt)

        return prompt_item

    @staticmethod
    def podcast_generate_structured_outline(
        item: dl.Item, progress: dl.Progress, context: dl.Context, prompt_guide: str = None
    ):
        """
        Convert raw outline text to structured PodcastOutline format.

        Args:
            item (dl.Item): Dataloop item containing the raw outline
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object
            prompt_guide (str): Guide for the prompt

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
        # valid_filenames = [pdf.filename for pdf in request.pdf_metadata]
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

        prompt = dl.Prompt(key="2")  # "2_structured_outline")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)
        prompt_item.prompts.append(prompt)

        return prompt_item

    @staticmethod
    def _podcast_process_segment(
        item: dl.Item, segment: Any, idx: int, request: TranscriptionRequest
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

        prompt_item = dl.PromptItem.from_json(item)
        item_metadata = item.metadata
        podcast_metadata = item_metadata.get("user", {}).get("podcast", None)
        guide = podcast_metadata.get("guide", None)
        duration = podcast_metadata.get("duration", 10)
        summary = podcast_metadata.get("summary", None)

        pdf_text = prompt_item.prompts[0].elements[0].value
        text_content = [f"Document: {item.filename}\n{pdf_text}"]

        # TODO add support for context pdfs
        # # Get reference content if it exists
        # text_content = []
        # if segment.references:
        #     for ref in segment.references:
        #         # Find matching PDF metadata by filename
        #         pdf = next((pdf for pdf in request.pdf_metadata if pdf.filename == ref), None)
        #         if pdf:
        #             text_content.append(pdf.markdown)

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
    def podcast_process_segments(
        item: dl.Item, outline: PodcastOutline, progress: dl.Progress, context: dl.Context
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
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {}
            for idx, segment in enumerate(outline.segments):
                task = DialogueFlow._podcast_process_segment(item, segment, idx, request)
                futures[idx] = executor.submit(task)

            for idx, future in futures.items():
                results.append(future.result())

        # Convert results to dictionary
        return dict(results)

    @staticmethod
    def podcast_generate_dialogue_segment(segment: Any, idx: int, segment_text: str) -> Dict[str, str]:
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
    def podcast_generate_dialogue(segments: Dict[str, str], outline: PodcastOutline) -> List[Dict[str, str]]:
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

        # TODO fix to use threadpool
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

                future = executor.submit(DialogueFlow.podcast_generate_dialogue_segment, segment, idx, segment_text)
                futures[idx] = future

            # Process all dialogues in parallel and preserve order
            for idx, future in futures.items():
                dialogue_segments[idx] = future.result()
        # Convert dictionary to ordered list
        dialogue_segments = [dialogue_segments[i] for i in sorted(dialogue_segments.keys())]

        return dialogue_segments

    @staticmethod
    def podcast_combine_dialogues(segment_dialogues: List[Dict[str, str]], outline: PodcastOutline) -> str:
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

            # combined: AIMessage = await llm_manager.query_async(
            #     "iteration", [{"role": "user", "content": prompt}], f"combine_dialogues_{idx}"
            # )

            # prompt_tracker.track(
            #     f"combine_dialogues_{idx}", prompt, llm_manager.model_configs["iteration"].name, combined.content
            # )

            # current_dialogue = combined.content

        return current_dialogue

    @staticmethod
    def podcast_create_convo_json(item: dl.Item, dialogue: str) -> dl.Item:
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
        speaker_1_name = podcast_metadata.get("speaker_1_name", DEFAULT_SPEAKER_1_NAME)
        speaker_2_name = podcast_metadata.get("speaker_2_name", DEFAULT_SPEAKER_2_NAME)

        schema = Conversation.model_json_schema()
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

        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)

        return new_item

    @staticmethod
    def podcast_create_final_conversation(dir_item: dl.Item, dialogue: str) -> dl.Item:
        """
        Create a final conversation from the dialogue.
        """
        filters = dl.Filters()
        filters.add(field='dir', values=dir_item.dir)
        filters.add(field='metadata.system.mimetype', values='*json*')
        items = dir_item.dataset.items.list(filters=filters).all()

        # Ensure all strings are unescaped
        if "dialogues" in conversation_json:
            for entry in conversation_json["dialogues"]:
                if "text" in entry:
                    entry["text"] = unescape_unicode_string(entry["text"])

        prompt_tracker.track(
            "create_final_conversation", prompt, llm_manager.model_configs["json"].name, json.dumps(conversation_json)
        )

        return podcast_create_convo_json(item, dialogue)

    @staticmethod
    def _unescape_unicode_string(s: str) -> str:
        """
        Convert escaped Unicode sequences to actual Unicode characters.

        Args:
            s (str): String potentially containing escaped Unicode sequences

        Returns:
            str: String with Unicode sequences converted to actual characters

        Example:
            >>> unescape_unicode_string("Hello\\u2019s World")
            "Hello's World"
        """
        # This handles both raw strings (with extra backslashes) and regular strings
        return s.encode("utf-8").decode("unicode-escape")


if __name__ == "__main__":
    dl.setenv("prod")
    item = dl.items.get(item_id="6758139d45821d442ba1f6e1")
    progress = dl.Progress()
    context = dl.Context()

    processed_item = ServiceRunner.prepare_and_summarize_pdf(item, True, progress, context)

    try:
        output_file = ServiceRunner.generate_audio(processed_item)
        logger.info(f"Successfully generated audio file: {output_file}")
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise
