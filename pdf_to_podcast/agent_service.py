"""
Main service for PDF-to-podcast conversion.

This service coordinates the PDF-to-podcast conversion process by managing jobs,
orchestrating LLM calls, and handling both monologue and dialogue podcast generation.
"""

from shared.api_types import ServiceType, JobStatus, TranscriptionRequest
from shared.podcast_types import Conversation, PodcastOutline
from pdf_to_podcast.dialogue_flow import (
    podcast_summarize_pdfs,
    podcast_generate_raw_outline,
    podcast_generate_structured_outline,
    podcast_process_segments,
    podcast_generate_dialogue,
    podcast_combine_dialogues,
    podcast_create_final_conversation,
)
from monologue_flow import (
    monologue_summarize_pdfs,
    monologue_generate_raw_outline,
    monologue_generate_monologue,
    monologue_create_final_conversation,
)
from shared.storage import StorageManager
from shared.llmmanager import LLMManager
from shared.job import JobStatusManager
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig
from opentelemetry.trace.status import StatusCode
import ujson as json
import os
import logging
from shared.prompt_tracker import PromptTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up OpenTelemetry instrumentation
telemetry = OpenTelemetryInstrumentation()
config = OpenTelemetryConfig(
    service_name="agent-service",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://jaeger:4317"),
    enable_redis=True,
    enable_requests=True,
)
telemetry.initialize(config)

# Initialize managers
job_manager = JobStatusManager(ServiceType.AGENT, telemetry=telemetry)
storage_manager = StorageManager(telemetry=telemetry)


async def process_transcription(job_id: str, request: TranscriptionRequest):
    """
    Main processing function for transcription requests.

    Handles both monologue and dialogue podcast generation workflows by coordinating
    multiple steps including PDF summarization, outline generation, and conversation creation.

    Args:
        job_id (str): Unique identifier for the transcription job
        request (TranscriptionRequest): Contains all parameters for the transcription including:
            - PDF metadata
            - Voice mapping
            - Speaker names
            - Duration target
            - Processing preferences

    Raises:
        Exception: If any step in the process fails, with error details in job status
    """
    with telemetry.tracer.start_as_current_span("agent.process_transcription") as span:
        try:
            # Initialize LLM manager and prompt tracker
            llm_manager = LLMManager(
                api_key=os.getenv("NVIDIA_API_KEY"), telemetry=telemetry, config_path=os.getenv("MODEL_CONFIG_PATH")
            )
            span.set_attribute("model_config_path", os.getenv("MODEL_CONFIG_PATH"))
            prompt_tracker = PromptTracker(job_id, request.userId, storage_manager)

            # Initialize processing
            job_manager.update_status(job_id, JobStatus.PROCESSING, "Initializing processing")

            if request.monologue:
                # Summarize PDFs
                summarized_pdfs = await monologue_summarize_pdfs(
                    request.pdf_metadata, job_id, llm_manager, prompt_tracker, job_manager, logger
                )

                # Generate raw outline
                raw_outline = await monologue_generate_raw_outline(
                    summarized_pdfs, request, llm_manager, prompt_tracker, job_id, job_manager
                )

                # Generate monologue
                monologue = await monologue_generate_monologue(
                    raw_outline, request, llm_manager, prompt_tracker, job_id, job_manager
                )

                # Create final conversation
                final_conversation = await monologue_create_final_conversation(
                    monologue, request, llm_manager, prompt_tracker, job_id, job_manager
                )

                # Store result
                job_manager.set_result_with_expiration(job_id, final_conversation.model_dump_json().encode(), ex=120)
                job_manager.update_status(job_id, JobStatus.COMPLETED, "Transcription completed successfully")

            else:
                # Summarize PDFs
                summarized_pdfs = await podcast_summarize_pdfs(
                    request.pdf_metadata, job_id, llm_manager, prompt_tracker, job_manager, logger
                )

                # Generate initial outline
                raw_outline = await podcast_generate_raw_outline(
                    summarized_pdfs, request, llm_manager, prompt_tracker, job_id, job_manager, logger
                )

                # Convert outline to structured format
                outline: PodcastOutline = await podcast_generate_structured_outline(
                    raw_outline, request, llm_manager, prompt_tracker, job_id, job_manager, logger
                )

                # Process segments in parallel
                segments = await podcast_process_segments(
                    outline, request, llm_manager, prompt_tracker, job_id, job_manager, logger
                )

                # Generate dialogues from segments in parallel
                segment_dialogues = await podcast_generate_dialogue(
                    segments, outline, request, llm_manager, prompt_tracker, job_id, job_manager, logger
                )

                # Combine transcripts iteratively
                combined_dialogues = await podcast_combine_dialogues(
                    segment_dialogues, outline, llm_manager, prompt_tracker, job_id, job_manager, logger
                )

                # Create final conversation by formatting as JSON
                final_conversation: Conversation = await podcast_create_final_conversation(
                    combined_dialogues, request, llm_manager, prompt_tracker, job_id, job_manager, logger
                )
                # Store result
                job_manager.set_result_with_expiration(job_id, final_conversation.model_dump_json().encode(), ex=120)
                job_manager.update_status(job_id, JobStatus.COMPLETED, "Transcription completed successfully")

        except Exception as e:
            span.set_status(StatusCode.ERROR, "transcription failed")
            span.record_exception(e)
            logger.error(f"Error processing job {job_id}: {str(e)}")
            job_manager.update_status(job_id, JobStatus.FAILED, str(e))
            raise


if __name__ == "__main__":
    # Example usage can be added here
    pass
