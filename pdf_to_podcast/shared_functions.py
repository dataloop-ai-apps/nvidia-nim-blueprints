import os
import re
import json
import dotenv
import logging
import dtlpy as dl

from io import BytesIO
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict
from elevenlabs.client import ElevenLabs

from pdf_to_podcast.monologue_prompts import FinancialSummaryPrompts
from pdf_to_podcast.podcast_prompts import PodcastPrompts
from pdf_to_podcast.podcast_types import Conversation, PodcastOutline

# Load environment variables from .env file
dotenv.load_dotenv(".env")

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")

# Default voices configuration
DEFAULT_VOICE_1 = "EXAVITQu4vr4xnSDxMaL"
DEFAULT_VOICE_2 = "bIHbv24MWmeRgasZH58o"
DEFAULT_VOICE_MAPPING = {"speaker-1": DEFAULT_VOICE_1, "speaker-2": DEFAULT_VOICE_2}
DEFAULT_SPEAKER_1_NAME = "Alice"
DEFAULT_SPEAKER_2_NAME = "Will"


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

    def process_file(
        self, input_file: str, output_file: str, voice_mapping: dict = None
    ):
        """Process a JSON file containing dialogue or monologue"""
        # Use default voice mapping if none provided
        voice_mapping = voice_mapping or DEFAULT_VOICE_MAPPING

        try:
            # Read and parse the input JSON file
            with open(
                os.path.join(os.path.dirname(__file__), input_file),
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            # Extract dialogue entries
            dialogue = data.get("dialogue", [])
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
            with open(output_file, "wb") as f:
                f.write(combined_audio)

            logger.info(f"Successfully created audio file: {output_file}")

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise


class SharedServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def prepare_and_summarize_pdf(
        item: dl.Item,
        progress: dl.Progress,
        context: dl.Context,
        monologue: bool,
        focus: str = None,
        with_references: bool = None,
        duration: int = None,
        speaker_1_name: str = None,
        speaker_2_name: str = None,
    ):
        """
        Prepare the PDF file into a prompt item with the text to be processed

        Args:
            item (dl.Item): One of the child text items of the parent item
            monologue (bool): Whether to generate a monologue or a podcast
            progress (dl.Progress): The progress object to update the user
            context (dl.Context): The context object to access the item
            focus (str): The focus of the summary
            with_references (bool): Whether to include references in the summary
            duration (int): The duration of a dialogue podcast

        Returns:
            dl.Item: The prompt item with the text to be processed
        """
        logger.info(f"Preparing and summarizing PDF {item.id}")

        try:
            parent_item = dl.items.get(
                item_id=item.metadata.get("user", {}).get("original_item_id")
            )
        except Exception as e:
            logger.info(
                f"No parent item id key found for item {item.id}. Using input item as parent."
            )
            parent_item = item
        logger.info(f"Parent item {parent_item.filename} {parent_item.id}")

        # gather all text together
        pdf_text = SharedServiceRunner._collect_text_items(parent_item)
        pdf_name = Path(parent_item.name).stem

        if speaker_1_name is None:
            speaker_1_name = DEFAULT_SPEAKER_1_NAME
        if speaker_2_name is None:
            speaker_2_name = DEFAULT_SPEAKER_2_NAME
        if with_references is None:
            with_references = False

        # upload text to dataloop item
        text_filename = f"{pdf_name}_text.txt"
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(pdf_text)
        text_item = parent_item.dataset.items.upload(
            local_path=text_filename,
            remote_name=text_filename,
            remote_path=SharedServiceRunner._get_hidden_dir(item=parent_item),
            overwrite=True,
            item_metadata={
                "user": {
                    "parentItemId": parent_item.id,
                    "podcast": {"original_item_name": pdf_name},
                }
            },
        )
        logger.info(f"Uploaded pdf text item {text_item.id}")

        if monologue is True:
            template = FinancialSummaryPrompts.get_template("monologue_summary_prompt")
        else:
            template = PodcastPrompts.get_template("podcast_summary_prompt")
        llm_prompt = template.render(text=pdf_text)

        new_name = f"{pdf_name}_{'monologue_' if monologue is True else 'podcast_'}prompt1_summary"
        new_item_metadata = item.metadata.get("user", {})
        new_item_metadata.update(
            {
                "podcast": {
                    "pdf_id": parent_item.id,
                    "pdf_name": pdf_name,
                    "focus": focus,
                    "monologue": monologue,
                    "with_references": with_references,
                    "speaker_1_name": speaker_1_name,
                    "speaker_2_name": speaker_2_name,
                },
            }
        )
        if duration is not None:
            new_item_metadata["podcast"]["duration"] = duration
        new_item = SharedServiceRunner._create_and_upload_prompt_item(
            dataset=parent_item.dataset,
            item_name=new_name,
            prompt=llm_prompt,
            remote_dir=SharedServiceRunner._get_hidden_dir(item=parent_item),
            item_metadata={"user": new_item_metadata},
        )

        logger.info(
            f"Successfully created prompt item for {pdf_name} in new item {new_item.id}"
        )

        actions = ["monologue", "dialogue"]
        progress.update(action=actions[0] if monologue is True else actions[1])

        return new_item

    @staticmethod
    def create_final_json(
        item: dl.Item, progress: dl.Progress, context: dl.Context
    ) -> dl.Item:
        """
        Check the final conversation JSON and make sure all strings are unescaped

        Args:
            item (dl.Item): Dataloop item containing the dialogue

        Returns:
            new_item (dl.Item): Dataloop item containing the structured conversation JSON
        """
        logger.info("Formatting final conversation")

        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        pdf_name = podcast_metadata.get("pdf_name")

        conversation_json_str = SharedServiceRunner._get_last_message(item=item)
        if conversation_json_str is None:
            raise ValueError(
                f"No conversation JSON found in the conversation segment item {item.id}."
            )

        if isinstance(conversation_json_str, str):
            # Extract JSON from possible LLM preamble
            extracted = SharedServiceRunner._extract_json_string(conversation_json_str)
            try:
                conversation_json = json.loads(extracted)
            except json.JSONDecodeError as e:
                logger.warning(f"Conversation JSON parse failed: {e}. Attempting repair...")
                repaired = SharedServiceRunner._repair_conversation_json(extracted)
                if repaired is not None:
                    conversation_json = json.loads(repaired)
                    logger.info("Successfully repaired truncated conversation JSON.")
                else:
                    raise
        else:
            conversation_json = conversation_json_str

        # Remove unnecessary escaping from dialogue text entries
        if "dialogue" in conversation_json:
            for entry in conversation_json["dialogue"]:
                if "text" in entry and isinstance(entry["text"], str):
                    # First, handle escaped quotes
                    entry["text"] = entry["text"].replace('\\"', '"')
                    # Then handle any remaining escaped slashes
                    entry["text"] = entry["text"].replace("\\\\", "\\")
                    # Finally, handle any remaining single slashes
                    entry["text"] = entry["text"].replace("\\", "")
                    # Clean up any remaining unicode escapes
                    entry["text"] = SharedServiceRunner._unescape_unicode_string(
                        entry["text"]
                    )
        final_conversation = Conversation.model_validate(conversation_json)

        # upload the final conversation
        new_name = f"{Path(pdf_name).stem}_final_transcript.json"
        json_path = Path.cwd() / new_name
        with open(json_path, "w", encoding="utf-8") as f:
            json_file = final_conversation.model_dump_json(indent=2)
            f.write(json_file)

        new_item = item.dataset.items.upload(
            local_path=str(json_path),
            remote_name=new_name,
            remote_path=SharedServiceRunner._get_hidden_dir(item=item),
            overwrite=True,
            item_metadata=item.metadata,
        )
        return new_item

    @staticmethod
    def generate_audio(
        item: dl.Item,
        progress,
        context,
        voice_mapping: dict = None,
        output_file: str = None,
    ):
        """
        Generate audio from the conversation JSON file

        Args:
            item (dl.Item): Dataloop JSON item containing the conversation
            voice_mapping (dict): A dictionary mapping speaker names to voice IDs
            output_file (str): The name of the output audio file

        Returns:
            str: Path to the generated audio file
        """
        podcast_metadata = SharedServiceRunner._get_podcast_metadata(item)
        pdf_name = podcast_metadata.get("pdf_name")
        monologue = podcast_metadata.get("monologue")
        original_item = dl.items.get(item_id=item.metadata.get("user", {}).get("original_item_id", None))
        if original_item is None:
            raise ValueError(f"No original item id found in the final transcript item {item.id}. Please check that item was prepared correctly.")

        if monologue is True:
            output_file_name = Path(pdf_name).stem + "_monologue.mp3"
        else:
            output_file_name = Path(pdf_name).stem + "_podcast.mp3"

        if output_file is None:
            # Create output directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(__file__), "output")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, output_file_name)

        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required")

        converter = TTSConverter(api_key=api_key)

        # Download and process the conversation JSON from the last response
        conversation_json = item.download(save_locally=False).read().decode("utf-8")
        if conversation_json is None:
            raise ValueError("No conversation JSON found in the prompt item.")

        try:
            conversation_json = json.loads(conversation_json)

            # Create a temporary JSON file for the converter
            temp_file = os.path.join(
                os.path.dirname(__file__), "temp_conversation.json"
            )
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(conversation_json, f)

            # Process the file and generate audio
            converter.process_file(temp_file, output_file, voice_mapping)

            # Clean up temporary file
            os.remove(temp_file)

        except Exception as e:
            logger.error(f"Error processing conversation JSON: {e} from item {item.id}")
            raise

        mp3_item = item.dataset.items.upload(
            output_file,
            remote_name=output_file_name,
            remote_path=original_item.dir,
            overwrite=True,
            item_metadata=item.metadata,
        )
        logger.info(f"Successfully uploaded audio file: {mp3_item.id}")
        return mp3_item

    @staticmethod
    def _collect_text_items(item: dl.Item) -> str:
        """
        Collect the text from the PDF file

        Child items are found in the dataset by searching for the item id in the metadata

        Args:
            item (dl.Item): The oroginal PDF file to be processed

        Returns:
            str: The text from the PDF file
        """
        filters = dl.Filters()
        filters.add(field='hidden', values=True)
        filters.add(field="metadata.user.original_item_id", values=item.id)
        filters.sort_by(field="name", value=dl.FiltersOrderByDirection.ASCENDING)
        pages = item.dataset.items.list(filters=filters)

        pdf_text = ""
        for child_item in pages.all():
            # check that the item is a text item and the end of the string isn't summary.txt
            if "text" in child_item.mimetype and not child_item.name.endswith("summary.txt"):
                buffer = child_item.download(save_locally=False)
                pdf_text += buffer.read().decode("utf-8")
        return pdf_text

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
    def _get_summary_from_id(summary_item_id: str) -> str:
        """
        Get the summary text from the summary item id.
        """
        summary_item = dl.items.get(item_id=summary_item_id)
        if summary_item is None:
            raise ValueError(f"Summary item not found for id: {summary_item_id}")
        if "text" not in summary_item.mimetype:
            raise ValueError(
                f"Summary item is not a text file for id: {summary_item_id}"
            )
        text = summary_item.download(save_locally=False).read().decode("utf-8")
        if text is None:
            raise ValueError(
                f"No text found in summary item {summary_item_id}. Please check that the item was prepared correctly."
            )
        return text

    @staticmethod
    def _extract_json_string(text: str) -> str:
        """
        Extract a JSON object or array string from text that may contain
        preamble or surrounding markdown fences (e.g. ```json ... ```).
        Skips JSON schema definitions ($defs) if the actual data follows.
        """
        # Try to strip markdown code fences first
        fence_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
        if fence_match:
            candidate = fence_match.group(1).strip()
            # If the fenced block is a JSON schema, look for another block
            if '"$defs"' not in candidate and '"$schema"' not in candidate:
                return candidate
            # Try to find a second fenced block with actual data
            remaining = text[fence_match.end():]
            second_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", remaining)
            if second_match:
                return second_match.group(1).strip()

        # Find all top-level JSON objects in the text
        candidates = []
        depth = 0
        start = None
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
                    start = None

        # Prefer a candidate that looks like actual data (not a JSON schema)
        for candidate in candidates:
            if '"$defs"' not in candidate and '"$schema"' not in candidate:
                return candidate

        # Fallback: return the first candidate, or the last one, or original text
        if candidates:
            return candidates[0]

        # Try array extraction as fallback
        first_bracket = text.find("[")
        if first_bracket != -1:
            end = text.rfind("]")
            if end != -1:
                return text[first_bracket : end + 1]

        return text

    @staticmethod
    def _repair_outline_json(text: str) -> Optional[str]:
        """
        Attempt to repair truncated or malformed outline JSON from an LLM.

        Handles common issues:
        1. Model outputs a segments array [...] instead of {title, segments} object
        2. JSON is truncated (missing closing brackets/braces)
        3. Last segment is incomplete after truncation repair

        Returns the repaired JSON string, or None if repair is not possible.
        """
        text = text.strip()
        was_bare_array = False

        # --- Case 1: model output a bare array of segments ---
        if text.startswith("["):
            was_bare_array = True
            text = '{"title": "Untitled Podcast", "segments": ' + text + "}"

        # --- Case 2: try to close truncated JSON ---
        repaired = SharedServiceRunner._close_truncated_json(text)
        if repaired is None:
            return None

        # --- Case 3: validate and trim incomplete last segment ---
        # Try parsing as-is
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return None

        # Ensure it's an object with segments
        if not isinstance(data, dict):
            return None
        if "segments" not in data and was_bare_array:
            return None

        # Add default title if missing
        if "title" not in data:
            data["title"] = "Untitled Podcast"

        segments = data.get("segments", [])
        if isinstance(segments, list) and len(segments) > 0:
            # Validate each segment; drop the last one if it's incomplete
            valid_segments = []
            for seg in segments:
                if (
                    isinstance(seg, dict)
                    and "section" in seg
                    and "topics" in seg
                    and "duration" in seg
                    and "references" in seg
                ):
                    valid_segments.append(seg)
                else:
                    # Incomplete segment — only keep it if it's not the last one
                    # (non-last incomplete segments indicate a deeper problem)
                    if seg is not segments[-1]:
                        logger.warning(
                            f"Dropping incomplete segment in the middle: {seg}"
                        )
            if len(valid_segments) < len(segments):
                logger.info(
                    f"Trimmed {len(segments) - len(valid_segments)} incomplete "
                    f"segment(s) from truncated outline (kept {len(valid_segments)})."
                )
            if not valid_segments:
                return None
            data["segments"] = valid_segments

        return json.dumps(data)

    @staticmethod
    def _close_truncated_json(text: str) -> Optional[str]:
        """
        Attempt to close truncated JSON by appending missing brackets/braces.
        Returns valid JSON string or None.
        """
        text = text.strip()

        # Already valid?
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        # Count unmatched braces/brackets
        stack = []
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == '\\' and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in ('{', '['):
                stack.append(ch)
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()

        if not stack:
            return None  # Balanced but still invalid — can't help

        # If we're inside a string, close it first
        if in_string:
            text += '"'

        # Close all open braces/brackets in reverse order
        closing = ""
        for opener in reversed(stack):
            closing += '}' if opener == '{' else ']'

        # Try with and without trailing comma removal
        for t in [text, text.rstrip().rstrip(',')]:
            candidate = t + closing
            try:
                json.loads(candidate)
                logger.info(
                    f"Repaired truncated outline JSON by closing {len(stack)} "
                    f"unclosed bracket(s)/brace(s)."
                )
                return candidate
            except json.JSONDecodeError:
                continue

        return None

    @staticmethod
    def _repair_conversation_json(text: str) -> Optional[str]:
        """
        Attempt to repair truncated conversation JSON from an LLM.

        The expected structure is: {"scratchpad": "...", "dialogue": [{text, speaker}, ...]}
        Handles:
        1. Truncated JSON (unclosed brackets/braces/strings)
        2. Incomplete last dialogue entry
        """
        repaired = SharedServiceRunner._close_truncated_json(text)
        if repaired is None:
            return None

        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        # Add default scratchpad if missing
        if "scratchpad" not in data:
            data["scratchpad"] = ""

        dialogue = data.get("dialogue", [])
        if isinstance(dialogue, list) and len(dialogue) > 0:
            # Drop incomplete dialogue entries (missing text or speaker)
            valid_entries = []
            for entry in dialogue:
                if (
                    isinstance(entry, dict)
                    and "text" in entry
                    and "speaker" in entry
                    and isinstance(entry["text"], str)
                    and len(entry["text"].strip()) > 0
                ):
                    valid_entries.append(entry)

            if len(valid_entries) < len(dialogue):
                logger.info(
                    f"Trimmed {len(dialogue) - len(valid_entries)} incomplete "
                    f"dialogue entry/entries from truncated conversation "
                    f"(kept {len(valid_entries)})."
                )
            if not valid_entries:
                return None
            data["dialogue"] = valid_entries

        return json.dumps(data)

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
        if isinstance(outline_dict, dict):
            # If the dict is a JSON schema (e.g. model echoed back the schema), reject it
            if "$defs" in outline_dict or "$schema" in outline_dict:
                raise ValueError(
                    f"LLM returned the JSON schema definition instead of actual outline data "
                    f"for item {outline_item.id}. The model may need to be re-run."
                )
            outline_dict = json.dumps(outline_dict)
        elif isinstance(outline_dict, str):
            extracted = SharedServiceRunner._extract_json_string(outline_dict)
            if extracted != outline_dict:
                logger.warning(
                    "Outline response contained non-JSON preamble, extracted JSON from LLM response."
                )
            # Check if extracted content is a JSON schema instead of actual data
            if '"$defs"' in extracted or '"$schema"' in extracted:
                logger.warning(
                    "LLM returned the JSON schema definition instead of outline data. "
                    "Checking if actual data follows the schema in the response..."
                )
                # Try to find actual data after the schema in the original text
                second_pass = SharedServiceRunner._extract_json_string(outline_dict)
                if second_pass != extracted and '"$defs"' not in second_pass:
                    extracted = second_pass
                    logger.info("Found actual outline data after the schema definition.")
                else:
                    raise ValueError(
                        f"LLM returned the JSON schema definition instead of actual outline data "
                        f"for item {outline_item.id}. The model may need to be re-run."
                    )
            outline_dict = extracted

        # Attempt to parse; if it fails, try to repair the JSON
        try:
            outline = PodcastOutline.model_validate_json(outline_dict)
            return outline
        except Exception as first_error:
            logger.warning(f"Initial outline parse failed: {first_error}")
            repaired = SharedServiceRunner._repair_outline_json(outline_dict)
            if repaired is not None and repaired != outline_dict:
                logger.info("Attempting parse with repaired JSON...")
                try:
                    outline = PodcastOutline.model_validate_json(repaired)
                    logger.info("Successfully parsed repaired outline JSON.")
                    return outline
                except Exception as repair_error:
                    logger.warning(f"Repaired JSON also failed: {repair_error}")
            # Re-raise the original error
            raise first_error

    @staticmethod
    def _get_podcast_metadata(item: dl.Item) -> Dict:
        """
        Get the metadata from the item.
        """
        metadata = item.metadata.get("user", {}).get("podcast", None)
        if metadata is None:
            raise ValueError(
                f"No podcast metadata found in the prompt item. Please check that item was prepared correctly."
            )
        if metadata.get("pdf_name") is None:
            raise ValueError(
                f"No pdf_name found in the prompt item. Please check that item was prepared correctly."
            )

        # take all the metadata fields and check whether they exist, if not set default values
        podcast_metadata = {
            "pdf_name": metadata.get("pdf_name"),
            "monologue": metadata.get("monologue", False),
            "focus": metadata.get("focus", None),
            "with_references": metadata.get("with_references", False),
            "speaker_1_name": metadata.get("speaker_1_name", DEFAULT_SPEAKER_1_NAME),
            "speaker_2_name": metadata.get("speaker_2_name", DEFAULT_SPEAKER_2_NAME),
            "duration": metadata.get("duration", 10),
            "references": metadata.get("references", None),
            "summary_item_id": metadata.get("summary_item_id", None),
            "outline_item_id": metadata.get("outline_item_id", None),
            "segment_idx": metadata.get("segment_idx", None),
            "total_segments": metadata.get("total_segments", None),
        }

        return podcast_metadata

    @staticmethod
    def _get_hidden_dir(item: dl.Item) -> str:
        """
        Get the hidden directory for the item.

        If the item is already in the hidden directory, return the item.dir.
        Otherwise, return the hidden subdirectory.
        """
        HIDDEN_DIR = ".pdf2podcast"
        if item.dir == "/":
            hidden_dir = f"/{HIDDEN_DIR}"
        else:
            if HIDDEN_DIR in item.dir:
                hidden_dir = item.dir
            else:
                hidden_dir = f"{item.dir}/{HIDDEN_DIR}"
        return hidden_dir

    @staticmethod
    def _create_and_upload_prompt_item(
        dataset: dl.Dataset,
        item_name: str,
        prompt: str,
        remote_dir: str,
        item_metadata: dict,
        overwrite: bool = True,
    ) -> dl.Item:
        """
        Create a prompt item and upload it to the item.
        """
        prompt_item = dl.PromptItem(name=item_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": prompt}]}
        )
        prompt_item = dataset.items.upload(
            prompt_item,
            remote_name=item_name,
            remote_path=remote_dir,
            overwrite=overwrite,
            item_metadata=item_metadata,
        )
        return prompt_item

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
    item = dl.items.get(item_id="")

    progress = dl.Progress()
    context = dl.Context()

    p_item = SharedServiceRunner.create_final_json(
        item=item,
        progress=progress,
        context=context,
    )

    print(p_item)
    print(p_item.platform_url)
    print()
