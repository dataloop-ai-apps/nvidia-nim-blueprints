import logging
import io
import os

import dtlpy as dl

logger = logging.getLogger('vss-nodes.transcript-to-text')

DEFAULT_OUTPUT_DIR = '/text_responses_dir'


class TranscriptToText(dl.BaseServiceRunner):

    def run(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Extract transcript from an ASR-processed PromptItem, create a text item.
        Mirrors the Prompt to Text node for the audio branch.
        """
        logger.info(f"Processing ASR prompt item: {item.id} ({item.name})")

        node_config = context.node.metadata.get('customNodeConfig', {})
        output_dir = node_config.get('output_dir', DEFAULT_OUTPUT_DIR)
        text_prefix = node_config.get('text_prefix', 'Audio transcript: ')
        source_type = node_config.get('source_type', 'audio_transcript')

        prompt_item = dl.PromptItem.from_item(item=item)

        response_texts = []
        for prompt in prompt_item.prompts:
            for msg in prompt.messages:
                if msg.get('role') == 'assistant':
                    for element in msg.get('content', []):
                        if element.get('mimetype') == dl.PromptType.TEXT:
                            response_texts.append(element['value'])

        if not response_texts:
            raise ValueError(f"No assistant response found in prompt item {item.id}")

        full_response = "\n\n".join(response_texts)
        if text_prefix:
            full_response = text_prefix + full_response

        logger.info(f"Extracted transcript ({len(full_response)} chars) from prompt item {item.id}")

        base_name = os.path.splitext(item.name)[0]
        text_item_name = f"{base_name}-transcript.txt"

        buffer = io.BytesIO()
        buffer.name = text_item_name
        buffer.write(full_response.encode('utf-8'))
        buffer.seek(0)

        dataset = item.dataset
        uploaded_item = dataset.items.upload(
            local_path=buffer,
            remote_path=output_dir,
        )
        logger.info(f"Uploaded transcript text item: {uploaded_item.id}")

        uploaded_item.metadata['user'] = uploaded_item.metadata.get('user', {})
        uploaded_item.metadata['user']['source_type'] = source_type

        origin_video_name = item.metadata.get('origin_video_name', None)
        if origin_video_name is not None:
            uploaded_item.metadata['origin_video_name'] = origin_video_name
        run_time = item.metadata.get('time', None)
        if run_time is not None:
            uploaded_item.metadata['time'] = run_time

        sub_videos_intervals = item.metadata.get('user', {}).get('sub_videos_intervals', None)
        if sub_videos_intervals is not None:
            uploaded_item.metadata['user']['sub_videos_intervals'] = sub_videos_intervals

        uploaded_item = uploaded_item.update()
        return uploaded_item
