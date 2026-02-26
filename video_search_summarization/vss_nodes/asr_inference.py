import logging
import os
import base64

import dtlpy as dl
import requests

logger = logging.getLogger('vss-nodes.asr-inference')


class ASRInference(dl.BaseServiceRunner):
    """
    Placeholder ASR inference node. Calls an ASR API endpoint and writes the
    transcript as an assistant response in the PromptItem.

    When a proper ASR NIM DPK becomes available, replace this custom node
    with an ml-type node in the pipeline template. The surrounding nodes
    (Audio Extract and Transcript to Text) require no changes since this
    node preserves the PromptItem in/out contract.
    """

    def predict(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Reads audio from PromptItem, calls ASR endpoint, writes transcript back.
        """
        logger.info(f"ASR inference on prompt item: {item.id} ({item.name})")

        node_config = context.node.metadata.get('customNodeConfig', {})
        asr_base_url = node_config.get('asr_base_url', 'https://integrate.api.nvidia.com/v1')
        asr_model_name = node_config.get('asr_model_name', 'nvidia/parakeet-ctc-1.1b-asr')
        language_code = node_config.get('language_code', 'en-US')

        for annotation in item.annotations.list():
            annotation.delete()

        prompt_item = dl.PromptItem.from_item(item=item)

        audio_source = None
        is_placeholder = False
        for prompt in prompt_item.prompts:
            for element in prompt.elements:
                if element['mimetype'] == dl.PromptType.AUDIO:
                    audio_source = element['value']
                    break
                if element['mimetype'] == dl.PromptType.TEXT and 'No audio available' in str(element.get('value', '')):
                    is_placeholder = True
            if audio_source or is_placeholder:
                break

        if audio_source is None:
            transcript = "Audio transcript not available."
            logger.warning(f"No audio element found in prompt item {item.id}")
        else:
            transcript = self._call_asr(
                audio_source=audio_source,
                base_url=asr_base_url,
                model_name=asr_model_name,
                language_code=language_code
            )

        prompt_item.add(
            message={
                'role': 'assistant',
                'content': [{'mimetype': dl.PromptType.TEXT, 'value': transcript}]
            }
        )

        item = dl.items.get(item_id=item.id)
        logger.info(f"ASR transcript written to prompt item {item.id} ({len(transcript)} chars)")
        return item

    @staticmethod
    def _call_asr(audio_source: str, base_url: str, model_name: str, language_code: str) -> str:
        """
        Call the ASR API. Handles both local file paths and remote URLs
        (PromptItem audio references become URLs after upload to the platform).
        """
        try:
            api_key = os.environ.get('NVIDIA_API_KEY', os.environ.get('NGC_API_KEY', ''))

            if os.path.isfile(audio_source):
                with open(audio_source, 'rb') as f:
                    audio_bytes = f.read()
            elif 'dataloop.ai' in audio_source and '/items/' in audio_source:
                item_id = audio_source.split("/stream")[0].split("/items/")[-1]
                audio_bytes = dl.items.get(item_id=item_id).download(save_locally=False).getvalue()
            elif audio_source.startswith('http'):
                resp = requests.get(audio_source, timeout=60)
                resp.raise_for_status()
                audio_bytes = resp.content
            else:
                logger.warning(f"Audio source is not a local file or URL: {audio_source}")
                return "Audio transcript not available."

            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            url = f"{base_url}/audio/transcriptions"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': model_name,
                'language': language_code,
                'audio': audio_b64,
            }

            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()
            transcript = result.get('text', result.get('transcript', ''))
            return transcript if transcript else "Audio transcript not available."

        except Exception as e:
            logger.error(f"ASR API call failed: {e}")
            return "Audio transcript not available."
