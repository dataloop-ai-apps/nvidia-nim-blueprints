import logging
import os
import tempfile

import dtlpy as dl

logger = logging.getLogger('vss-nodes.asr-inference')

NVCF_GRPC_URI = 'grpc.nvcf.nvidia.com:443'
PARAKEET_RNNT_FUNCTION_ID = '71203149-d3b7-4460-8231-1be2543a1fca'


class ASRInference(dl.BaseServiceRunner):
    """
    ASR inference node using NVIDIA Riva ASR gRPC hosted endpoint.

    Connects to the hosted NVIDIA Riva ASR service at grpc.nvcf.nvidia.com:443
    via the nvidia-riva-client library. When a proper ASR NIM DPK becomes
    available, replace this custom node with an ml-type node in the pipeline
    template. The surrounding nodes (Audio Extract, Transcript to Text) require
    no changes since this node preserves the PromptItem in/out contract.
    """

    def __init__(self):
        self._asr_service = None
        self._current_function_id = None

    def _get_asr_service(self, function_id: str, grpc_uri: str):
        import riva.client

        if self._asr_service is not None and self._current_function_id == function_id:
            return self._asr_service

        api_key = os.environ.get('NGC_API_KEY', '')
        if not api_key:
            raise ValueError("Missing NGC_API_KEY environment variable.")

        auth = riva.client.Auth(
            use_ssl=True,
            uri=grpc_uri,
            metadata_args=[
                ["function-id", function_id],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        self._asr_service = riva.client.ASRService(auth)
        self._current_function_id = function_id
        return self._asr_service

    def predict(self, item: dl.Item, context: dl.Context) -> dl.Item:
        logger.info(f"ASR inference on prompt item: {item.id} ({item.name})")

        node_config = context.node.metadata.get('customNodeConfig', {})
        function_id = node_config.get('asr_function_id', PARAKEET_RNNT_FUNCTION_ID)
        grpc_uri = node_config.get('asr_grpc_uri', NVCF_GRPC_URI)
        language_code = node_config.get('language_code', 'en-US')

        for annotation in item.annotations.list():
            annotation.delete()

        prompt_item = dl.PromptItem.from_item(item=item)

        audio_source = None
        for prompt in prompt_item.prompts:
            for element in prompt.elements:
                if element['mimetype'] == dl.PromptType.AUDIO:
                    audio_source = element['value']
                    break
            if audio_source:
                break

        if audio_source is None:
            transcript = "Audio transcript not available."
            logger.warning(f"No audio element found in prompt item {item.id}")
        else:
            transcript = self._transcribe(
                audio_source=audio_source,
                function_id=function_id,
                grpc_uri=grpc_uri,
                language_code=language_code,
            )

        prompt_item.add(
            message={
                'role': 'assistant',
                'content': [{'mimetype': dl.PromptType.TEXT, 'value': transcript}],
            }
        )

        item = dl.items.get(item_id=item.id)
        logger.info(f"ASR transcript written to prompt item {item.id} ({len(transcript)} chars)")
        return item

    def _transcribe(self, audio_source: str, function_id: str, grpc_uri: str, language_code: str) -> str:
        import riva.client

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                wav_path = self._download_audio(audio_source, tmp_dir)
                if wav_path is None:
                    return "Audio transcript not available."

                asr_service = self._get_asr_service(function_id, grpc_uri)

                config = riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                    verbatim_transcripts=False,
                    language_code=language_code,
                )
                riva.client.add_audio_file_specs_to_config(config, wav_path)

                with open(wav_path, 'rb') as f:
                    audio_bytes = f.read()

                response = asr_service.offline_recognize(audio_bytes, config)

                transcript_parts = []
                for result in response.results:
                    if result.alternatives:
                        transcript_parts.append(result.alternatives[0].transcript)

                transcript = ' '.join(transcript_parts).strip()
                return transcript if transcript else "Audio transcript not available."

        except Exception as e:
            logger.error(f"ASR inference failed: {e}", exc_info=True)
            return "Audio transcript not available."

    @staticmethod
    def _download_audio(audio_source: str, tmp_dir: str) -> str:
        """Download audio to a local WAV file and return the path."""
        try:
            dest_path = os.path.join(tmp_dir, 'audio.wav')

            if os.path.isfile(audio_source):
                import shutil
                shutil.copy2(audio_source, dest_path)
                return dest_path

            if 'dataloop.ai' in audio_source and '/items/' in audio_source:
                item_id = audio_source.split("/stream")[0].split("/items/")[-1]
                dl.items.get(item_id=item_id).download(local_path=dest_path)
                return dest_path

            if audio_source.startswith('http'):
                import requests
                resp = requests.get(audio_source, timeout=60)
                resp.raise_for_status()
                with open(dest_path, 'wb') as f:
                    f.write(resp.content)
                return dest_path

            logger.warning(f"Unrecognized audio source format: {audio_source}")
            return None

        except Exception as e:
            logger.error(f"Failed to download audio: {e}")
            return None
