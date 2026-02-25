import logging
import os
import subprocess
import tempfile

import dtlpy as dl

logger = logging.getLogger('vss-nodes.audio-extraction')

DEFAULT_OUTPUT_DIR = '/audio_prompt_items'


class AudioExtraction(dl.BaseServiceRunner):

    def extract_audio(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Extract audio from a sub-video item and create a PromptItem with the audio data.
        Matches NVIDIA's 16kHz mono WAV format.
        """
        logger.info(f"Extracting audio from sub-video: {item.id} ({item.name})")

        node_config = context.node.metadata.get('customNodeConfig', {})
        output_dir = node_config.get('output_dir', DEFAULT_OUTPUT_DIR)
        sample_rate = node_config.get('sample_rate', 16000)

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = item.download(local_path=tmp_dir)
            audio_path = os.path.join(tmp_dir, 'audio.wav')

            has_audio = self._extract_audio_ffmpeg(
                video_path=video_path,
                audio_path=audio_path,
                sample_rate=sample_rate
            )

            dataset = item.dataset
            base_name = os.path.splitext(item.name)[0]
            prompt_item_name = f"{base_name}-audio"

            prompt_item = dl.PromptItem(name=prompt_item_name)

            if has_audio and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                audio_element = {
                    'mimetype': 'audio/wav',
                    'value': audio_path
                }
                prompt_item.prompts.append(
                    dl.Prompt(key='audio', messages=[
                        {'role': 'user', 'content': [audio_element]}
                    ])
                )
            else:
                logger.warning(f"No audio track found in {item.name}, creating placeholder prompt")
                prompt_item.prompts.append(
                    dl.Prompt(key='audio', messages=[
                        {'role': 'user', 'content': [
                            {'mimetype': dl.PromptType.TEXT, 'value': 'No audio available for this video segment.'}
                        ]}
                    ])
                )

            uploaded = dataset.items.upload(
                local_path=prompt_item,
                remote_path=output_dir,
            )
            logger.info(f"Uploaded audio PromptItem: {uploaded.id}")

            uploaded.metadata['origin_video_name'] = item.metadata.get('origin_video_name', item.name)
            run_time = item.metadata.get('time', None)
            if run_time is not None:
                uploaded.metadata['time'] = run_time
            sub_videos_intervals = item.metadata.get('user', {}).get('sub_videos_intervals', None)
            if sub_videos_intervals is not None:
                uploaded.metadata['user'] = uploaded.metadata.get('user', {})
                uploaded.metadata['user']['sub_videos_intervals'] = sub_videos_intervals

            uploaded = uploaded.update()
            return uploaded

    @staticmethod
    def _extract_audio_ffmpeg(video_path: str, audio_path: str, sample_rate: int = 16000) -> bool:
        """Run FFmpeg to extract mono WAV audio. Returns False if video has no audio stream."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a',
                 '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path],
                capture_output=True, text=True, timeout=30
            )
            if 'audio' not in result.stdout:
                logger.info(f"No audio stream found in {video_path}")
                return False

            subprocess.run(
                ['ffmpeg', '-y', '-i', video_path, '-vn',
                 '-ar', str(sample_rate), '-ac', '1', '-f', 'wav', audio_path],
                capture_output=True, text=True, timeout=120, check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio extraction failed: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("FFmpeg not found. Ensure FFmpeg is installed in the runtime image.")
            return False
