import logging
import os

import dtlpy as dl

logger = logging.getLogger('vss-nodes.prepare-graph-prompt')

DEFAULT_ENTITY_TYPES = '["Person", "Vehicle", "Location", "Object"]'

DEFAULT_EXTRACTION_INSTRUCTION = (
    "Extract entities and relationships from the following video description or audio transcript.\n"
    "Entity types: {entity_types}\n"
    "Return ONLY valid JSON in this exact format:\n"
    '{{"entities": [{{"type": "...", "name": "...", "description": "..."}}], '
    '"relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}\n\n'
    "Text:\n{text_content}"
)


class PrepareGraphPrompt(dl.BaseServiceRunner):

    def run(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Read a text item (VLM caption or audio transcript) and build a PromptItem
        with an entity extraction instruction for the Graph LLM.
        """
        logger.info(f"Preparing graph extraction prompt for item: {item.id} ({item.name})")

        node_config = context.node.metadata.get('customNodeConfig', {})
        entity_types = node_config.get('entity_types', DEFAULT_ENTITY_TYPES)
        output_dir = node_config.get('output_dir', '/graph_prompt_items')

        text_content = self._read_text_item(item)
        if not text_content or not text_content.strip():
            logger.warning(f"Empty text content in item {item.id}, skipping graph extraction")
            text_content = "(empty)"

        extraction_prompt = DEFAULT_EXTRACTION_INSTRUCTION.format(
            entity_types=entity_types,
            text_content=text_content
        )

        base_name = os.path.splitext(item.name)[0]
        prompt_name = f"{base_name}-graph-extract"

        prompt_item = dl.PromptItem(name=prompt_name)
        prompt_item.prompts.append(
            dl.Prompt(key='graph_extraction', messages=[
                {'role': 'user', 'content': [
                    {'mimetype': dl.PromptType.TEXT, 'value': extraction_prompt}
                ]}
            ])
        )

        dataset = item.dataset
        uploaded = dataset.items.upload(
            local_path=prompt_item,
            remote_path=output_dir,
        )

        uploaded.metadata['user'] = uploaded.metadata.get('user', {})
        uploaded.metadata['user']['source_text_item_id'] = item.id
        uploaded.metadata['user']['source_type'] = item.metadata.get('user', {}).get('source_type', 'unknown')

        origin_video_name = item.metadata.get('origin_video_name', None)
        if origin_video_name is not None:
            uploaded.metadata['origin_video_name'] = origin_video_name
        run_time = item.metadata.get('time', None)
        if run_time is not None:
            uploaded.metadata['time'] = run_time
        chunk_index = item.metadata.get('user', {}).get('chunk_index', None)
        if chunk_index is not None:
            uploaded.metadata['user']['chunk_index'] = chunk_index

        uploaded = uploaded.update()
        logger.info(f"Uploaded graph extraction prompt: {uploaded.id}")
        return uploaded

    @staticmethod
    def _read_text_item(item: dl.Item) -> str:
        """Download and read the text content of a text item."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = item.download(local_path=tmp_dir)
            with open(local_path, 'r', encoding='utf-8') as f:
                return f.read()
