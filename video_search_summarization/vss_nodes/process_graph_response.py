import logging
import json
import io
import os

import dtlpy as dl

logger = logging.getLogger('vss-nodes.process-graph-response')

DEFAULT_GRAPH_DIR = '/.graph/entities'


class ProcessGraphResponse(dl.BaseServiceRunner):

    def run(self, item: dl.Item, context: dl.Context) -> list:
        """
        Parse the Graph LLM response, create/update entity items in /.graph/entities/,
        and update the source text item with graph metadata.

        Returns a list of items: [source_text_item, ...entity_items] so all flow
        through Clone + Embed downstream.
        """
        logger.info(f"Processing graph LLM response: {item.id} ({item.name})")

        node_config = context.node.metadata.get('customNodeConfig', {})
        graph_dir = node_config.get('graph_dir', DEFAULT_GRAPH_DIR)

        prompt_item = dl.PromptItem.from_item(item=item)

        llm_response_text = ''
        for prompt in prompt_item.prompts:
            for msg in prompt.messages:
                if msg.get('role') == 'assistant':
                    for element in msg.get('content', []):
                        if element.get('mimetype') == dl.PromptType.TEXT:
                            llm_response_text = element.get('value', '')
                            break

        entities, relationships = self._parse_extraction_response(llm_response_text)

        source_text_item_id = item.metadata.get('user', {}).get('source_text_item_id')
        if not source_text_item_id:
            logger.error(f"No source_text_item_id in prompt item {item.id}")
            return [item]

        dataset = item.dataset
        source_item = dataset.items.get(item_id=source_text_item_id)

        entity_items = []
        entity_refs = []
        for entity in entities:
            entity_item = self._upsert_entity(
                dataset=dataset,
                entity=entity,
                chunk_item_id=source_text_item_id,
                graph_dir=graph_dir
            )
            if entity_item:
                entity_items.append(entity_item)
                entity_refs.append({
                    'name': entity.get('name', ''),
                    'type': entity.get('type', ''),
                    'entity_item_id': entity_item.id
                })

        source_item.metadata['user'] = source_item.metadata.get('user', {})
        source_item.metadata['user']['entities'] = entity_refs
        source_item.metadata['user']['relationships'] = relationships
        source_item = source_item.update()
        logger.info(
            f"Updated source text item {source_item.id} with "
            f"{len(entity_refs)} entities, {len(relationships)} relationships"
        )

        result_items = [source_item] + entity_items
        return result_items

    @staticmethod
    def _parse_extraction_response(response_text: str) -> tuple:
        """Parse the LLM JSON response into entities and relationships lists."""
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start == -1 or end == 0:
                logger.warning(f"No JSON found in LLM response")
                return [], []
            json_str = response_text[start:end]
            data = json.loads(json_str)
            entities = data.get('entities', [])
            relationships = data.get('relationships', [])
            return entities, relationships
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse graph extraction JSON: {e}")
            return [], []

    @staticmethod
    def _upsert_entity(dataset: dl.Dataset, entity: dict,
                       chunk_item_id: str, graph_dir: str) -> dl.Item:
        """
        Check if entity exists in graph_dir by name+type, create or update.
        Uses a naming convention to reduce duplicates.
        """
        entity_name = entity.get('name', 'unknown').strip()
        entity_type = entity.get('type', 'unknown').strip()
        entity_desc = entity.get('description', '')

        safe_name = entity_name.lower().replace(' ', '_').replace('/', '_')[:50]
        safe_type = entity_type.lower().replace(' ', '_')[:20]
        item_name = f"{safe_type}_{safe_name}.json"

        try:
            filters = dl.Filters()
            filters.add(field='dir', values=graph_dir)
            filters.add(field='name', values=item_name)
            pages = dataset.items.list(filters=filters)

            if pages.items_count > 0:
                existing_item = pages.items[0]
                chunk_ids = existing_item.metadata.get('user', {}).get('chunk_ids', [])
                if chunk_item_id not in chunk_ids:
                    chunk_ids.append(chunk_item_id)
                    existing_item.metadata['user']['chunk_ids'] = chunk_ids

                    existing_desc = existing_item.metadata.get('user', {}).get('description', '')
                    if entity_desc and entity_desc not in existing_desc:
                        existing_item.metadata['user']['description'] = (
                            f"{existing_desc}; {entity_desc}" if existing_desc else entity_desc
                        )
                    existing_item = existing_item.update()
                    logger.info(f"Updated existing entity: {item_name}")
                return existing_item

            entity_data = {
                'name': entity_name,
                'type': entity_type,
                'description': entity_desc,
            }
            buffer = io.BytesIO()
            buffer.name = item_name
            buffer.write(json.dumps(entity_data, indent=2).encode('utf-8'))
            buffer.seek(0)

            new_item = dataset.items.upload(
                local_path=buffer,
                remote_path=graph_dir,
            )
            new_item.metadata['user'] = new_item.metadata.get('user', {})
            new_item.metadata['user']['entity_name'] = entity_name
            new_item.metadata['user']['entity_type'] = entity_type
            new_item.metadata['user']['description'] = entity_desc
            new_item.metadata['user']['chunk_ids'] = [chunk_item_id]
            new_item = new_item.update()
            logger.info(f"Created new entity: {item_name}")
            return new_item

        except Exception as e:
            logger.error(f"Failed to upsert entity {item_name}: {e}")
            return None
