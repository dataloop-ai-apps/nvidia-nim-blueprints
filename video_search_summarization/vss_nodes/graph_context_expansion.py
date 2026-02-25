import logging
import json

import dtlpy as dl

logger = logging.getLogger('vss-nodes.graph-context-expansion')


class GraphContextExpansion(dl.BaseServiceRunner):

    def run(self, item: dl.Item, context: dl.Context) -> dl.Item:
        """
        Expand retrieval results with graph context. Reads the nearest items
        from the retriever's prompt item metadata, fetches their entity/relationship
        metadata and adjacent chunks (temporal context), then assembles an enriched
        context string and adds it to the prompt item.

        Replicates NVIDIA's graph traversal (entity_search, next_chunk, bfs tools).
        """
        logger.info(f"Expanding graph context for prompt item: {item.id}")

        node_config = context.node.metadata.get('customNodeConfig', {})
        graph_dir = node_config.get('graph_dir', '/.graph/entities')
        temporal_window = node_config.get('temporal_window', 1)

        prompt_item = dl.PromptItem.from_item(item=item)

        nearest_items_meta = item.metadata.get('user', {}).get('nearest_items', [])
        if not nearest_items_meta:
            logger.warning(f"No nearest items found in prompt item {item.id}")
            return item

        dataset_id = node_config.get('retrieval_dataset_id', None)
        if dataset_id:
            dataset = dl.datasets.get(dataset_id=dataset_id)
        else:
            dataset = item.dataset

        enriched_contexts = []
        for nearest_meta in nearest_items_meta:
            nearest_item_id = nearest_meta.get('item_id') or nearest_meta.get('id')
            if not nearest_item_id:
                continue

            try:
                nearest_item = dataset.items.get(item_id=nearest_item_id)
            except Exception as e:
                logger.warning(f"Failed to fetch nearest item {nearest_item_id}: {e}")
                continue

            chunk_context = self._build_chunk_context(
                item=nearest_item,
                dataset=dataset,
                temporal_window=temporal_window,
                graph_dir=graph_dir
            )
            enriched_contexts.append(chunk_context)

        if enriched_contexts:
            graph_context_str = "\n\n---\n\n".join(enriched_contexts)
            self._append_context_to_prompt(prompt_item, graph_context_str)
            prompt_item.to_item(item=item)
            item = dl.items.get(item_id=item.id)
            logger.info(f"Added graph context ({len(graph_context_str)} chars) to prompt item")

        return item

    def _build_chunk_context(self, item: dl.Item, dataset: dl.Dataset,
                             temporal_window: int, graph_dir: str) -> str:
        """Build enriched context string for a single retrieved chunk."""
        user_meta = item.metadata.get('user', {})
        entities = user_meta.get('entities', [])
        relationships = user_meta.get('relationships', [])
        source_type = user_meta.get('source_type', 'unknown')
        origin_video = item.metadata.get('origin_video_name', 'unknown')
        chunk_index = user_meta.get('chunk_index', None)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                local_path = item.download(local_path=tmp_dir)
                with open(local_path, 'r', encoding='utf-8') as f:
                    chunk_text = f.read()
            except Exception:
                chunk_text = "(content unavailable)"

        parts = [f"[{source_type}] from {origin_video}"]
        if chunk_index is not None:
            parts[0] += f" (chunk {chunk_index})"
        parts.append(chunk_text)

        if entities:
            entity_descs = []
            for ent in entities:
                ent_name = ent.get('name', '')
                ent_type = ent.get('type', '')
                ent_item_id = ent.get('entity_item_id', '')
                desc = self._fetch_entity_description(dataset, ent_item_id, graph_dir)
                entity_descs.append(f"  - {ent_type}: {ent_name}" + (f" ({desc})" if desc else ""))
            parts.append("Entities:\n" + "\n".join(entity_descs))

        if relationships:
            rel_strs = [
                f"  - {r.get('source', '')} --[{r.get('type', '')}]--> {r.get('target', '')}"
                for r in relationships
            ]
            parts.append("Relationships:\n" + "\n".join(rel_strs))

        if chunk_index is not None and temporal_window > 0:
            adjacent_texts = self._fetch_adjacent_chunks(
                dataset=dataset,
                origin_video=origin_video,
                chunk_index=chunk_index,
                window=temporal_window
            )
            if adjacent_texts:
                parts.append("Adjacent chunks:\n" + "\n".join(
                    f"  [chunk {idx}]: {text[:200]}..." if len(text) > 200 else f"  [chunk {idx}]: {text}"
                    for idx, text in adjacent_texts
                ))

        return "\n".join(parts)

    @staticmethod
    def _fetch_entity_description(dataset: dl.Dataset, entity_item_id: str, graph_dir: str) -> str:
        if not entity_item_id:
            return ''
        try:
            entity_item = dataset.items.get(item_id=entity_item_id)
            return entity_item.metadata.get('user', {}).get('description', '')
        except Exception:
            return ''

    @staticmethod
    def _fetch_adjacent_chunks(dataset: dl.Dataset, origin_video: str,
                               chunk_index: int, window: int) -> list:
        """Fetch text items from adjacent chunks based on chunk_index and origin_video_name."""
        results = []
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            target_idx = chunk_index + offset
            if target_idx < 0:
                continue
            try:
                filters = dl.Filters()
                filters.add(field='metadata.origin_video_name', values=origin_video)
                filters.add(field='metadata.user.chunk_index', values=target_idx)
                pages = dataset.items.list(filters=filters)
                if pages.items_count > 0:
                    adj_item = pages.items[0]
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        local_path = adj_item.download(local_path=tmp_dir)
                        with open(local_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                    results.append((target_idx, text))
            except Exception as e:
                logger.warning(f"Failed to fetch adjacent chunk {target_idx}: {e}")
        return sorted(results, key=lambda x: x[0])

    @staticmethod
    def _append_context_to_prompt(prompt_item: dl.PromptItem, context_str: str):
        """Append graph context as a system or user message to the prompt."""
        context_message = (
            "The following enriched context includes entity information, relationships, "
            "and adjacent temporal chunks from the video analysis graph. "
            "Use this to provide accurate, evidence-based answers with timestamps.\n\n"
            + context_str
        )
        for prompt in prompt_item.prompts:
            existing_content = []
            for msg in prompt.messages:
                if msg.get('role') == 'user':
                    existing_content = msg.get('content', [])
                    break

            existing_content.append({
                'mimetype': dl.PromptType.TEXT,
                'value': context_message
            })
            break
