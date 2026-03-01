import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
import dtlpy as dl
import logging
import json
import re
import os
import tempfile

logger = logging.getLogger("[GRAPH-RAG]")

# ====================================================================== #
#  add_chunk_to_graph accepts three input formats:                       #
#                                                                        #
#  1. Prompt item — LLM guided-JSON response.                            #
#     user message  = chunk text                                         #
#     assistant msg = JSON matching GRAPH_EXTRACTION_SCHEMA               #
#                                                                        #
#  2. JSON file item:                                                    #
#     {"chunk_name", "text", "entities": [...], "relationships": [...]}  #
#                                                                        #
#  One graph is maintained per dataset (knowledge_graph.json).           #
# ====================================================================== #

GRAPH_EXTRACTION_PROMPT = (
    "You are a knowledge-graph extraction engine.\n"
    "Given a text passage, extract all meaningful entities and the "
    "relationships between them.\n"
    "Rules:\n"
    "- Each entity has a \"name\" and a \"type\".\n"
    "- \"source\" and \"target\" in relationships MUST exactly match an entity \"name\".\n"
    "- \"relation\" must be a short UPPER_SNAKE_CASE verb.\n"
    "- Return valid JSON matching the provided schema."
)

GRAPH_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Canonical name of the entity",
                    },
                    "type": {
                        "type": "string",
                        "description": "Entity type, e.g. Person, Object, Location, "
                        "Organisation, Concept, Event, Equipment, Attribute",
                    },
                },
                "required": ["name", "type"],
            },
            "minItems": 2,
            "maxItems": 8,
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Must match an entity name exactly",
                    },
                    "target": {
                        "type": "string",
                        "description": "Must match an entity name exactly",
                    },
                    "relation": {
                        "type": "string",
                        "description": "UPPER_SNAKE_CASE verb, e.g. PLACES, CAUSES, LOCATED_IN",
                    },
                    "description": {
                        "type": "string",
                        "description": "Free-text description of this relationship",
                    },
                },
                "required": ["source", "target", "relation"],
            },
            "minItems": 1,
            "maxItems": 10,
        },
    },
    "required": ["entities", "relationships"],
}

GRAPH_PATH = "/graph_rag"


class ServiceRunner(dl.BaseServiceRunner):

    # ------------------------------------------------------------------ #
    #  Graph persistence — single graph per dataset                       #
    # ------------------------------------------------------------------ #
    GRAPH_FILENAME = "knowledge_graph.json"

    def _load_graph(self, dataset: dl.Dataset) -> nx.DiGraph:
        try:
            filters = dl.Filters()
            filters.add(field="name", values=self.GRAPH_FILENAME)
            filters.add(field="dir", values=GRAPH_PATH)
            pages = dataset.items.list(filters=filters)
            for graph_item in pages.all():
                buf = graph_item.download(save_locally=False)
                data = json.loads(buf.read().decode("utf-8"))
                G = nx.node_link_graph(data)
                logger.info(
                    f"Loaded graph: "
                    f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                )
                return G
        except Exception as e:
            logger.info(f"No existing graph found ({e}), creating new")
        return nx.DiGraph()

    def _save_graph(self, G: nx.DiGraph, dataset: dl.Dataset) -> dl.Item:
        data = nx.node_link_data(G)
        data["_meta"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
        }
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
        )
        try:
            json.dump(data, tmp, indent=2)
            tmp.close()
            return dataset.items.upload(
                local_path=tmp.name,
                remote_name=self.GRAPH_FILENAME,
                remote_path=GRAPH_PATH,
                overwrite=True,
                item_metadata={
                    "user": {
                        "type": "knowledge_graph",
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                    }
                },
            )
        finally:
            os.remove(tmp.name)

    # ------------------------------------------------------------------ #
    #  Merge structured data into the graph                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_into_graph(
        G: nx.DiGraph,
        chunk_name: str,
        text: str,
        item_id: str,
        entities: list[dict],
        relationships: list[dict],
    ):
        chunk_node = f"Chunk:{chunk_name}"
        G.add_node(chunk_node, type="chunk", text=text, item_id=item_id)

        entity_map: dict[str, str] = {}
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "Entity").strip()
            if not name:
                continue
            nid = f"{etype}:{name}"
            G.add_node(nid, type=etype.lower(), label=name)
            G.add_edge(chunk_node, nid, label="MENTIONS")
            entity_map[name.lower()] = nid

        for rel in relationships:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            relation = rel.get("relation", "RELATED_TO").strip()
            desc = rel.get("description", "")
            src_id = entity_map.get(src.lower())
            tgt_id = entity_map.get(tgt.lower())
            if src_id and tgt_id:
                G.add_edge(src_id, tgt_id, label=relation, description=desc)

    # ------------------------------------------------------------------ #
    #  1. Build graph — incremental, one item at a time                   #
    # ------------------------------------------------------------------ #
    def add_chunk_to_graph(self, item: dl.Item) -> dl.Item:
        """
        Pipeline node — accepts one of:

        • **Prompt item** with an LLM response (guided JSON) as the last
          assistant message containing {entities[], relationships[]}.
          The user message is used as the chunk text.

        • **JSON item** (.json) with the structured schema:
          {chunk_name, text, entities[], relationships[]}

        Raises ValueError for unsupported item formats.
        A single graph is maintained per dataset.
        """
        chunk_name, text, entities, relationships = self._parse_item(item)

        dataset = item.dataset
        G = self._load_graph(dataset)

        self._merge_into_graph(G, chunk_name, text, item.id, entities, relationships)

        self._save_graph(G, dataset)
        logger.info(
            f"Added chunk {chunk_name} to graph "
            f"- {len(entities)} entities, {len(relationships)} relations"
        )
        return item

    @staticmethod
    def _is_prompt_item(item: dl.Item) -> bool:
        return (
            item.metadata.get("system", {})
            .get("shebang", {})
            .get("dltype")
            == "prompt"
        )

    @staticmethod
    def _parse_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]:
        """
        Extract (chunk_name, text, entities, relationships) from an item.
        Supports prompt items and structured JSON items only.
        Raises ValueError for any other format.
        """
        if ServiceRunner._is_prompt_item(item):
            return ServiceRunner._parse_prompt_item(item)

        mimetype = item.metadata.get("system", {}).get("mimetype", "")
        if mimetype.startswith("application/json") or item.name.endswith(".json"):
            return ServiceRunner._parse_json_item(item)

        raise ValueError(
            f"Unsupported item format for '{item.name}' (mimetype={mimetype}). "
            f"Expected a prompt item or a .json file."
        )

    @staticmethod
    def _parse_prompt_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]:
        """Parse a prompt item — user message = text, assistant message = guided JSON."""
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()

        user_text = ""
        assistant_raw = None
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", [])
            if not content:
                continue
            value = content[0].get("text", "")
            if role == "user" and value:
                user_text = value
            elif role == "assistant" and value:
                assistant_raw = value

        if not assistant_raw:
            raise ValueError(
                f"Prompt item '{item.name}' has no assistant response to extract."
            )

        data = ServiceRunner._extract_json(assistant_raw)
        entities, relationships = ServiceRunner._split_entities_and_relationships(data)
        return (
            item.name,
            user_text,
            entities,
            relationships,
        )

    @staticmethod
    def _extract_json(text: str):
        """Extract JSON from a raw LLM response, stripping markdown fences and surrounding text."""
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        for start in range(len(text)):
            if text[start] in ("{", "["):
                bracket = "}" if text[start] == "{" else "]"
                for end in range(len(text) - 1, start - 1, -1):
                    if text[end] == bracket:
                        return json.loads(text[start:end + 1])

        raise ValueError("No valid JSON found in LLM response.")

    @staticmethod
    def _split_entities_and_relationships(data) -> tuple[list[dict], list[dict]]:
        """
        Handle both structured {entities, relationships} and flat-array
        formats where entities and relationships are mixed in one list.
        """
        if isinstance(data, dict):
            return data.get("entities", []), data.get("relationships", [])

        if isinstance(data, list):
            entities = []
            relationships = []
            for obj in data:
                if "source" in obj and "target" in obj:
                    relationships.append(obj)
                elif "name" in obj:
                    entities.append(obj)
            return entities, relationships

        raise ValueError(f"Unexpected JSON type: {type(data).__name__}")

    @staticmethod
    def _parse_json_item(item: dl.Item) -> tuple[str, str, list[dict], list[dict]]:
        """Parse a structured JSON item with entities and relationships."""
        buf = item.download(save_locally=False)
        raw = buf.read().decode("utf-8", errors="replace").strip()
        if not raw:
            raise ValueError(f"JSON item '{item.name}' is empty.")

        data = json.loads(raw)
        return (
            data.get("chunk_name", item.name),
            data.get("text", ""),
            data.get("entities", []),
            data.get("relationships", []),
        )

    # ------------------------------------------------------------------ #
    #  2. Retrieve from graph — keyword search, returns LLM-ready context #
    # ------------------------------------------------------------------ #
    def query_graph(self, item: dl.Item) -> dl.Item:
        """
        Pipeline node — receives a prompt item, extracts the last user
        message, searches the dataset graph for matching entities within
        2 hops, builds an LLM-ready context block (relationship triples +
        source chunk texts), and adds it to the prompt item as an
        assistant message so the next LLM node can consume it.

        Returns the updated prompt item.
        """
        query = self._extract_query_from_prompt(item)
        if not query:
            logger.warning(f"No user message found in prompt item {item.id}")
            return item

        dataset = item.dataset
        G = self._load_graph(dataset)
        if G.number_of_nodes() == 0:
            logger.warning("No graph data available in this dataset.")
            return item

        keywords = {w.lower() for w in query.split() if len(w) > 2}

        matched = set()
        for nid, d in G.nodes(data=True):
            if d.get("type") == "chunk":
                continue
            label = d.get("label", "").lower()
            if any(kw in label for kw in keywords):
                matched.add(nid)

        if not matched:
            logger.info(f"No entities matching query: {query}")
            return item

        relevant_edges = []
        chunk_texts = []
        seen_chunks = set()

        for node in matched:
            for u, v, d in G.edges(data=True):
                if u != node and v != node:
                    continue
                relevant_edges.append((u, v, d))
                for ep in (u, v):
                    ep_data = G.nodes.get(ep, {})
                    if ep_data.get("type") == "chunk" and ep not in seen_chunks:
                        seen_chunks.add(ep)
                        chunk_texts.append(ep_data.get("text", ""))
                    for u2, v2, _ in G.edges(data=True):
                        if u2 == ep or v2 == ep:
                            for ep2 in (u2, v2):
                                nd = G.nodes.get(ep2, {})
                                if nd.get("type") == "chunk" and ep2 not in seen_chunks:
                                    seen_chunks.add(ep2)
                                    chunk_texts.append(nd.get("text", ""))

        context = self._build_context(G, query, relevant_edges, chunk_texts)

        prompt_item = dl.PromptItem.from_item(item)
        prompt_item.add(
            message={
                "role": "assistant",
                "content": [{"mimetype": dl.PromptType.TEXT, "value": context}],
            }
        )
        prompt_item.update()

        logger.info(
            f"Query '{query[:60]}' matched {len(matched)} entities, "
            f"{len(seen_chunks)} chunks - context added to prompt"
        )
        return item

    @staticmethod
    def _build_context(
        G: nx.DiGraph,
        query: str,
        edges: list[tuple],
        chunk_texts: list[str],
    ) -> str:
        """Format graph retrieval results as an LLM-readable context block."""
        lines = [
            "=== Graph-RAG Context ===",
            f"User query: {query}",
            "",
            "Relevant relationships:",
        ]
        seen = set()
        for u, v, d in edges:
            u_lbl = G.nodes[u].get("label", u) if u in G.nodes else u
            v_lbl = G.nodes[v].get("label", v) if v in G.nodes else v
            rel = d.get("label", "RELATED")
            if rel == "MENTIONS":
                continue
            desc = d.get("description", "")
            key = (u_lbl, rel, v_lbl)
            if key in seen:
                continue
            seen.add(key)
            line = f"  {u_lbl} -[{rel}]-> {v_lbl}"
            if desc:
                line += f"  ({desc})"
            lines.append(line)

        if chunk_texts:
            lines += ["", "Source passages:"]
            for i, txt in enumerate(chunk_texts[:10], 1):
                lines.append(f"  [{i}] {txt[:500]}")

        lines.append("=== End Context ===")
        return "\n".join(lines)

    @staticmethod
    def _extract_query_from_prompt(item: dl.Item) -> str:
        """Extract the last user message text from a Dataloop PromptItem."""
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        if not messages:
            return ""
        last_message = messages[-1]
        content = last_message.get("content", [])
        if not content:
            return ""
        return content[0].get("text", "")

    # ------------------------------------------------------------------ #
    #  3. Visualize & upload                                              #
    # ------------------------------------------------------------------ #
    def export_graph(self, dataset: dl.Dataset) -> dl.Item:
        G = self._load_graph(dataset)
        if G.number_of_nodes() == 0:
            logger.warning("No graph data in dataset")
            return None
        return self._visualize_and_upload(G, dataset)

    def _visualize_and_upload(
        self, G: nx.DiGraph, dataset: dl.Dataset,
    ) -> dl.Item:
        TYPE_STYLES = {
            "chunk":     {"color": "#90CAF9", "size": 1400, "shape": "o", "edge": "#1565C0"},
            "person":    {"color": "#EF9A9A", "size": 1100, "shape": "o", "edge": "#C62828"},
            "object":    {"color": "#81C784", "size": 1000, "shape": "s", "edge": "#2E7D32"},
            "equipment": {"color": "#FFD54F", "size": 1000, "shape": "h", "edge": "#F57F17"},
            "location":  {"color": "#CE93D8", "size": 1000, "shape": "d", "edge": "#6A1B9A"},
            "event":     {"color": "#FFAB91", "size": 1000, "shape": "^", "edge": "#BF360C"},
            "attribute": {"color": "#B0BEC5", "size": 800,  "shape": "o", "edge": "#455A64"},
        }
        DEFAULT_STYLE = {"color": "#E0E0E0", "size": 900, "shape": "o", "edge": "#616161"}

        fig, ax = plt.subplots(figsize=(26, 18))

        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center",
                    fontsize=18, color="gray")
        else:
            k = 3.0 / (G.number_of_nodes() ** 0.5) if G.number_of_nodes() > 1 else 1.0
            pos = nx.spring_layout(G, seed=42, k=k, iterations=100)

            drawn_types = set()
            for nid, d in G.nodes(data=True):
                ntype = d.get("type", "").lower()
                drawn_types.add(ntype)
                style = TYPE_STYLES.get(ntype, DEFAULT_STYLE)
                nx.draw_networkx_nodes(
                    G, pos, nodelist=[nid], ax=ax,
                    node_color=style["color"], node_size=style["size"],
                    node_shape=style["shape"], alpha=0.92,
                    edgecolors=style["edge"], linewidths=1.2,
                )

            labels = {}
            for n, d in G.nodes(data=True):
                lbl = d.get("label", n.split(":")[-1] if ":" in n else n)
                labels[n] = "\n".join(textwrap.wrap(str(lbl), width=14))
            nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                                    font_size=6, font_weight="bold")

            mention_edges = [(u, v) for u, v, d in G.edges(data=True)
                             if d.get("label") == "MENTIONS"]
            relation_edges = [(u, v) for u, v, d in G.edges(data=True)
                              if d.get("label") != "MENTIONS"]

            if mention_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=mention_edges, ax=ax,
                    arrowstyle="-|>", arrowsize=8,
                    edge_color="#90CAF9", alpha=0.3, style="dashed",
                )
            if relation_edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=relation_edges, ax=ax,
                    arrowstyle="-|>", arrowsize=12,
                    edge_color="#455A64", alpha=0.7,
                    connectionstyle="arc3,rad=0.1",
                )

            edge_labels = {
                (u, v): d.get("label", "")
                for u, v, d in G.edges(data=True)
                if d.get("label") != "MENTIONS"
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                         font_color="#1565C0", font_size=5)

            legend_items = []
            for tname, style in TYPE_STYLES.items():
                if tname in drawn_types:
                    marker = {"o": "o", "s": "s", "h": "h", "d": "D", "^": "^"}.get(
                        style["shape"], "o"
                    )
                    legend_items.append(
                        plt.Line2D([], [], marker=marker, color="w",
                                   markerfacecolor=style["color"], markersize=10,
                                   label=tname.capitalize())
                    )
            if legend_items:
                ax.legend(handles=legend_items, loc="upper left",
                          fontsize=9, framealpha=0.9)

        title = f"Knowledge Graph - {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
        ax.axis("off")
        fig.tight_layout()

        img_name = "knowledge_graph.png"
        local_dir = os.path.join(os.path.dirname(__file__), "graph_outputs")
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, img_name)

        fig.savefig(local_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"Graph image saved locally: {local_path}")

        uploaded = dataset.items.upload(
            local_path=local_path,
            remote_name=img_name,
            remote_path=GRAPH_PATH,
            overwrite=True,
            item_metadata={
                    "user": {
                        "type": "knowledge_graph_visualization",
                        "num_nodes": G.number_of_nodes(),
                        "num_edges": G.number_of_edges(),
                    }
            },
        )
        return uploaded
