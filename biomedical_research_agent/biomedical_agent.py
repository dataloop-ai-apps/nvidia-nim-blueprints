"""
NVIDIA Biomedical AI-Q Research Agent - Dataloop Service Runner

Extends the Enterprise Research Agent with virtual screening capabilities
for biomedical drug discovery. Adapted from NVIDIA Biomedical AI-Q Research
Agent Blueprint:
https://github.com/NVIDIA-AI-Blueprints/biomedical-aiq-research-agent

Pipeline flow:
  Input -> [Init] -> [Biomedical Agent]
                       |-- "research"           -> [Research Node] -> [Biomedical Agent] (cycle)
                       |-- "virtual_screening"  -> [VS Node]       -> [Biomedical Agent]
                       '-- "generate_report"    -> [NIM Llama 3.3 70B Instruct] (end)

After the reflection loop completes, the agent checks if virtual screening is
appropriate for the topic. If yes, it identifies a target protein and seed
molecule from research results, then routes to the Virtual Screening node which
calls MolMIM (molecule generation) and DiffDock (molecular docking). The VS
results are integrated into the report before finalization.
"""

import dtlpy as dl
import os
import io
import json
import logging
import csv

import requests
import pubchempy as pcp
from rcsbapi.search import TextQuery, AttributeQuery

from enterprise_research_agent.aiq_agent import AIQEnterpriseAgent
from enterprise_research_agent.prompts import (
    SUMMARIZER_INSTRUCTIONS,
    REPORT_EXTENDER,
    REFLECTION_INSTRUCTIONS,
)
from biomedical_research_agent.prompts import (
    CHECK_VIRTUAL_SCREENING,
    CHECK_PROTEIN_MOLECULE_FOUND,
    COMBINE_VS_INTO_REPORT,
)

logger = logging.getLogger('[AIQ-Biomedical-Research]')

DEFAULT_MOLMIM_URL = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"
DEFAULT_DIFFDOCK_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
MAX_PROTEIN_MOLECULE_ITERATIONS = 3


class BiomedicalResearchAgent(AIQEnterpriseAgent):
    """Service runner extending the Enterprise Research Agent with virtual screening."""

    def __init__(self):
        super().__init__()
        self.nvidia_api_key = os.environ.get("NGC_API_KEY", "")
        self.molmim_url = os.environ.get("MOLMIM_ENDPOINT_URL", DEFAULT_MOLMIM_URL)
        self.diffdock_url = os.environ.get("DIFFDOCK_ENDPOINT_URL", DEFAULT_DIFFDOCK_URL)

    # ─── Virtual Screening Decision Helpers ───────────────────────────────────

    def _check_vs_intended(self, topic: str, report_organization: str) -> bool:
        """Ask the reasoning LLM whether virtual screening is appropriate."""
        prompt = CHECK_VIRTUAL_SCREENING.format(
            topic=topic,
            report_organization=report_organization,
        )
        response = self._invoke_llm(prompt)
        result = self._parse_json_response(response)
        if isinstance(result, dict):
            return result.get("intention", "no").lower() == "yes"
        return False

    def _find_protein_and_molecule(
        self, topic: str, state: dict, main_item: dl.Item
    ) -> tuple[str, str, str]:
        """Iteratively search for target protein and seed molecule.

        Uses existing research results as initial knowledge, then generates
        follow-up queries through the same RAG/web pipeline if needed.

        Returns (target_protein, small_molecule, vs_citations).
        """
        rag_pipeline_id = state.get("rag_pipeline_id", "")
        dataset = main_item.dataset

        knowledge_parts = []
        if state.get("running_summary"):
            knowledge_parts.append(state["running_summary"])

        vs_citations = ""
        target_protein = ""
        small_molecule = ""

        for iteration in range(MAX_PROTEIN_MOLECULE_ITERATIONS):
            knowledge_text = (
                "\n".join(knowledge_parts) if knowledge_parts
                else "No existing knowledge found."
            )
            prompt = CHECK_PROTEIN_MOLECULE_FOUND.format(
                topic=topic,
                knowledge_sources=knowledge_text,
            )
            response = self._invoke_llm(prompt)
            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                logger.warning(f"VS ingredient check iteration {iteration}: unparseable response")
                continue

            if "target_protein" in result and "recent_small_molecule_therapy" in result:
                target_protein = result["target_protein"]
                small_molecule = result["recent_small_molecule_therapy"]
                logger.info(
                    f"VS ingredients found: protein={target_protein}, molecule={small_molecule}"
                )
                break

            if "query" in result:
                query_text = result["query"]
                logger.info(f"VS ingredient search iteration {iteration}: {query_text[:80]}")
                query_result = self._process_single_query(
                    {"query": query_text}, rag_pipeline_id, dataset, main_item.id
                )
                answer = query_result.get("rag_answer") or query_result.get("web_answer", "")
                knowledge_parts.append(f"Q: {query_text}\nA: {answer}")

                citation = query_result.get("rag_citation") or query_result.get("web_citation", "")
                if citation:
                    vs_citations = (vs_citations + "\n" + citation).strip()

        return target_protein, small_molecule, vs_citations

    # ─── Molecule / Protein Lookup Helpers ────────────────────────────────────

    @staticmethod
    def _get_smiles_from_name(compound_name: str) -> str | None:
        """Look up a compound's SMILES string via PubChem."""
        try:
            compounds = pcp.get_compounds(compound_name, "name")
            if not compounds:
                logger.warning(f"PubChem: no compound found for '{compound_name}'")
                return None
            for prop in compounds[0].to_dict().get("record", {}).get("props", []):
                urn = prop.get("urn", {})
                if urn.get("label") == "SMILES" and urn.get("name") == "Absolute":
                    return prop["value"]["sval"]
            return compounds[0].isomeric_smiles
        except Exception as e:
            logger.error(f"PubChem lookup failed for '{compound_name}': {e}")
            return None

    @staticmethod
    def _get_protein_id(protein_name: str) -> str | None:
        """Look up a PDB ID for a human protein via RCSB PDB."""
        try:
            q_text = TextQuery(protein_name)
            q_human = AttributeQuery(
                attribute="rcsb_entity_source_organism.scientific_name",
                operator="exact_match",
                value="Homo sapiens",
            )
            q_em = AttributeQuery(
                attribute="exptl.method",
                operator="exact_match",
                value="electron microscopy",
            )
            query = q_text & (q_human & q_em)
            for rid in query():
                logger.info(f"RCSB PDB: found protein ID {rid} for '{protein_name}'")
                return str(rid)
            logger.warning(f"RCSB PDB: no results for '{protein_name}'")
            return None
        except Exception as e:
            logger.error(f"RCSB PDB lookup failed for '{protein_name}': {e}")
            return None

    @staticmethod
    def _fetch_pdb_string(protein_id: str) -> str | None:
        """Download a PDB structure from RCSB and return it as a string.

        No local file is created — the content stays in memory.
        """
        url = f"https://files.rcsb.org/download/{protein_id}.pdb"
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            logger.info(f"Fetched PDB structure for {protein_id} ({len(resp.text)} chars)")
            return resp.text
        except Exception as e:
            logger.error(f"PDB fetch failed for {protein_id}: {e}")
            return None

    # ─── NIM Calls ────────────────────────────────────────────────────────────

    def _call_molmim(self, smiles: str) -> str | None:
        """Call MolMIM NIM to generate novel molecules from a seed SMILES."""
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Accept": "application/json",
        }
        payload = {
            "smi": smiles,
            "num_molecules": 3,
            "algorithm": "CMA-ES",
            "property_name": "QED",
            "min_similarity": 0.7,
            "iterations": 10,
        }
        try:
            if self.molmim_url == DEFAULT_MOLMIM_URL:
                resp = requests.post(self.molmim_url, headers=headers, json=payload, timeout=120)
            else:
                resp = requests.post(self.molmim_url, json=payload, timeout=120)
            resp.raise_for_status()
            body = resp.json()
            if "molecules" in body:
                molecules = json.loads(body["molecules"])
                return "\n".join(v["sample"] for v in molecules)
            return "\n".join(v["smiles"] for v in body.get("generated", []))
        except Exception as e:
            logger.error(f"MolMIM call failed: {e}")
            return None

    def _call_diffdock(self, protein_pdb: str, ligands: str) -> dict | None:
        """Call DiffDock NIM and return the parsed response (no local file I/O)."""
        headers = {
            "Authorization": f"Bearer {self.nvidia_api_key}",
            "Accept": "application/json",
        }
        payload = {
            "protein": protein_pdb,
            "ligand": ligands,
            "ligand_file_type": "txt",
            "num_poses": 10,
            "time_divisions": 20,
            "num_steps": 18,
            "save_trajectory": "true",
        }
        try:
            if self.diffdock_url == DEFAULT_DIFFDOCK_URL:
                resp = requests.post(self.diffdock_url, headers=headers, json=payload, timeout=300)
            else:
                resp = requests.post(
                    self.diffdock_url, headers={"Accept": "application/json"},
                    json=payload, timeout=300,
                )
            resp.raise_for_status()
            body = resp.json()
            return {
                "status": body.get("status", []),
                "position_confidence": body.get("position_confidence", []),
                "ligand_positions": body.get("ligand_positions", []),
            }
        except Exception as e:
            logger.error(f"DiffDock call failed: {e}")
            return None

    def _upload_vs_artifacts(
        self, docking_result: dict, dataset, remote_path: str
    ) -> str:
        """Upload DiffDock output artifacts (CSV + .mol files) to Dataloop.

        Returns a summary string for the report.
        """
        status_list = docking_result["status"]
        confidence = docking_result["position_confidence"]
        ligand_positions = docking_result["ligand_positions"]

        uploaded_names = []

        # Upload confidence scores as CSV
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        if confidence:
            num_poses = len(confidence[0]) if confidence else 0
            for i in range(num_poses):
                row = [confidence[j][i] for j in range(len(confidence))]
                writer.writerow(row)

        csv_content = csv_buffer.getvalue()
        if csv_content:
            self._upload_data_file(dataset, csv_content, remote_path, "confidence_scores.csv")
            uploaded_names.append("confidence_scores.csv")

        # Upload .mol ligand position files
        for i, positions in enumerate(ligand_positions):
            if isinstance(positions, list):
                for j, pos in enumerate(positions):
                    name = f"ligand_{i}_{j}.mol"
                    self._upload_data_file(dataset, pos, remote_path, name)
                    uploaded_names.append(name)
            else:
                name = f"ligand_{i}.mol"
                self._upload_data_file(dataset, positions, remote_path, name)
                uploaded_names.append(name)

        info = (
            f"\nDiffDock docking completed. Status: [{', '.join(status_list)}].\n"
            f"Position confidence scores: {confidence}\n"
            f"Uploaded {len(uploaded_names)} artifact files to dataset: "
            f"{', '.join(uploaded_names)}\n"
        )
        return info

    # ─── Overridden Agent Node ────────────────────────────────────────────────

    def run_agent(self, item: dl.Item, context: dl.Context, progress: dl.Progress):
        """Pipeline node: Biomedical AIQ Agent (orchestrator)

        Extends the enterprise agent with a virtual_screening action that fires
        after the reflection loop completes but before generating the final report.

        Actions:
          - "research": send queries to Research node (inherited)
          - "virtual_screening": send to VS node for MolMIM + DiffDock
          - "generate_report": send PromptItem to NIM Llama (inherited)
        """
        logger.info("=== Biomedical AIQ Agent node ===")

        if self._is_temp_item(item):
            main_item = self._get_main_item(item)
            source = item.metadata.get("user", {}).get("source", "unknown")
            logger.info(f"Received item from: {source}")
        else:
            main_item = item
            source = "init"
            logger.info("First call - initializing research")

        state = self._get_state(main_item)

        # ── RETURNING FROM VIRTUAL SCREENING ──
        if source == "virtual_screening":
            vs_results_id = item.metadata.get("user", {}).get("vs_results_file_id", "")
            if vs_results_id:
                try:
                    vs_data = json.loads(self._download_data_file(vs_results_id))
                    vs_info = vs_data.get("vs_steps_info", "")
                    vs_queries = vs_data.get("vs_queries", "")
                    vs_queries_results = vs_data.get("vs_queries_results", "")
                    vs_citations = vs_data.get("vs_citations", "")

                    if vs_info:
                        combine_prompt = COMBINE_VS_INTO_REPORT.format(
                            report_organization=state["report_organization"],
                            report=state["running_summary"],
                            vs_queries=vs_queries,
                            vs_queries_results=vs_queries_results,
                            vs_info=vs_info,
                        )
                        state["running_summary"] = self._invoke_llm(combine_prompt)
                        logger.info("VS results integrated into report")

                    if vs_citations:
                        state["citations"] = (
                            state.get("citations", "") + "\n" + vs_citations
                        ).strip()
                except Exception as e:
                    logger.error(f"Could not read VS results: {e}")

            return self._prepare_for_report(main_item, state, progress)

        # ── RETURNING FROM RESEARCH: check if VS should happen after reflections ──
        if source == "research":
            result = self._handle_research_return(item, main_item, state, progress)
            if result is not None:
                return result

            # Reflections complete — check virtual screening
            if not state.get("vs_checked"):
                state["vs_checked"] = True
                vs_intended = self._check_vs_intended(
                    state["topic"], state["report_organization"]
                )
                state["do_virtual_screening"] = vs_intended
                logger.info(f"Virtual screening intended: {vs_intended}")

                if vs_intended:
                    target_protein, small_molecule, vs_citations = (
                        self._find_protein_and_molecule(state["topic"], state, main_item)
                    )
                    state["target_protein"] = target_protein
                    state["small_molecule"] = small_molecule
                    state["vs_citations"] = vs_citations

                    if target_protein and small_molecule:
                        main_item = self._set_state(main_item, state)
                        temp_item = self._create_temp_item(
                            main_item,
                            content=json.dumps({
                                "target_protein": target_protein,
                                "small_molecule": small_molecule,
                            }),
                            name="virtual_screening_request",
                        )
                        temp_item.metadata.setdefault("user", {})
                        temp_item.metadata["user"]["source"] = "agent_to_vs"
                        temp_item.update(system_metadata=True)

                        progress.update(action="virtual_screening")
                        return temp_item
                    else:
                        logger.warning(
                            "VS intended but could not find protein/molecule — skipping VS"
                        )

            return self._prepare_for_report(main_item, state, progress)

        # ── FIRST CALL or other sources: delegate to parent ──
        return super().run_agent(item, context, progress)

    def _handle_research_return(
        self, item: dl.Item, main_item: dl.Item, state: dict, progress: dl.Progress
    ) -> dl.Item | None:
        """Process research results and run reflection loop.

        Returns a temp_item routed to research if more reflections are needed,
        or None if reflections are complete.
        """
        results_file_id = item.metadata.get("user", {}).get("results_file_id", "")
        sources_xml = ""
        citations = ""
        if results_file_id:
            try:
                results_json = json.loads(self._download_data_file(results_file_id))
                sources_xml = results_json.get("sources_xml", "")
                citations = results_json.get("citations", "")
            except Exception as e:
                logger.error(f"Could not read research results file: {e}")

        state["citations"] = (state.get("citations", "") + "\n" + citations).strip()

        if not state.get("running_summary"):
            summary_prompt = SUMMARIZER_INSTRUCTIONS.format(
                report_organization=state["report_organization"],
                source=sources_xml,
            )
            state["running_summary"] = self._invoke_llm(summary_prompt)
            logger.info("Initial summary generated")
        else:
            extend_prompt = REPORT_EXTENDER.format(
                report_organization=state["report_organization"],
                report=state["running_summary"],
                source=sources_xml,
            )
            state["running_summary"] = self._invoke_llm(extend_prompt)
            logger.info("Report extended with new sources")

        iteration = state.get("iteration", 0)
        max_reflections = state.get("num_reflections", 2)

        reflection_prompt = REFLECTION_INSTRUCTIONS.format(
            topic=state["topic"],
            report_organization=state["report_organization"],
            report=state["running_summary"],
        )
        reflection_response = self._invoke_llm(reflection_prompt)
        reflection = self._parse_json_response(reflection_response)

        state["iteration"] = iteration + 1

        if state["iteration"] < max_reflections and reflection:
            follow_up_query = (
                reflection.get("query", state["topic"])
                if isinstance(reflection, dict)
                else state["topic"]
            )
            logger.info(
                f"Reflection {state['iteration']}/{max_reflections}: {follow_up_query[:80]}"
            )
            state["pending_queries"] = [{"query": follow_up_query}]
            self._set_state(main_item, state)

            temp_item = self._create_temp_item(
                main_item, content=follow_up_query, name="followup_research"
            )
            temp_item.metadata.setdefault("user", {})
            temp_item.metadata["user"]["source"] = "agent_to_research"
            temp_item.update(system_metadata=True)

            progress.update(action="research")
            return temp_item

        self._set_state(main_item, state)
        return None

    # ─── Virtual Screening Pipeline Node ──────────────────────────────────────

    def virtual_screening(self, item: dl.Item):
        """Pipeline node: Virtual Screening (MolMIM + DiffDock)

        Receives a temp item routed from the agent node.
        Looks up the protein structure (RCSB PDB) and molecule SMILES (PubChem),
        then calls MolMIM for molecule generation and DiffDock for docking.
        All output artifacts are uploaded to Dataloop — no local files persist.
        """
        logger.info("=== Biomedical VS node ===")

        main_item_id = item.metadata.get("user", {}).get("main_item")
        main_item = dl.items.get(item_id=main_item_id) if main_item_id else item
        state = self._get_state(main_item)

        target_protein = state.get("target_protein", "")
        small_molecule = state.get("small_molecule", "")
        vs_steps_info = ""

        if not target_protein or not small_molecule:
            vs_steps_info = "Virtual screening skipped: missing protein or molecule."
            logger.warning(vs_steps_info)
            return self._finish_vs(item, main_item, vs_steps_info, state)

        vs_steps_info += (
            f"Target protein: {target_protein}\nSeed molecule: {small_molecule}\n"
        )

        # Step 1: Resolve protein PDB ID and fetch structure into memory
        protein_id = self._get_protein_id(target_protein)
        if not protein_id:
            vs_steps_info += (
                f"\nCould not find PDB ID for protein '{target_protein}'. "
                "Skipping virtual screening.\n"
            )
            return self._finish_vs(item, main_item, vs_steps_info, state)
        vs_steps_info += f"Found PDB ID: {protein_id}\n"

        protein_structure = self._fetch_pdb_string(protein_id)
        if not protein_structure:
            vs_steps_info += (
                f"\nFailed to fetch PDB structure for {protein_id}. "
                "Skipping virtual screening.\n"
            )
            return self._finish_vs(item, main_item, vs_steps_info, state)
        vs_steps_info += f"Fetched PDB structure ({len(protein_structure)} chars)\n"

        # Step 2: Resolve molecule SMILES
        smiles = self._get_smiles_from_name(small_molecule)
        if not smiles:
            vs_steps_info += (
                f"\nCould not find SMILES for molecule '{small_molecule}'. "
                "Skipping virtual screening.\n"
            )
            return self._finish_vs(item, main_item, vs_steps_info, state)
        vs_steps_info += f"Resolved SMILES: {smiles}\n"

        # Step 3: MolMIM — generate novel molecules
        generated_ligands = self._call_molmim(smiles)
        if not generated_ligands:
            vs_steps_info += "\nMolMIM molecule generation failed. Skipping docking.\n"
            return self._finish_vs(item, main_item, vs_steps_info, state)
        vs_steps_info += f"Generated ligands from MolMIM:\n{generated_ligands}\n"

        # Step 4: DiffDock — dock generated molecules against protein
        docking_result = self._call_diffdock(protein_structure, generated_ligands)
        if not docking_result:
            vs_steps_info += "\nDiffDock docking call failed.\n"
            return self._finish_vs(item, main_item, vs_steps_info, state)

        # Upload docking artifacts (.mol files, confidence CSV) to Dataloop
        artifact_path = self._item_folder(main_item.id) + "vs_artifacts/"
        artifact_info = self._upload_vs_artifacts(
            docking_result, main_item.dataset, artifact_path
        )
        vs_steps_info += artifact_info

        return self._finish_vs(item, main_item, vs_steps_info, state)

    def _finish_vs(
        self, item: dl.Item, main_item: dl.Item, vs_steps_info: str, state: dict
    ) -> dl.Item:
        """Store VS results as a JSON file in Dataloop and route back to the agent."""
        vs_data = json.dumps({
            "vs_steps_info": vs_steps_info,
            "vs_queries": state.get("target_protein", ""),
            "vs_queries_results": state.get("small_molecule", ""),
            "vs_citations": state.get("vs_citations", ""),
        }, ensure_ascii=False)

        results_file = self._upload_data_file(
            dataset=main_item.dataset,
            data=vs_data,
            remote_path=self._item_folder(main_item.id),
            filename=f"vs_results_{item.id[:8]}.json",
        )

        item.metadata.setdefault("user", {})
        item.metadata["user"]["source"] = "virtual_screening"
        item.metadata["user"]["vs_results_file_id"] = results_file.id
        item.update(system_metadata=True)

        return item
