# Virtual Screening utilities adapted from NVIDIA Biomedical AI-Q Research Agent Blueprint
# Handles PubChem molecule lookup, RCSB PDB protein lookup, MolMIM generation, and DiffDock docking
# SPDX-License-Identifier: Apache-2.0

import os
import json
import logging
import csv
import requests
import datetime
import pubchempy as pcp
from rcsbapi.search import TextQuery, AttributeQuery

logger = logging.getLogger('[VirtualScreening]')


def get_isomeric_smiles(pcp_compound):
    """Extract the absolute SMILES string from a PubChem compound."""
    absolute_smiles = [
        i['value']["sval"]
        for i in pcp_compound.to_dict().get("record", {}).get("props", [])
        if i["urn"]["label"] == "SMILES" and i["urn"]["name"] == "Absolute"
    ]
    if len(absolute_smiles) == 0:
        return None
    return absolute_smiles[0]


def get_smiles_from_molecule_name(compound_name: str):
    """Look up a molecule by name in PubChem and return its SMILES string."""
    info = ""
    compounds = pcp.get_compounds(compound_name, 'name')

    if len(compounds) == 0:
        info += f"Could not find a molecule from name: {compound_name}.\n"
        return None, info

    smiles = get_isomeric_smiles(compounds[0])
    info += f"Found SMILES string: {smiles} for molecule {compounds[0].cid} with name {compound_name} in PubChem.\n"

    if len(compounds) > 1:
        info += "Multiple molecules found in PubChem, using the first molecule's SMILES string.\n"

    return smiles, info


def get_protein_id_from_name(protein_name: str):
    """Search RCSB PDB for a protein ID from a protein name (Homo sapiens, electron microscopy)."""
    info = ""
    q1 = TextQuery(protein_name)
    q2 = AttributeQuery(
        attribute="rcsb_entity_source_organism.scientific_name",
        operator="exact_match",
        value="Homo sapiens"
    )
    q3 = AttributeQuery(
        attribute="exptl.method",
        operator="exact_match",
        value="electron microscopy"
    )
    query = q1 & (q2 & q3)
    info += f"Looking for protein ID from protein name '{protein_name}' (Homo sapiens, electron microscopy).\n"

    results = query()
    first_id = None
    other_ids = []

    for rid in results:
        rid = str(rid)
        if first_id is None:
            first_id = rid
            info += f"First protein ID found: {first_id}\n"
        else:
            other_ids.append(rid)

    if first_id is None:
        info += f"Could not find protein ID from protein name: {protein_name}\n"

    if len(other_ids) > 0:
        info += f"Other protein IDs found: {' '.join(other_ids)}\n"

    return first_id, info


def download_pdb_from_protein_id(protein_id: str, output_dir: str):
    """Download PDB file from RCSB for a given protein ID."""
    url = f"https://files.rcsb.org/download/{protein_id}.pdb"
    response = requests.get(url)
    info = ""

    if response.status_code == 200:
        filename = os.path.join(output_dir, f"{protein_id}.pdb")
        with open(filename, "wb") as f:
            f.write(response.content)
        info += f"Downloaded PDB file from {url} to {filename}\n"
        return filename, info
    else:
        info += f"Failed to download PDB file from {url} (status {response.status_code})\n"
        return None, info


def pdb_to_string(pdb_filepath: str) -> str:
    """Read a PDB file and return its contents as a string."""
    with open(pdb_filepath, 'r') as f:
        return f.read()


def generate_molecule(molecule: str, molmim_invoke_url: str) -> str:
    """Call MolMIM NIM to generate molecules similar to a target molecule.
    Returns generated ligands in SMILES format.
    """
    logger.info("Calling MolMIM NIM")
    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {nvidia_api_key}",
        "Accept": "application/json",
    }
    payload = {
        'smi': molecule,
        'num_molecules': 3,
        'algorithm': 'CMA-ES',
        'property_name': 'QED',
        'min_similarity': 0.7,
        'iterations': 10,
    }

    session = requests.Session()

    if molmim_invoke_url == "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate":
        response = session.post(molmim_invoke_url, headers=headers, json=payload)
        response.raise_for_status()
        response_body = response.json()
        molecules = json.loads(response_body['molecules'])
        generated_ligands = '\n'.join([v['sample'] for v in molecules])
    else:
        response = session.post(molmim_invoke_url, json=payload)
        response.raise_for_status()
        response_body = response.json()
        generated_ligands = '\n'.join(v["smiles"] for v in response_body['generated'])

    return generated_ligands


def dock_molecule(curr_out_dir: str, folded_protein: str, generated_ligands: str, diffdock_invoke_url: str) -> str:
    """Call DiffDock NIM to dock generated ligands against a protein structure.
    Returns docking status information as a string.
    """
    logger.info("Calling DiffDock NIM")
    nvidia_api_key = os.getenv("NVIDIA_API_KEY", "")

    headers = {
        "Authorization": f"Bearer {nvidia_api_key}",
        "Accept": "application/json",
    }
    payload = {
        'protein': folded_protein,
        'ligand': generated_ligands,
        'ligand_file_type': 'txt',
        'num_poses': 10,
        'time_divisions': 20,
        'num_steps': 18,
        'save_trajectory': 'true',
    }
    docking_status = ""

    try:
        if diffdock_invoke_url == "https://health.api.nvidia.com/v1/biology/mit/diffdock":
            response = requests.post(diffdock_invoke_url, headers=headers, json=payload)
        else:
            response = requests.post(diffdock_invoke_url, headers={"Accept": "application/json"}, json=payload)

        response.raise_for_status()
        response_body = response.json()

        diffdock_position_confidence = response_body["position_confidence"]
        ret_conf_scores = []
        for i in range(10):
            current_pos = [diffdock_position_confidence[j][i] for j in range(3)]
            ret_conf_scores.append(current_pos)

        logger.info("Confidence scores from DiffDock:\n" + "\n".join(str(sc) for sc in ret_conf_scores))

        with open(os.path.join(curr_out_dir, 'confidence_scores.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(ret_conf_scores)

        for i in range(len(response_body['ligand_positions'])):
            if isinstance(response_body['ligand_positions'][i], list):
                for j in range(len(response_body['ligand_positions'][i])):
                    with open(os.path.join(curr_out_dir, f'ligand_{i}_{j}.mol'), "w") as f:
                        f.write(response_body['ligand_positions'][i][j])
            else:
                with open(os.path.join(curr_out_dir, f'ligand_{i}.mol'), "w") as f:
                    f.write(response_body['ligand_positions'][i])

        docking_status += f"\nDocking completed. Status: [{', '.join(response_body['status'])}].\n"
        docking_status += f"Position confidence scores:\n{chr(10).join(str(sc) for sc in diffdock_position_confidence)}\n"
        docking_status += f"Ligand positions saved to {curr_out_dir}.\n"
        return docking_status

    except Exception as e:
        logger.error(f"DiffDock error: {e}")
        docking_status += f"\nDocking in DiffDock failed. Error: {e}.\n"
        return docking_status


def run_virtual_screening(target_protein: str, recent_molecule: str) -> str:
    """Run the full virtual screening pipeline:
    1. Look up protein in RCSB PDB
    2. Look up molecule in PubChem
    3. Generate similar molecules with MolMIM
    4. Dock generated molecules with DiffDock

    Returns a comprehensive info string with all results.
    """
    all_info = ""

    if not target_protein or not recent_molecule:
        if not target_protein:
            all_info += "The target protein was not found from the search.\n"
        if not recent_molecule:
            all_info += "The recent small molecule therapy was not found from the search.\n"
        all_info += "Not proceeding with Virtual Screening.\n"
        return all_info

    all_info += f"Using target protein: {target_protein}, recent molecule: {recent_molecule}\n"

    # 1. Get protein ID and PDB structure
    protein_id, info = get_protein_id_from_name(target_protein)
    all_info += info

    if protein_id is None:
        all_info += f"Abandoning virtual screening: no protein found for '{target_protein}'\n"
        return all_info

    # 2. Get molecule SMILES
    molecule_smiles, info = get_smiles_from_molecule_name(recent_molecule)
    all_info += info

    if molecule_smiles is None:
        all_info += f"Abandoning virtual screening: no molecule found for '{recent_molecule}'\n"
        return all_info

    # 3. Download PDB file
    try:
        curr_out_dir = os.path.join("/tmp", "virtual_screening_output",
                                    datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(curr_out_dir, exist_ok=True)
    except Exception as e:
        all_info += f"Error creating output directory: {e}\n"
        return all_info

    pdb_filepath, info = download_pdb_from_protein_id(protein_id, curr_out_dir)
    all_info += info

    if pdb_filepath is None:
        all_info += "Abandoning virtual screening: could not download PDB file.\n"
        return all_info

    # 4. Read protein structure
    try:
        protein_structure = pdb_to_string(pdb_filepath)
    except Exception as e:
        all_info += f"Error reading PDB file: {e}\n"
        return all_info

    # 5. Generate molecules with MolMIM
    try:
        molmim_url = os.getenv("MOLMIM_ENDPOINT_URL",
                               "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate")
        generated_ligands = generate_molecule(molecule=molecule_smiles, molmim_invoke_url=molmim_url)
        all_info += f"Generated ligands from MolMIM:\n{generated_ligands}\n"
    except Exception as e:
        all_info += f"Error in MolMIM molecule generation: {e}\n"
        return all_info

    # 6. Dock molecules with DiffDock
    try:
        diffdock_url = os.getenv("DIFFDOCK_ENDPOINT_URL",
                                 "https://health.api.nvidia.com/v1/biology/mit/diffdock")
        dock_info = dock_molecule(curr_out_dir, protein_structure, generated_ligands, diffdock_url)
        all_info += dock_info
    except Exception as e:
        all_info += f"Error in DiffDock docking: {e}\n"

    return all_info
