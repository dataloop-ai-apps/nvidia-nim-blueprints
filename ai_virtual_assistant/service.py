import logging
import dtlpy as dl
import json
import os
from collections import defaultdict
from typing import List
# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class ServiceRunner(dl.BaseServiceRunner):
    
    @staticmethod
    def create_temp_item(item: dl.Item) -> dl.Item:
        """
        Creates a temporary item from the item.
        """
        temp_item = item.clone(
            dst_dataset_id=item.dataset_id,  # Clone to the same dataset
            remote_filepath=f"/.dataloop/temp_{os.path.basename(item.name)}",
            with_annotations=True,          # Keep annotations (default)
            with_metadata=True,             # Keep original metadata (default)
            wait=True                       # Wait for the clone operation to complete
        )

        temp_item.metadata['original_item_id'] = item.id
        temp_item.hidden = True
        temp_item.update(system_metadata=True)
        return temp_item
    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _handle_tool_call(item: dl.Item, progress: dl.Progress = dl.Progress()) -> dl.Item:
        """
        Process a single item:

        1. Convert it to a PromptItem.
        2. Locate the last assistant message and extract its JSON payload.
        3. Validate and interpret the schema (`toolCall`, `response`).
           • If toolCall == "none"   → replace text with the *response* field.
           • Otherwise               → replace text with
                                        "\nUsing tools: <tool-name>...\n\n"
        4. Delete the previous assistant annotation (coordinates are read-only)
           and create a brand-new annotation with the updated text.
        5. If toolCall != "none", add `toolCall` to the item 's metadata.
        6. Return the modified item.
        """
        # ----------------------------------------------------------------- #
        # 1. Ensure we are working with a PromptItem                        #
        # ----------------------------------------------------------------- #
        try:
            prompt_item: dl.PromptItem = dl.PromptItem.from_item(item=item)
        except Exception as e:
            logger.error(f"{item.id}: cannot cast to PromptItem - {e}")
            return item

        # ----------------------------------------------------------------- #
        # 2. Fetch the last assistant message                               #
        # ----------------------------------------------------------------- #
        last_assistant_msg = ServiceRunner._get_last_assistant_message(prompt_item, item)
        if last_assistant_msg is None:
            return item

        # ----------------------------------------------------------------- #
        # 3. Parse the JSON payload                                         #
        # ----------------------------------------------------------------- #
        parsed = ServiceRunner._parse_payload(last_assistant_msg, item)
        if parsed is None:
            return item
        tool_call: str = parsed["toolCall"]
        response_text: str = parsed["response"]

        # Compose the replacement text according to the rules
        new_text = (
            response_text
            if tool_call == "none"
            else f"\n`Using tools: {tool_call}...`\n\n"
        )
        # ----------------------------------------------------------------- #
        # 4. Delete the old annotation & upload a new one                   #
        # ----------------------------------------------------------------- #
        if not ServiceRunner._replace_annotation(
            prompt_item=prompt_item, item=item, new_text=new_text, tool_call=tool_call
        ):
            return item  # replacement failed - keep original item

        # ----------------------------------------------------------------- #
        # 5. Add toolCall to metadata (if needed)                           #
        # ----------------------------------------------------------------- #
        if tool_call != "none":
            item.metadata["toolCall"] = tool_call
            try:
                item.update(system_metadata=False)
            except Exception as e:
                logger.warning(f"{item.id}: failed to update metadata - {e}")
                
        progress.update(action=tool_call)

        # Done
        return item
    
    @staticmethod
    def handle_tool_call_primary(item: dl.Item, progress: dl.Progress = dl.Progress()) -> dl.Item:
        return ServiceRunner._handle_tool_call(item, progress)
    
    @staticmethod
    def handle_tool_call_order(item: dl.Item, progress: dl.Progress = dl.Progress()) -> dl.Item:
        return ServiceRunner._handle_tool_call(item, progress)
    
    @staticmethod
    def handle_tool_call_product(item: dl.Item, progress: dl.Progress = dl.Progress()) -> dl.Item:
        return ServiceRunner._handle_tool_call(item, progress)
    
    @staticmethod
    def query_order_db(item: dl.Item, progress: dl.Progress = dl.Progress()) -> dl.Item:
        """
        Query the order database for the order id provided by the user.
        """
        print(f"Querying order database for item: {item.id}")
        prompt_item = dl.PromptItem.from_item(item=item)
        tool_result = """
        {
            "customer_id": "NV-12345",
            "customer_name": "NVIDIA Corporation",
            "order_number": "ORD-987654",
            "order_date": "2025-04-28",
            "status": "Shipped",
            "estimated_delivery": "2025-05-08",
            "items": [
                {
                    "product_id": "GPU-RTX4090",
                    "description": "NVIDIA RTX 4090 Graphics Cards",
                    "quantity": 500,
                    "unit_price": 1599.99,
                    "subtotal": 799995.00
                },
                {
                    "product_id": "GPU-RTX4080",
                    "description": "NVIDIA RTX 4080 Graphics Cards",
                    "quantity": 750,
                    "unit_price": 1199.99,
                    "subtotal": 899992.50
                },
                {
                    "product_id": "DGX-A100",
                    "description": "NVIDIA DGX A100 Systems",
                    "quantity": 10,
                    "unit_price": 199999.00,
                    "subtotal": 1999990.00
                }
            ],
            "shipping_address": {
                "street": "2788 San Tomas Expressway",
                "city": "Santa Clara",
                "state": "CA",
                "zip": "95051",
                "country": "USA"
            },
            "total_amount": 3699977.50,
            "payment_method": "Corporate Account",
            "notes": "Priority enterprise shipment with insurance"
        }
        """
        for prompt_turn in reversed(prompt_item.prompts):
            for element in prompt_turn.elements:
                element["value"] = "<user_prompt>" + element["value"] + "</user_prompt>\n<tool_output>" + tool_result + "</tool_output>"
        prompt_item.update()
        return item
    
    
    @staticmethod
    def query_product_db(item: dl.Item, progress: dl.Progress = dl.Progress()) -> dl.Item:
        """
        Query the product database for the product id provided by the user.
        """
        print(f"Querying order database for item: {item.id}")
        prompt_item = dl.PromptItem.from_item(item=item)
        tool_result = """
        {
            "product_id": "GPU-RTX4090",
            "name": "NVIDIA GeForce RTX 4090",
            "category": "Graphics Cards",
            "architecture": "Ada Lovelace",
            "cuda_cores": 16384,
            "tensor_cores": 512,
            "rt_cores": 128,
            "memory": {
                "size": "24GB",
                "type": "GDDR6X",
                "bus_width": "384-bit",
                "bandwidth": "1008 GB/s"
            },
            "performance": {
                "boost_clock": "2.52 GHz",
                "tflops_fp32": 82.6,
                "ray_tracing": "3rd Generation",
                "dlss": "DLSS 3.5"
            },
            "power": {
                "tdp": "450W",
                "recommended_psu": "850W"
            },
            "connectivity": {
                "display_ports": 3,
                "hdmi": 1,
                "pcie_interface": "PCIe 4.0 x16"
            },
            "dimensions": {
                "length": "304mm",
                "height": "137mm",
                "width": "61mm"
            },
            "msrp": 1599.99,
            "release_date": "2022-10-12",
            "related_products": [
                "GPU-RTX4080",
                "GPU-RTX4070Ti",
                "GPU-RTX4070"
            ],
            "features": [
                "NVIDIA DLSS 3.5",
                "NVIDIA Reflex",
                "NVIDIA Broadcast",
                "NVIDIA Studio",
                "NVIDIA Omniverse",
                "AV1 Encode/Decode"
            ]
        }
        """
        for prompt_turn in reversed(prompt_item.prompts):
            for element in prompt_turn.elements:
                element["value"] = "<user_prompt>" + element["value"] + "</user_prompt>\n<tool_output>" + tool_result + "</tool_output>"
        prompt_item.update()
        return item
    
    @staticmethod
    def extract_user_prompt(item: dl.Item) -> str:
        """
        Extracts only the user prompt from the item.
        """
        prompt_item = dl.PromptItem.from_item(item=item)
        for prompt_turn in reversed(prompt_item.prompts):
            for element in prompt_turn.elements:
                text = element["value"]
                if "<user_prompt>" in text and "</user_prompt>" in text:
                    start_idx = text.find("<user_prompt>") + len("<user_prompt>")
                    end_idx = text.find("</user_prompt>")
                    if start_idx < end_idx:
                        user_prompt = text[start_idx:end_idx]
                    element["value"] = user_prompt
        prompt_item.update()
        return item
    
    @staticmethod
    def consolidate_responses(item: dl.Item):
        """
        Finds and combines text annotations within an item that share the same
        promptId in their system metadata. Concatenates text and keeps metadata
        from the latest annotation in each group.
        """
        original_item_id = item.metadata.get('original_item_id', item.id)
        if original_item_id == item.id:
            logger.warning(f"{item.id}: original item id is the same as the item id - {item.metadata}")
        original_item = item.dataset.items.get(item_id=original_item_id)
        
        print(f"Starting annotation combination for item: {item.id}")
        annotations = item.annotations.list()
        print(f"Found {len(annotations)} total annotations.")

        grouped_annotations = defaultdict(list)
        for ann in annotations:
            # Only consider text annotations with a promptId
            if ann.type != "text":
                continue
            prompt_id = ann.metadata.get('system', {}).get('promptId')
            if prompt_id is not None:
                grouped_annotations[prompt_id].append(ann)

        print(f"Found annotations for {len(grouped_annotations)} promptIds.")
        original_prompt_item = dl.PromptItem.from_item(item=original_item)

        if not grouped_annotations:
            print("No promptIds found to process.")
            item.delete() # deletes the temporary item
            return original_item

        # Determine the latest promptId (assuming numeric and sortable)
        latest_prompt_id = ""
        if grouped_annotations:
            # Sort keys numerically if they are digit strings, otherwise lexicographically
            try:
                latest_prompt_id = max(grouped_annotations.keys(), key=lambda pid: int(pid) if pid.isdigit() else pid)
            except ValueError: # Handle cases where conversion to int might fail for some keys
                latest_prompt_id = max(grouped_annotations.keys())

        print(f"Latest promptId to process: '{latest_prompt_id}'")

        if latest_prompt_id in grouped_annotations:
            group = grouped_annotations[latest_prompt_id]
            prompt_id = latest_prompt_id # Use the determined latest_prompt_id

            # for prompt_id, group in grouped_annotations.items(): # OLD LOOP
            if len(group) >= 1: # This check might be redundant if group is guaranteed by latest_prompt_id in grouped_annotations
                print(f"Found {len(group)} annotations for promptId '{prompt_id}'. Combining...")

                # Sort by creation time to ensure order
                group.sort(key=lambda a: a.created_at)

                # Combine text coordinates, separated by newlines
                if len(group) > 1:
                    combined_text = "\n\n".join([ann.coordinates for ann in group])
                else:
                    combined_text = group[0].coordinates
                print(f"Combined text length: {len(combined_text)}")

                # Use the last annotation for metadata and label
                last_ann = group[-1]
                label = last_ann.label
                metadata = last_ann.metadata
                original_ids = [ann.id for ann in group]

                print(f"Deleting {len(group)} original annotations: {original_ids}")
                # Delete original annotations
                for ann in group:
                    # Use print instead of logger, no try-except per request
                    print(f"Attempting to delete annotation: {ann.id}")
                    item.annotations.delete(annotation_id=ann.id)
                    print(f"Deleted annotation: {ann.id}")

                # Create the new combined annotation
                print(f"Creating new combined annotation for promptId '{prompt_id}'...")
                # Ensure system metadata and promptId are preserved/set correctly
                if 'system' not in metadata:
                    metadata['system'] = {}
                metadata['system']['promptId'] = prompt_id
                # Add provenance
                metadata['system']['combinedFrom'] = original_ids
                metadata['system']['automated'] = metadata.get('system', {}).get('automated', True) # Keep automated flag

                original_prompt_item.add(
                    message={
                        "role": "assistant",
                        "content": [{"value": combined_text, "mimetype": dl.PromptType.TEXT}],
                    },
                )
        else:
            print(f"Latest promptId '{latest_prompt_id}' not found in grouped_annotations. No action taken for combining.")

        if not (original_item_id == item.id):
            item.delete() # deletes the temporary item
        print(f"Finished annotation combination for item: {item.id}")
        return original_item

    # --------------------------------------------------------------------- #
    # helpers                                                               #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _get_last_assistant_message(
        prompt_item: dl.PromptItem, item: dl.Item
    ) -> dict | None:
        messages = prompt_item.to_messages(include_assistant=True)
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg
        logger.warning(f"{item.id}: no assistant messages found.")
        return None

    @staticmethod
    def _parse_payload(message: dict, item: dl.Item) -> dict | None:
        """Return the JSON-parsed `content` of a text element or None."""
        text_elem = next(
            (e for e in message.get("content", []) if e.get("type") == "text"), None
        )
        if text_elem is None:
            logger.warning(f"{item.id}: assistant message has no text element.")
            return None

        raw_text = text_elem.get("text", "")
        if not isinstance(raw_text, str) or not raw_text.strip():
            logger.warning(f"{item.id}: assistant text is empty.")
            return None

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error(f"{item.id}: assistant text is not valid JSON:\n{raw_text}")
            return None

        if not {"toolCall", "response"} <= payload.keys():
            logger.error(f"{item.id}: JSON missing required keys - {payload}")
            return None
        return payload

    @staticmethod
    def _replace_annotation(
        prompt_item: dl.PromptItem, item: dl.Item, new_text: str, tool_call: str
    ) -> bool:
        """
        Delete the existing assistant annotation and create a new one with the
        updated text content. Returns True on success.
        """
        # Locate the last assistant prompt metadata
        last_prompt = prompt_item.assistant_prompts[-1]
        ann_id = last_prompt.metadata.get("id")
        if not ann_id:
            logger.error(
                f"{item.id}: assistant prompt has no annotation id - {last_prompt.metadata}"
            )
            return False
        # Delete the old annotation (coordinates are immutable)
        try:
            item.annotations.delete(annotation_id=ann_id)
        except Exception as e:
            logger.warning(f"{item.id}: failed to delete annotation {ann_id} - {e}")
        # Re-add an assistant message via PromptItem API - this handles all
        # required metadata & prompt-key bookkeeping for us.
        model_info = last_prompt.metadata.get("model_info")
        prompt_key = last_prompt.key
        prompt_item.fetch()  # fetch the latest prompt item
        prompt_item.add(
            message={
                "role": "assistant",
                "content": [{"value": new_text, "mimetype": dl.PromptType.TEXT}],
            },
            prompt_key=prompt_key,
            model_info=model_info,
        )
        # Append toolCall metadata if necessary
        if tool_call != "none":
            prompt_item.assistant_prompts[-1].add_element(
                value={"toolCall": tool_call}, mimetype=dl.PromptType.METADATA
            )
        return True
    
def run_prediction(model_id, item: dl.Item, dataset: dl.Dataset):
    """
    Run a prediction on an item using the specified model.
    
    Args:
        model_id (str): The ID of the model to use for prediction
        item (dl.Item): The item to run prediction on
        dataset (dl.Dataset): The dataset containing the item
        
    Returns:
        None
    """
    # --- Get Model ---
    try:
        # You can get by name or ID
        model: dl.Model = dl.models.get(model_id=model_id)
        print(f"Using model: '{model.name}' (ID: {model.id})")

        # --- Run Prediction ---
        print(f"Running prediction on item ID: {item.id} using model '{model.name}'...")
        # The predict method requires a list of item IDs
        prediction_execution = model.predict(item_ids=[item.id])
        print(f"Prediction initiated. Execution ID: {prediction_execution.id}")
        print(f"Execution status: {prediction_execution.status}")

        # You might want to wait for the execution to complete
        prediction_execution.wait()
        print(f"Execution finished with status: {prediction_execution.latest_status['status']}")

    except dl.exceptions.NotFound:
        print(f"Model '{model_id}' not found in project '{item.project.name}'.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
    
    refetched_item = dataset.items.get(item_id=item.id)
    # Load annotations
    annotations = refetched_item.annotations.list()
    if len(annotations) > 0: # MODIFIED LINE
        print(f"\nFound {len(annotations)} annotation(s) on the item (potential prediction result):") # MODIFIED LINE
        # Re-create PromptItem from the updated item to easily see the conversation
        updated_prompt_item = dl.PromptItem.from_item(refetched_item)
        # Print the conversation history including the assistant's response
        messages = updated_prompt_item.to_messages()
        for msg in messages:
            role = msg.get('role', 'unknown')
            content_list = msg.get('content', [])
            if content_list:
                # Assuming text response for simplicity
                text_content = next((c.get('text', '') for c in content_list if c.get('type') == 'text'), '')
                print(f"  {role.capitalize()}: {text_content}")
            else:
                print(f"  {role.capitalize()}: [Empty Content]")

    else:
        print("\nNo annotations found on the item yet. The prediction might still be running or didn't produce annotations.")
        print("You can check the execution status in the Dataloop UI using the Execution ID.")