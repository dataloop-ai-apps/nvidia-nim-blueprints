import dtlpy as dl
import tempfile
import logging
import json
import os

logger = logging.getLogger("image-to-prompt")


class ServiceRunner(dl.BaseServiceRunner):

    def post_processing(self, item: dl.Item) -> dl.Item:
        valid_file_types = ["application/json", "text/plain"]
        new_item = None
        remote_path = "/preprocessed_text_files"

        if item.mimetype not in valid_file_types and not item.mimetype.startswith("image"):
            raise ValueError(f"Unsupported item type: {item.mimetype}")

        if item.mimetype.startswith("image"):
            new_item = self._process_image_item(item, remote_path)
        elif item.mimetype == "application/json":
            new_item = self._process_prompt_item(item, remote_path)
        elif item.mimetype == "text/plain":
            new_item = item  # No processing needed

        return new_item

    def _process_image_item(self, item: dl.Item, remote_path: str) -> dl.Item:
        processed_item = None
        # Extract OCR annotations
        filters = dl.Filters(field="type", values="box", resource=dl.FiltersResource.ANNOTATION)
        annotations = item.annotations.list(filters=filters)
        sorted_anns = sorted(annotations, key=lambda ann: (ann.coordinates[0]["y"], ann.coordinates[0]["x"]))
        ocr_lines = [ann.label for ann in sorted_anns if ann.label]

        if ocr_lines != []:
            processed_item = self._upload_temp_text_file(
                item=item,
                content="\n".join(ocr_lines),
                remote_path=remote_path,
                suffix="ocr_postprocessed",
                user_metadata={"text_annotations_from_prompt": True},
            )

        # Extract cached response from metadata
        metadata = item.metadata.get("user", {}).get("cached_response")
        if metadata != {}:
            processed_item = self._upload_temp_text_file(
                item=item,
                content=json.dumps(metadata, indent=2),
                remote_path=remote_path,
                suffix="cached_postprocessed",
                user_metadata={"from_cached_response": True},
            )

        return processed_item

    def _process_prompt_item(self, item: dl.Item, remote_path: str) -> dl.Item:
        filters = dl.Filters(field="type", values="text", resource=dl.FiltersResource.ANNOTATION)
        annotations = item.annotations.list(filters=filters)
        lines = [ann.coordinates for ann in annotations]

        processed_item = None
        if lines != []:
            processed_item = self._upload_temp_text_file(
                item=item,
                content="\n".join(lines),
                remote_path=remote_path,
                suffix="deplot_postprocessed",
                user_metadata={"text_annotations_from_prompt": True},
            )

        return processed_item

    def _upload_temp_text_file(self, item, content: str, remote_path: str, suffix: str, user_metadata: dict) -> dl.Item:
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = f"{os.path.splitext(item.name)[0]}_{suffix}.txt"
            temp_file_path = os.path.join(temp_dir, filename)

            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(content)

            logger.info(f"Uploading processed text to: {temp_file_path}")

            uploaded_item = item.dataset.items.upload(
                remote_name=filename,
                local_path=temp_file_path,
                remote_path=remote_path,
                item_metadata={"user": {"original_item_id": item.id, **user_metadata}},
            )

        return uploaded_item


if __name__ == "__main__":
    dl.setenv("rc")
    item = dl.items.get(item_id="67cef9eff57e50ad019c1822")
    prompt_item = dl.items.get(item_id="67d012514202ed0897b5c857")
    runner = ServiceRunner()
    runner.post_processing(item=item)
    runner.post_processing(item=prompt_item)
