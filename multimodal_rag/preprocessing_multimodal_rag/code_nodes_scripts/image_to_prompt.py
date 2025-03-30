import dtlpy as dl
import logging
import os
logger = logging.getLogger("image-to-prompt")


class ServiceRunner(dl.BaseServiceRunner):

    def image_to_prompt(self, item: dl.Item, prompt_text: str=None) -> dl.Item:
        """
        Converts an image item into a prompt item.

        This method takes an image item and generates a prompt item that can be used for further processing or analysis.
        It ensures the item is an image and uploads the generated prompt to a specified remote path.

        :param item: The image item to be converted into a prompt.
        :param prompt_text: The text to be added to the prompt.
        :raises ValueError: If the provided item is not an image.
        :return: The newly created prompt item.
        """


        if "image" not in item.mimetype:
            raise ValueError(f"Item id : {item.id} is not an image file! This functions excepts image only.")

        # Create a prompt item
        prompt_item = dl.PromptItem(name=f"{os.path.splitext(item.name)[0]}-prompt")

        content = [{"mimetype": dl.PromptType.IMAGE, "value": dl.items.get(item_id=item.id).stream}]
        if prompt_text is not None:
            content.append({"mimetype": dl.PromptType.TEXT, "value": prompt_text})
            
        prompt_item.add(
            message={
                "role": "user",
                "content": content,
            }
        )
        
        prompt_item = item.dataset.items.upload(
            prompt_item, remote_path='/prompts_from_images', item_metadata={"user": {"original_item_id": item.id}}
        )

        return prompt_item
    
    
if __name__ == "__main__":
    dl.setenv("rc")
    item = dl.items.get(item_id="")
    runner = ServiceRunner()
    runner.image_to_prompt(item=item, prompt_text="generate underlying data table of the figure below")
