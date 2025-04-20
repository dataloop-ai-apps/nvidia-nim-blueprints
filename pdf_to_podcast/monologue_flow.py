import json
import logging
import dtlpy as dl

from pathlib import Path
from pdf_to_podcast.podcast_types import Conversation  # Podcast conversation data structures
from pdf_to_podcast.monologue_prompts import FinancialSummaryPrompts
from pdf_to_podcast.shared_functions import SharedServiceRunner

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")

SPEAKER_1_NAME = "Alex"

class MonologueServiceRunner(dl.BaseServiceRunner):
    @staticmethod
    def generate_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate an outline from the pdf text summary

        Args:
            item (dl.Item): Dataloop item containing the podcast summary
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        # get the podcast metadata from the item
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        if podcast_metadata is None:
            raise ValueError("No podcast metadata found in the prompt item. Try running the previous step again.")
        focus = podcast_metadata.get("focus", None)
        pdf_name = podcast_metadata.get("pdf_name", None)

        # get the summary from the last prompt annotation
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        summary = last_message.get("content", [])[0].get("text", None)
        if summary is None:
            raise ValueError("No text summary found in the prompt item. Try running the previous step again.")

        logger.info("Preparing to generate outline")

        # create summary file
        summary_filename = f"{Path(pdf_name).stem}_summary.txt"
        with open(summary_filename, "w", encoding='utf-8') as f:
            f.write(summary)

        summary_item = item.dataset.items.upload(
            local_path=summary_filename,
            remote_name=summary_filename,
            remote_path=item.dir,
            overwrite=True,
            item_metadata={"user": item.metadata['user']},
        )
        logger.info(f"Saved PDF summary as text item {summary_item.id}")

        # retreieve documents for context
        if podcast_metadata.get("with_references", None) is None:
            documents = [f"Document: {item.filename}\n{summary}"]
        else:
            documents = [f"Document: {item.filename}\n{summary}"]
            # TODO support multiple documents as context
            # # get all documents 
            # reference_docs = MonologueServiceRunner._get_references(item)
            # documents = [f"Document: {item.filename}\n{summary}" for doc in reference_docs]

        template = FinancialSummaryPrompts.get_template("monologue_multi_doc_synthesis_prompt")
        llm_prompt = template.render(
            focus_instructions=focus if focus is not None else None, documents="\n\n".join(documents)
        )

        new_name = f"{Path(pdf_name).stem}_prompt2_summary_to_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_metadata = item.metadata
        new_metadata["user"]["podcast"]["summary_item_id"] = summary_item.id
        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=new_metadata
        )
        return new_item

    @staticmethod
    def generate_monologue(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Generate a monologue from the outline

        Args:
            item (dl.Item): Dataloop item containing the outline
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        # prep prompt item to get the outline
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        if podcast_metadata is None:
            raise ValueError("No podcast metadata found in the prompt item. Try running the previous step again.")

        focus = podcast_metadata.get("focus", None)
        pdf_name = podcast_metadata.get("pdf_name", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", SPEAKER_1_NAME)

        # get the summary from the summary item
        summary_item_id = podcast_metadata.get("summary_item_id", None)
        try:
            summary = SharedServiceRunner._get_summary_text(summary_item_id)
        except ValueError as e:
            raise ValueError(f"Error getting summary text: {e} for item {item.id}")

        logger.info("Preparing to generate monologue")

        # get the outline from the last respoinse
        prompt_item = dl.PromptItem.from_item(item)
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        outline = last_message.get("content", [])[0].get("text", None)
        if outline is None:
            raise ValueError("No outline found in the prompt item.")

        documents = [f"Document: {item.filename}\n{summary}"]

        template = FinancialSummaryPrompts.get_template("monologue_transcript_prompt")
        llm_prompt = template.render(
            raw_outline=outline,
            documents=documents,
            focus=focus if focus else "key financial metrics and performance indicators",
            speaker_1_name=speaker_1_name,
        )

        # add the prompt to the prompt item
        new_name = f"{Path(pdf_name).stem}_prompt3_outline_to_monologue"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )

        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=item.metadata
        )
        return new_item

    @staticmethod
    def create_convo_json(item: dl.Item, progress: dl.Progress, context: dl.Context):
        """
        Create a final conversation from the monologue in JSON format

        Args:
            item (dl.Item): Dataloop item containing the monologue
            progress (dl.Progress): Dataloop progress object
            context (dl.Context): Dataloop context object

        Returns:
            item (dl.Item): the prompt item
        """
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        pdf_name = podcast_metadata.get("pdf_name", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", SPEAKER_1_NAME)

        # prep prompt item to get the monologue
        prompt_item = dl.PromptItem.from_item(item)

        # get the monologue from the last prompt
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        monologue = last_message.get("content", [])[0].get("text", None)
        if monologue is None:
            raise ValueError("No monologue found in the prompt item. Try running the previous step again.")

        logger.info("Preparing to generate final monologue transcript JSON")

        # create the final conversation in JSON format
        schema = Conversation.model_json_schema()
        template = FinancialSummaryPrompts.get_template("monologue_dialogue_prompt")
        llm_prompt = template.render(speaker_1_name=speaker_1_name, text=monologue, schema=json.dumps(schema, indent=2))

        new_name = f"{Path(pdf_name).stem}_prompt4_monologue_to_convo_json"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )

        new_item = item.dataset.items.upload(
            prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True, item_metadata=item.metadata
        )
        return new_item


if __name__ == "__main__":
    env = "prod"

    dl.setenv(env)
    progress = dl.Progress()
    context = dl.Context()

    # test one function
    # processed_item = dl.items.get(item_id="67f39b649d79af1780847add")  #prompt item for db-context pdf

    # outline = MonologueServiceRunner.generate_outline(processed_item, progress, context)
    # print(f"2/5: Successfully generated outline: {outline.name} ({outline.id})")
    # print(f"Link here: {outline.url}")

    outline_item = dl.items.get(item_id="67f3d39c17f211184f0beced")
    MonologueServiceRunner.create_convo_json(outline_item, progress, context)
