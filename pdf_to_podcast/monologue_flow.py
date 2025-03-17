import logging
import dtlpy as dl
from pdf_to_podcast.monologue_prompts import FinancialSummaryPrompts

# Configure logging
logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


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
        # retrieve the summary from the last prompt message
        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        focus = podcast_metadata.get("focus", None)

        messages = prompt_item.to_messages()
        last_message = messages[-1]
        summary = last_message.get("content", [])[0].get("text", None)
        if summary is None:
            raise ValueError("No summary found in the prompt item.")

        item.metadata.get("user", {}).update({"podcast": {"summary": summary}})
        item.update()

        # generate the outline
        documents = [f"Document: {item.filename}\n{summary}"]

        template = FinancialSummaryPrompts.get_template("monologue_multi_doc_synthesis_prompt")
        llm_prompt = template.render(
            focus_instructions=focus if focus is not None else None, documents="\n\n".join(documents)
        )

        new_name = f"{item.filename}_prompt2_summary_to_outline"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )
        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
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
        focus = podcast_metadata.get("focus", None)
        summary = podcast_metadata.get("summary", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", None)
        if summary is None:
            raise ValueError("No summary found in the prompt item. Try running the previous step again.")

        # get the outline from the last prompt
        prompt_item = dl.PromptItem.from_json(item)
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
        new_name = f"{item.filename}_prompt3_outline_to_monologue"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )

        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
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
        # prep prompt item to get the monologue
        prompt_item = dl.PromptItem.from_json(item)

        # get the monologue from the last prompt
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        monologue = last_message.get("content", [])[0].get("text", None)
        if monologue is None:
            raise ValueError("No monologue found in the prompt item.")

        # create the final conversation in JSON format
        template = FinancialSummaryPrompts.get_template("monologue_dialogue_prompt")
        llm_prompt = template.render(monologue=monologue)

        new_name = f"{item.filename}_prompt4_monologue_to_convo_json"
        prompt_item = dl.PromptItem(name=new_name)
        prompt_item.add(
            message={"content": [{"mimetype": dl.PromptType.TEXT, "value": llm_prompt}]}  # role default is user
        )

        new_item = item.dataset.items.upload(prompt_item, remote_name=new_name, remote_path=item.dir, overwrite=True)
        return new_item
