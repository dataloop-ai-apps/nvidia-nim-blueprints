import dtlpy as dl
import logging

from services.AgentService.dialogue_prompts import PodcastPrompts
from services.AgentService.monologue_prompts import FinancialSummaryPrompts

logger = logging.getLogger("[NVIDIA-NIM-BLUEPRINTS]")


class MonologueService(dl.BaseServiceRunner):
    def monologue_generate_outline(item: dl.Item, progress: dl.Progress, context: dl.Context):
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
        # item = dl.items.get(item_id="6758139d45821d442ba1f6e1") # DEBUG
        # buffer = item.download(save_locally=False)
        # try:
        #     text = buffer.read().decode('utf-8')
        # except Exception as e:
        #     logger.error(f"Error decoding the item: {e}. Please check the item is a prompt item and the prompt contains text.")
        #     raise e

        # prep prompt item to get the summary
        prompt_item = dl.PromptItem.from_json(item)

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        guide = podcast_metadata.get("guide", None)

        messages = prompt_item.to_messages()
        last_message = messages[-1]
        summary = last_message.get("content", [])[0].get("text", None)

        if summary is None:
            raise ValueError("No summary found in the prompt item.")

        # generate the outline
        documents = [f"Document: {item.filename}\n{summary}"]

        template = FinancialSummaryPrompts.get_template("monologue_multi_doc_synthesis_prompt")
        llm_prompt = template.render(
            focus_instructions=guide if guide is not None else None, documents="\n\n".join(documents)
        )

        prompt = dl.Prompt(key="2")  # "2_summary_to_outline")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)

        prompt_item.prompts.append(prompt)

        return prompt_item

    def monologue_generate_monologue(
        item: dl.Item, progress: dl.Progress, context: dl.Context, prompt_guide: str = None
    ):
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
        prompt_item = dl.PromptItem.from_json(item)

        podcast_metadata = item.metadata.get("user", {}).get("podcast", None)
        guide = podcast_metadata.get("guide", None)
        speaker_1_name = podcast_metadata.get("speaker_1_name", None)

        # get the outline from the last prompt
        messages = prompt_item.to_messages()
        last_message = messages[-1]
        outline = last_message.get("content", [])[0].get("text", None)
        if outline is None:
            raise ValueError("No outline found in the prompt item.")

        # get the summary docs context, from first prompt
        summary = messages[0].get("content", [])[0].get("text", None)
        if summary is None:
            try:
                summary = podcast_metadata.get("summary")
            except Exception as e:
                raise ValueError("No summary found in the prompt item.")

        documents = [f"Document: {item.filename}\n{summary}"]

        # generate the monologue
        template = FinancialSummaryPrompts.get_template("monologue_transcript_prompt")
        llm_prompt = template.render(
            raw_outline=outline,
            documents=documents,
            focus=guide if guide else "key financial metrics and performance indicators",
            speaker_1_name=speaker_1_name,
        )

        # add the prompt to the prompt item
        prompt = dl.Prompt(key="3")  # "3_outline_to_monologue")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)

        prompt_item.prompts.append(prompt)
        return prompt_item

    def monologue_create_convo_json(item: dl.Item, progress: dl.Progress, context: dl.Context):
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
        template = FinancialSummaryPrompts.get_template("monologue_convo_json_prompt")
        llm_prompt = template.render(monologue=monologue)

        prompt = dl.Prompt(key="4")  # "4_monologue_to_convo_json")
        prompt.add_element(mimetype=dl.PromptType.TEXT, value=llm_prompt)

        prompt_item.prompts.append(prompt)

        return prompt_item


if __name__ == "__main__":
    dl.setenv("prod")
    item = dl.items.get(item_id="6758139d45821d442ba1f6e1")
    MonologueService.monologue_generate_outline(item, None, None)
    MonologueService.monologue_generate_monologue(item, None, None)
