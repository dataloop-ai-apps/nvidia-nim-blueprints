import dtlpy as dl

VIDEO_ITEM_ID = "69ba5c8cc8e939b69fbe65da"
MODEL_ID = "69b962c8b431d995c82ac27b"
PROMPT_UPLOAD_DIR = "/test_prompts"
STREAM_URL_TEMPLATE = "https://gate.dataloop.ai/api/v1/items/{item_id}/stream"

PROMPTS = [
    # 0 - Current prompt (baseline)
    "Analyze this video and provide a detailed, search-friendly description. "
    "Include: (1) Key objects, people, and entities visible; "
    "(2) Actions, movements, and events occurring; "
    "(3) Scene setting, location type, and environment; "
    "(4) Any text, signs, or identifiable information; "
    "(5) Notable changes or transitions throughout the video. "
    "Be specific and factual — mention colors, positions, counts, and directions where applicable.",

    # 1 - Chronological narration: forces the model to describe events in order
    "Watch this video from start to finish and narrate everything that happens in chronological order. "
    "For each distinct moment, describe: who or what is visible, what actions are being performed, "
    "where it takes place, and any notable details such as text, labels, equipment, or environmental conditions. "
    "Use timestamps or sequence markers like 'first', 'then', 'next', 'finally' to structure your description. "
    "Be exhaustive — do not skip any visible activity, object, or change in the scene.",

    # 2 - Structured sections: produces organized output under clear headings
    "Provide a comprehensive description of this video organized into the following sections:\n"
    "SETTING: Describe the location, environment, lighting, and time of day.\n"
    "SUBJECTS: List all people, animals, or machines visible and describe their appearance.\n"
    "ACTIONS: Describe every action and event that occurs, in the order they happen.\n"
    "OBJECTS: List all notable objects, tools, equipment, signs, or text visible.\n"
    "SUMMARY: Write a one-paragraph summary of the entire video suitable for search indexing.",

    # 3 - Journalist style: who/what/where/when/why framing for factual coverage
    "You are a journalist documenting this video. Describe what you observe by answering: "
    "Who is in the video? What are they doing? Where does this take place? "
    "When do key events happen relative to each other? Why might these actions be occurring? "
    "Report only what is directly visible or strongly implied. "
    "Be precise with quantities, positions, colors, and spatial relationships. "
    "Describe the full duration of the video, not just the first frame.",

    # 4 - Dense captioning: maximize information density per sentence
    "Generate a dense, information-rich caption for this video. "
    "Every sentence must introduce new factual information — no filler or repetition. "
    "Cover: all visible entities and their attributes, all actions and interactions, "
    "the physical environment and spatial layout, any on-screen text or signage, "
    "and how the scene evolves over time. "
    "Aim for maximum detail in minimum words. Write in present tense.",

    # 5 - Q&A self-prompting: model asks and answers its own questions
    "Analyze this video by asking and answering the following questions about it:\n"
    "1. What type of scene or activity is shown?\n"
    "2. Who or what are the main subjects, and what do they look like?\n"
    "3. What specific actions or events take place, and in what order?\n"
    "4. What objects, tools, or equipment are visible?\n"
    "5. What is the setting — indoor/outdoor, location type, lighting, weather?\n"
    "6. Is there any text, signage, or branding visible?\n"
    "7. What changes or transitions occur throughout the video?\n"
    "Answer each question with specific, factual details based only on what is visible.",
]


def create_prompt_item(dataset: dl.Dataset, video_item: dl.Item, instruction: str) -> dl.Item:
    video_stream_url = STREAM_URL_TEMPLATE.format(item_id=video_item.id)
    prompt_item = dl.PromptItem(name=f"test-prompt-{video_item.id[:12]}-{PROMPTS.index(instruction)}")
    message = {
        "role": "user",
        "content": [
            {
                "mimetype": dl.PromptType.TEXT,
                "value": f"{instruction} [video_url]({video_stream_url})"
            }
        ]
    }
    prompt_item.add(message=message)
    uploaded = dataset.items.upload(prompt_item, remote_path=PROMPT_UPLOAD_DIR)
    print(f"  Uploaded prompt item: {uploaded.id} ({uploaded.name})")
    return uploaded


def predict_and_get_response(model: dl.Model, prompt_item: dl.Item) -> str:
    ex = model.predict(item_ids=[prompt_item.id])
    ex.wait()
    updated = dl.items.get(item_id=prompt_item.id)
    annotations = updated.annotations.list()
    for annotation in annotations:
        if annotation.type == "text":
            return annotation.coordinates
    return "[no annotation found]"


def main():
    video_item = dl.items.get(item_id=VIDEO_ITEM_ID)
    model = dl.models.get(model_id=MODEL_ID)
    dataset = video_item.dataset

    print(f"Video: {video_item.name} ({video_item.id})")
    print(f"Model: {model.name} ({model.id})")
    print(f"Testing {len(PROMPTS)} prompt(s)\n")

    results = []
    for i, instruction in enumerate(PROMPTS):
        print(f"--- Prompt {i + 1}/{len(PROMPTS)} ---")
        print(f"Instruction: {instruction[:100]}...")
        prompt_item = create_prompt_item(dataset, video_item, instruction)
        print("  Waiting for prediction...")
        response = predict_and_get_response(model, prompt_item)
        results.append((instruction, response))
        print(f"  Response: {response[:200]}...\n")

    print("=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    for i, (instruction, response) in enumerate(results):
        print(f"\n--- Prompt {i + 1} ---")
        print(f"Instruction: {instruction}")
        print(f"\nResponse:\n{response}")
        print("-" * 80)


if __name__ == "__main__":
    main()
