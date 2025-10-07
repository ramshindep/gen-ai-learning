from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="gemma:2b")

def detect_action(prompt: str):
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["image", "picture", "draw", "generate image", "create image"]):
        return "IMAGE"
    elif any(word in prompt_lower for word in ["audio", "sound", "speech", "convert to audio"]):
        return "AUDIO"
    elif any(word in prompt_lower for word in ["translate", "translation", "from english", "to marathi"]):
        return "TRANSLATION"
    else:
        return "TEXT"

def main():
    user_input = input("Enter your prompt: ")

    action_type = detect_action(user_input)

    detailed_prompt = f"The user wants to perform a {action_type.lower()} task. Original input: {user_input}"

    response = llm.invoke([HumanMessage(content=detailed_prompt)])

    print("Detected Action:", action_type)
    print("Detailed Prompt Sent to Ollama:", detailed_prompt)
    print("Ollama Understanding:\n", response.content)

if _name_ == "_main_":
    main()