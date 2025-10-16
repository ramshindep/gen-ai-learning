from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3",
    max_tokens=100,
)

llm3vision = ChatOllama(
    model="llama3.2-vision",
    max_tokens=100,
)

def detect_action(prompt: str):
  
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["image", "picture", "draw", "generate image", "create image"]):
        response = llm3vision.invoke(prompt)
        action_type = "IMAGE"
    else:
        response = llm.invoke(prompt)
        action_type = "TEXT"

    return action_type, response


def main():
    user_input = input("Enter your prompt: ")


    action_type, response = detect_action(user_input)

    detailed_prompt = f"The user wants to perform a {action_type.lower()} task. Original input: {user_input}"

    print("\nDetected Action:", action_type)
    print("Detailed Prompt Sent to Ollama:", detailed_prompt)
    print("\nOllama Understanding:\n", response.content)


if __name__ == "__main__":
    main()
