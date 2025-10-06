from langchain_ollama import OllamaLLM

llm=OllamaLLM(
    model="gemma:2b",
    max_tokens=100,
)

prompt="What is the capital of France?"

response=llm.invoke(prompt)
print(response)

