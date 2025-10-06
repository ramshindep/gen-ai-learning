from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm =ChatOllama(
    model="llama3",
    max_tokens=100
)

context=SystemMessage(
    context="You are a java devloper "
)

human_message=HumanMessage("Tell me a joke")
response=llm.invoke([context,human_message])
print(response.content)

ai_message=AIMessage(response.content)


print("---------------------------------------------------------------------------------------")

context=SystemMessage(
    content="You are a Python developer"
)

human_message=HumanMessage("Tell me a joke")
response=llm.invoke([context,human_message,ai_message])
print(response.content)