from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

llm=ChatOllama(
    model="gemma:2b",
    max_tokens=100
)

while True:
    prompt=input("User: ")
    if prompt.lower() == "exit":
        break
    
    
    response=llm.invoke(prompt)
    print("Bot:"+response.content)
    

    