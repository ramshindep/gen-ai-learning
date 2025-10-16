from transformers import pipeline,logging

logging.set_verbosity_error()

classifier=pipeline (
    "text-generation",
    model="openai-community/gpt-2-",
    device="cuda"
)

result=classifier("Oh great,another mondey morning meeting at 7am")
print(result[0]['generated_text'])
