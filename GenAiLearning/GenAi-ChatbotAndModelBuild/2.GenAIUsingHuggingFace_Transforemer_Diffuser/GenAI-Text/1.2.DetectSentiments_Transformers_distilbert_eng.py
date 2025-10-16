from transformers import pipeline,logging

logging.set_verbosity_error()

classifier=pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device="cuda"
)

result=classifier("Oh great,another morning meeting at 7am")
print(result)
result=classifier(" The customer support was so helpful -they put me on hold for an hour .")

print(result)