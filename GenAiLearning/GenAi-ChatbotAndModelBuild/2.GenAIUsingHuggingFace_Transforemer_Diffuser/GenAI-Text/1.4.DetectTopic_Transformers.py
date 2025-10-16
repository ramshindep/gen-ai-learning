from transformers import pipeline,logging
logging.set_verbosity_error()

classifier=pipeline(
    "text-classification",
    model="cardiffnlp/tweet-topic-21-multi",
    device="cuda",
    top_k=5
)

result=classifier("Oh great,another Mondey Morning meeting at 7am")
print(result[0][:5])

