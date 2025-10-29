from transformers import pipeline

classifier=pipeline(task="image-classification", model="google/vit-base-patch16-224")

result=classifier("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")

print(result)