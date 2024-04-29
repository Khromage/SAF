import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model_path = './2500 reviews results/checkpoint-330'  # Path to your saved model, replace 'xxxx' with actual checkpoint number
model = DistilBertForSequenceClassification.from_pretrained(model_path)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def predict(texts):
    # Encode the inputs
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # Get predictions
    with torch.no_grad():  # Deactivates autograd, reducing memory usage and speeding up computation
        outputs = model(**encodings)

    # Process outputs (logits to probabilities)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # Convert to list and return
    return probabilities.tolist()


new_texts = ["This movie is okay. I have no strong opinions on it."]
predictions = predict(new_texts)

print(predictions)
