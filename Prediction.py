import torch

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

model_path = './10k reviews results/checkpoint-1500'  # Path to your saved model, replace 'xxxx' with actual checkpoint number
model = DistilBertForSequenceClassification.from_pretrained(model_path)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def predict(texts):
    # Tokenize input
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # Get predictions
    with torch.no_grad():  # Deactivates autograd, reducing memory usage and speeding up computation
        outputs = model(**encodings)

    # Process outputs
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)

    # Convert to list and return
    return probabilities.tolist()

# Input test text and run model
test_input = ["I loved this movie! I would definitely watch it again."]
predictions = predict(test_input)

print(predictions)
