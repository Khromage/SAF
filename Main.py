import spacy
import pandas as pd

df = pd.read_csv(r'IMDB_Dataset.csv')
df_copy = df.copy()

# Load the English tokenizer and language model
nlp = spacy.load('en_core_web_trf')

# Test for tokenization
test = "I love love, what the frick, dog water, long-short, crapbucket, U.K., the is"
tokens = [token.text for token in nlp(test)]
print(tokens)

<<<<<<< Updated upstream
# cleanTest = [token.text.lower() for token in test if not token.is_punct and not token.is_stop]
# print(cleanTest)
=======
# Clean up data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    doc = nlp(text)
    tokens = [(token.text, token.pos_) for token in doc if token.text]

    return tokens
>>>>>>> Stashed changes

# Process the text with spaCy
# doc = nlp(df['IMDB Dataset'])

<<<<<<< Updated upstream
# Clean tokens
=======
first_3_rows = df.head(3)

df['clean_review'] = df['review'].apply(clean_text)
# clean_test = clean_text(test)

print(first_3_rows)

# Print the tokens
# print(df['clean_review'].iloc[0])
# print(clean_test)
>>>>>>> Stashed changes
