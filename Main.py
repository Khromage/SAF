import spacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
import re

df = pd.read_csv(r'Data/IMDB_Dataset.csv')

# Load the English tokenizer and language model
nlp = spacy.load('en_core_web_sm')

# Test for tokenization
test = "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."
tokens = [token.text for token in nlp(test)]
print(tokens)

# Clean up data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens2 = nlp(text)
    tokens2 = [token.text for token in tokens2 if token.text]
    return tokens2

# df['clean_review'] = df['review'].apply(clean_text)
clean_test = clean_text(test)

# Print the tokens
# print(df['clean_review'].iloc[0])
print(clean_test)
