import pandas as pd
import re
import spacy

def remove_numbers(text):
    return re.sub('[0-9]+', '', text)
def remove_html_tags(text):
    return re.sub(r'<[^>]+>', '', text)
def remove_symbols(text):
    return re.sub("[!@#$%^&*(){}£\/'']",'',text)

def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)

df=pd.read_csv('IMDB Dataset.csv')

df['review']= df['review'].apply(lambda x: remove_numbers(x))
df['review']= df['review'].apply(lambda x: remove_html_tags(x))
df['review']= df['review'].apply(lambda x: remove_symbols(x))

nlp = spacy.load("en_core_web_sm")

df['preprocessed_review'] = df['review'].apply(preprocess)
df.to_csv('preprocessed_data.csv', index=False)