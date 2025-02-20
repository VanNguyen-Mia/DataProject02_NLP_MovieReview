import sys
import os
import streamlit as st
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from classifier import convert_tokens
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from lime import lime_text
import re

def load_vect_and_model():
    text_vectorizer = load("./saved_model/vectorizer.joblib")
    nb_model = load("./saved_model/nb_model.joblib")
    return text_vectorizer, nb_model

text_vectorizer, nb_model = load_vect_and_model()

def vectorize_text(texts):
    text_transformed = text_vectorizer.transform(texts)
    return text_transformed

def pred_class(texts):
    return nb_model.predict(vectorize_text(texts))

def pred_prob(texts):
    return nb_model.predict_proba(vectorize_text(texts))

def create_colored_review(review, word_contributions):
    tokens = convert_tokens(review)
    modified_review = ""
    for token in tokens:
        if token in word_contributions['Word'].values:
            idx = word_contributions['Word'].values.tolist().index(token)
            contribution = word_contributions.iloc[idx]['Contribution']
            modified_review += ":green[{}]".format(token) if contribution > 0 else ":red[{}]".format(token)
            modified_review += " "
        else:
            modified_review += token
            modified_review += " "

explainer = lime_text.LimeTextExplainer(class_names=nb_model.classes_)

st.title("Reviews Classification :green[Positive] vs :red[Negative] :tea: :coffee:")
review  = st.text_area(label="Enter Review Here: ", value = "Enjoy Dashboard", height=200)
submit = st.button("Classify")

if submit and review:
    prediction, probs = pred_class([review,]), pred_prob([review,])
    prediction, probs = prediction[0], probs[0]

    st.markdown("### Prediction: {}".format(prediction))
    st.metric(label="Confidence: ", value="{:.2f}%".format(probs[1]*100 if prediction == "pos" else probs[0]*100))

    explanation = explainer.explain_instance(review, classifier_fn=pred_prob, num_features=50)
    word_contribution = pd.DataFrame(explanation.as_list(), columns=["Word","Contribution"])
    modified_review = create_colored_review(review, word_contribution)
    st.write(modified_review)

    fig = explanation.as_pyplot_figure()
    fig.set_figheight(12)
    st.pyplot(fig, use_container_width=True)
