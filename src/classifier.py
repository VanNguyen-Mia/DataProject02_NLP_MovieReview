import joblib
import os
import random
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lime import lime_text

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# ----------------- Step 1: Load Data ----------------- #
def is_git_lfs_file(file_path: str) -> bool:
    """Check if a file is a Git LFS pointer

    Args:
        file_path (str): a file path to the text content file

    Returns:
        bool: True if first line is a git LFS pointer, False otherwise
    """
    with open(file_path, "r", encoding="utf-8") as file:
        first_line = file.readline().strip()
        return first_line.startswith("version https://git-lfs.github.com")

def load_data(train_path: str, categories: list) -> pd.DataFrame:
    """Load text data from directories and skip Git LFS files

    Args:
        train_path (str): a path to the folder of training data
        categories (list): list of categories to classify data

    Returns:
        Dataframe: dataframe with two columns: 'content' and 'category'
    """
    # """Load text data from directories and skip Git LFS files."""
    data = []
    for category in categories:
        category_path = os.path.join(train_path, category)
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            if is_git_lfs_file(file_path):  # Skip LFS pointers
                continue
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
            data.append((content, category))
    return pd.DataFrame(data, columns=['content', 'category'])

# ----------------- Step 2: Preprocess Text ----------------- #
def convert_tokens(text: str, verbose=False) -> list:
    """Tokenization, decapitalization, stopword removal, lemmatization, and filtering

    Args:
        text (str): the text to be processed
        verbose (bool): whether logging messages shoud be displayed

    Returns:
        list: a list of tokenized words from the original text
    """
    # 1. Tokenization
    pattern = r'\w+'
    tokenizer = RegexpTokenizer(pattern)
    token_words = tokenizer.tokenize(text)
    if (verbose):
        print('Tokens:'+str(token_words[0:10]))
    
    # 2. Decapitalization
    decap_token_words = [word.lower() for word in token_words]
    if (verbose):
        print("Decapitalized tokens:" + str(decap_token_words[0:10]))
    
    # 3. Remove stop words
    stopwords_nltk_en = set(stopwords.words('english'))
    rmsw_token_words = ([word for word in decap_token_words if word.lower() not in stopwords_nltk_en])
    if (verbose):
        print('Stopwords removed:' + str(rmsw_token_words[0:20]))
    
    # 4. Remove CAP words
    rmcap_token_words = []
    for word in rmsw_token_words:
        if word.isupper():
            rmcap_token_words.append(word.title())
        else:
            rmcap_token_words.append(word)
    if (verbose):
        print('CAPITALIZED removed:' + str(rmcap_token_words[0:20]))
    
    # 5. Remove salutation
    salutation = ["I", 'it', 'as','mr', 'mrs', 'ms', 'dr', 'phd', 'prof', 'rev']
    rmsalu_token_words = ([word for word in rmcap_token_words if word.lower() not in salutation])
    if (verbose):
        print('Salutation removed:' + str(rmsalu_token_words[0:20]))
    
    # 6. Define transfer tag function:
    def transfer_tag(treebank_tag):
        treebank_tag = treebank_tag.lower()
        if treebank_tag.startswith('j'):
            return "a"
        elif treebank_tag.startswith('v'):
            return "v"
        elif treebank_tag.startswith('n'):
            return 'n'
        elif treebank_tag.startswith('r'):
            return 'r'
        else:
            return 'n'
    
    # 7. Lemmatization
    wnl = WordNetLemmatizer()

    lemma_words = []
    for word, tag in nltk.pos_tag(rmsalu_token_words):
        firstletter = tag[0].lower()
        wtag = transfer_tag(firstletter)
        if not wtag:
            lemma_words.extend([word])
        else:
            lemma_words.extend([wnl.lemmatize(word, wtag)])
    if verbose:
        print('Lemma:' + str(lemma_words[0:10]))
    
    # 8. English words
    eng_words = [word for word in lemma_words if len(wn.synsets(word.lower())) > 1]

    # 9 Remove numbers
    rmnb_token_words = ([word for word in eng_words if not word.isdigit()])
    if (verbose):
        print('Number removed:' + str(rmnb_token_words[0:20]))
    
    return rmnb_token_words

# ----------------- Step 3: Feature Engineering ----------------- #
def vectorize_text(X_train: list, X_test: list, vectorizer_path:str) -> pd.DataFrame:
    """Convert text to TF-IDF vectors and save tfidf_vectorizer

    Args:
        X_train (pd.Series): A list of strings (preprocessed texts) for training
        X_test (pd.Series): A list of strings (preprocessed texts) for testing
        vectorizer_path (str): path to save vectorizer
    Returns:
        pd.Dataframe: dataframes with TF-IDF vectors
    """
    tfidf_vectorizer = TfidfVectorizer(norm=None)
    X_train_vect = tfidf_vectorizer.fit_transform(X_train)
    X_test_vect = tfidf_vectorizer.transform(X_test)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    return (
        pd.DataFrame(X_train_vect.toarray(), columns=tfidf_vectorizer.get_feature_names_out()),
        pd.DataFrame(X_test_vect.toarray(), columns=tfidf_vectorizer.get_feature_names_out()),
        tfidf_vectorizer
    )

# ----------------- Step 4: Train Model ----------------- #
def train_naive_bayes(X_train_df: pd.DataFrame, y_train: pd.Series):
    """Train a Naive Bayes model with hyperparameter tuning -> return the best NB model

    Args:
        X_train_df (pd.Dataframe): Training dataframe with TF-IDF vectors
        y_train (pd.Series): A training series contaning labels

    Returns:
        best Naive Bayes model
    """
    param_grid = {'alpha': [0.01, 0.1, 1, 10]}  # Smoothing parameter
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_df, y_train)
    
    print("Best Alpha:", grid_search.best_params_['alpha'])
    print("Best Validation Accuracy:", grid_search.best_score_)
    
    return grid_search.best_estimator_

# ----------------- Step 5: Evaluate and Save Model ----------------- #
def evaluate_model(model, X_test_df: pd.DataFrame, y_test: pd.Series, model_path: str):
    """Evaluate model performance, print metrics and save model to model_path

    Args:
        model: best model returned from the train_naive_bayes function
        X_test_df (pd.Dataframe): Testing dataframe with TF-IDF vecors
        y_test (pd.Series): A testing series containing labels
        model_path (str): path to save the model
    """
    y_pred = model.predict(X_test_df)
    
    print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # save the trained model:
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# ----------------- Step 6: Explain Predictions ----------------- #
def explain_prediction(model, tfidf_vectorizer, X_test, y_test):
    """Explain a random prediction using Lime

    Args:
        model: the trained classification model
        tfidf_vectorizer: a fitted TfidfVectorizer
        X_test (pd.Series): a series for testing 
        y_test (pd.Series): a series of labels 
    """
    explainer = lime_text.LimeTextExplainer(class_names=model.classes_)
    
    def pred_fn(text):
        text_vectorized = tfidf_vectorizer.transform(text)
        return model.predict_proba(text_vectorized)

    idx = random.randint(0, len(X_test) - 1)
    
    print(f"Actual Text: {X_test.iloc[idx]}")
    print(f"Prediction: {model.predict(tfidf_vectorizer.transform([X_test.iloc[idx]]))[0]}")
    print(f"Actual: {y_test.iloc[idx]}")
    
    explanation = explainer.explain_instance(X_test.iloc[idx], classifier_fn=pred_fn, num_features=10)
    explanation.show_in_notebook()




