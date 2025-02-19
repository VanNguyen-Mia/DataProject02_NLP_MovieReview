import joblib
from src.classifier import load_data, train_test_split, convert_tokens, vectorize_text, train_naive_bayes, evaluate_model, explain_prediction
# ----------------- Run Pipeline ----------------- #
def main():
    # Load data
    train_path = "./datasets/train"
    categories = ['neg', 'pos']
    
    df = load_data(train_path, categories)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['content'], df['category'], stratify=df['category'], train_size=0.8, random_state=42
    )

    # Preprocess text
    X_train_vect = X_train.apply(lambda row: convert_tokens(row, verbose=False))
    X_test_vect = X_test.apply(lambda row: convert_tokens(row, verbose=False))

    X_train_vect = [" ".join(tokens) for tokens in X_train_vect]
    X_test_vect = [" ".join(tokens) for tokens in X_test_vect]

    # Vectorize text
    X_train_df, X_test_df, tfidf_vectorizer = vectorize_text(X_train_vect, X_test_vect)

    # Train model
    model = train_naive_bayes(X_train_df, y_train)

    # Evaluate model
    evaluate_model(model, X_test_df, y_test)

    # Explain predictions
    explain_prediction(model, tfidf_vectorizer, X_test, y_test)

if __name__ == "__main__":
    main()

