import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def transform_ciphertext_for_ml(df):
    # lists to store features
    char_freq_features = []
    bigram_features = []
    trigram_features = []
    positional_features = []

    # create a CountVectorizer for n-grams, ignoring empty vocabulary issues
    bigram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2), token_pattern=r"(?u)\b\w+\b")
    trigram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3), token_pattern=r"(?u)\b\w+\b")

    for ciphertext in df['ciphertext']:
        # skip empty strings or strings with only whitespace just in case i didnt prune them out
        if not ciphertext.strip():
            continue

        # char freq
        char_counts = Counter(ciphertext)
        char_freq_features.append(char_counts)

        # n-grams (bigrams and trigrams)
        if len(ciphertext) > 1:
            bigram_counts = bigram_vectorizer.fit_transform([ciphertext]).toarray().flatten()
        else:
            bigram_counts = [0] * bigram_vectorizer.get_feature_names_out().shape[0]

        if len(ciphertext) > 2:
            trigram_counts = trigram_vectorizer.fit_transform([ciphertext]).toarray().flatten()
        else:
            trigram_counts = [0] * trigram_vectorizer.get_feature_names_out().shape[0]

        bigram_features.append(bigram_counts)
        trigram_features.append(trigram_counts)

        # pos info
        positions = {char: i for i, char in enumerate(ciphertext)}
        positional_features.append(positions)

    # list of dicts --> df
    char_freq_df = pd.DataFrame(char_freq_features).fillna(0)
    bigram_df = pd.DataFrame(bigram_features)
    trigram_df = pd.DataFrame(trigram_features)
    positional_df = pd.DataFrame(positional_features).fillna(-1)  # -1 indicates character not present

    combined_features_df = pd.concat([char_freq_df, bigram_df, trigram_df, positional_df], axis=1)

    return combined_features_df

if __name__ == '__main__':
    df = pd.read_csv('data_collection/pruned_substitution_cipher_dataset.csv')
    transformed_features_df = transform_ciphertext_for_ml(df)
    output_path = 'feature_engineering/transformed_ciphertext_features.csv'
    transformed_features_df.to_csv(output_path, index=False)
    print("done did it")
