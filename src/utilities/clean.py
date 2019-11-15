

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src import configs
clean_config = configs.clean

def tokenize(doc_array, tokenizer_type, model_config):

    print(f'begin tokenize {tokenizer_type}')
    if tokenizer_type == 'count':
        vectorizer = CountVectorizer(
            stop_words='english',
            ngram_range=model_config.count__ngram_range
        )
    
    elif tokenizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=model_config.tfidf__ngram_range
        )
    
    doc_words = vectorizer.fit_transform(doc_array)
    print(f'after tokenizer {tokenizer_type} shape: ', doc_words.shape)

    return vectorizer, doc_words
