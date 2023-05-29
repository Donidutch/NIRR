import re
from collections import Counter

import nltk

# import nltk
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyserini.search import LuceneSearcher
from sentence_transformers import util

# You might need to download the 'punkt' package if it's not already installed
nltk.download("punkt")
nltk.download("stopwords")


class QueryExpander:
    def __init__(self, query: str):
        self.query = query

    def expand_with_prf(self):
        # Implement Pseudo Relevance Feedback technique to expand query
        pass

    def expand_with_word2vec(self):
        # Implement Word2Vec technique to expand query
        pass

    def expand_with_bert(self):
        # Implement BERT technique to expand query
        pass


def pseudo_relevance_feedback(index_path, query, num_docs, num_terms):
    searcher = LuceneSearcher(index_path)
    hits = searcher.search(query, num_docs)

    # Gather top documents' terms
    doc_contents = [searcher.doc(hits[i].docid).contents() for i in range(num_docs)]

    # Filter out None elements from doc_contents
    doc_contents = [content for content in doc_contents if content is not None]

    # Remove XML tags and tokenize
    tokens = [word_tokenize(re.sub("<.*?>", "", content)) for content in doc_contents]
    flat_tokens = [item for sublist in tokens for item in sublist]  # Flatten the list
    term_freqs = Counter(flat_tokens)

    # Extract top terms excluding stop words
    stop_words = set(stopwords.words("english"))
    top_terms = [
        term for term, freq in term_freqs.most_common() if term not in stop_words
    ][:num_terms]

    return f"{query} " + " ".join(top_terms)


def expand_query_word2vec(model, query, num_terms):
    words_in_vocab = [word for word in query.split() if word in model.key_to_index]

    expanded_terms = model.most_similar(positive=words_in_vocab, topn=num_terms)
    return f"{query} " + " ".join([term[0] for term in expanded_terms])


def expand_query_bert(model, query, num_terms, corpus):
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    num_terms_to_retrieve = min(num_terms, cos_scores.shape[0])
    top_results = torch.topk(cos_scores, k=num_terms_to_retrieve)

    return f"{query} " + " ".join([corpus[i] for i in top_results.indices])
