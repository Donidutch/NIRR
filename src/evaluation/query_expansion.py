from pyserini.search import LuceneSearcher
from gensim.models import KeyedVectors

from sentence_transformers import SentenceTransformer, util


def pseudo_relevance_feedback(query, num_docs, num_terms):
    searcher = LuceneSearcher("path/to/index")
    hits = searcher.search(query, num_docs)

    # Gather top documents' terms
    doc_vectors = [
        searcher.doc(hits[i].docid).lucene_document().get("stored_fields")
        for i in range(num_docs)
    ]

    # Extract top terms
    top_terms = []  # Implement extraction of top terms based on frequency or tf-idf

    # Expand original query
    expanded_query = query + " " + " ".join(top_terms[:num_terms])

    return expanded_query


def expand_query_word2vec(query, num_terms):
    model = KeyedVectors.load_word2vec_format(
        "path/to/GoogleNews-vectors-negative300.bin", binary=True
    )
    expanded_terms = model.most_similar(positive=query.split(), topn=num_terms)

    # Add new terms to original query
    expanded_query = query + " " + " ".join([term[0] for term in expanded_terms])

    return expanded_query


def expand_query_bert(query, num_terms):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)  # Your corpus here

    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=num_terms)

    # Add top terms to original query
    expanded_query = query + " " + " ".join([corpus[i] for i in top_results.indices])

    return expanded_query
