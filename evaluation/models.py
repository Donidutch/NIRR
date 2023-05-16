from pyserini.search.lucene import LuceneSearcher


class Model:
    def __init__(self, index_path, k_hits=10):
        self.searcher = LuceneSearcher(index_path)
        self.k = k_hits
        self.k1 = 0.9
        self.b = 0.6
        self.mu = 1000

    def search(self, queries, qids):
        return self.searcher.batch_search(queries, qids, k=self.k)

    def set_bm25_parameters(self, k1, b):
        self.searcher.set_bm25(k1, b)

    def set_qld_parameters(self, mu):
        self.searcher.set_qld(mu)
