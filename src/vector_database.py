import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


class VectorDatabase:
    """
    A wrapper around ChromaDB with one collection
    """
    def __init__(self):
        self.client = chromadb.Client()

        self.data_loader = ImageLoader()
        self.embedding_function = OpenCLIPEmbeddingFunction()
        
        self.collection = self.client.create_collection(
            name='frames',
            embedding_function=self.embedding_function,
            data_loader=self.data_loader
        )

    def add(self, ids, uris):
        self.collection.add(
            ids=ids,
            uris=uris
        )

    def query(self, query_texts, include = None):
        results = self.collection.query(
            query_texts=query_texts,
            include=include
        )

        return results