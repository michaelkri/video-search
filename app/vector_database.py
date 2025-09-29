import chromadb
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import cv2
from loader import NumpyLoader


class Collection:
    '''
    A ChromaDB collection
    '''
    def __init__(self, client: ClientAPI, collection_name: str):
        self.embedding_function = OpenCLIPEmbeddingFunction()
        
        self.frames_collection = client.create_collection(
            name=collection_name
        )

    
    def embed(self, input):
        '''
        Embeds the input
        '''
        return self.embedding_function(input)

    
    def add(self, ids, embeddings, documents):
        '''
        Adds to the collection
        '''
        self.frames_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents
        )
    

    def query(self, search_query: str):
        '''
        Query the collection
        '''
        embedded_search_query = self.embedding_function(search_query)

        frames = self.frames_collection.query(
            query_embeddings=embedded_search_query,
            include=['distances', 'documents']
        )
        return frames