import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

client = chromadb.Client()

data_loader = ImageLoader()
embedding_function = OpenCLIPEmbeddingFunction()

collection = client.create_collection(
    name='frames',
    embedding_function=embedding_function,
    data_loader=data_loader
)

collection.add(
    ids=['1', '2'],
    uris=['/workspaces/video-search/images/car.jpg', '/workspaces/video-search/images/house.png']
)

results = collection.query(
    query_texts=['roof'],
    include=['uris', 'distances']
)

print(results)