from vector_database import VectorDatabase


db = VectorDatabase()

db.add(
    ids=['1', '2'],
    uris=[
        '/workspaces/video-search/images/car.jpg',
        '/workspaces/video-search/images/house.png'
    ]
)

results = db.query(
    query_texts=['roof'],
    include=['uris', 'distances']
)

print(results)