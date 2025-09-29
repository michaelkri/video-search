from vector_database import VectorDatabase


db = VectorDatabase()

results = db.search('images\BigBuckBunny_320x180.mp4', '')

print(results)