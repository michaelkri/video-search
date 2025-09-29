import chromadb
from vector_database import Collection
from frame_processor import FrameProcessor


VIDEO_PATH = 'images\BigBuckBunny_320x180.mp4'
SEARCH_QUERY = 'butterfly'


client = chromadb.Client()

frames_collection = Collection(client, 'frames')

frames_processor = FrameProcessor(frames_collection, VIDEO_PATH)

frames_processor.add_frames_to_collection()

results = frames_collection.query(SEARCH_QUERY)

print(results)