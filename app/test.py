import chromadb
from PIL import Image
from vector_database import Collection
from frame_processor import FrameProcessor


VIDEO_PATH = 'images\BigBuckBunny_320x180.mp4'
SEARCH_QUERY = 'butterfly'


client = chromadb.Client()

frames_collection = Collection(client, 'frames')

frames_processor = FrameProcessor(frames_collection, VIDEO_PATH)

frames_processor.add_frames_to_collection()

results = frames_collection.query(SEARCH_QUERY)

results_image_paths = [metadata['image_path'] for metadata in results['metadatas'][0]]

results_images = [Image.open(path) for path in results_image_paths[:3]]

print(results_images)