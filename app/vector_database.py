import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import cv2
import os


class VectorDatabase:
    """
    A wrapper around ChromaDB with one collection
    """
    def __init__(self):
        self.client = chromadb.Client()

        self.data_loader = ImageLoader()
        self.embedding_function = OpenCLIPEmbeddingFunction()
        
        self.frames_collection = self.client.create_collection(
            name='frames',
            # embedding_function=self.embedding_function,
            # data_loader=self.data_loader
        )

    
    def add(self, ids, uris):
        self.frames_collection.add(
            ids=ids,
            uris=uris
        )

    
    # def _save_video_frames(self, video_path: str, frame_interval: int = 30):
    #     cap = cv2.VideoCapture(video_path)
    #     output_dir = 'frames'
    #     os.makedirs(output_dir, exist_ok=True)
        
    #     i = 0
    #     ids = []
    #     uris = []
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if not ret:
    #             return
    #         if i % frame_interval == 0:
    #             frame_path = os.path.join(output_dir, f'frame_{i}.jpg')
    #             cv2.imwrite(frame_path, frame)
    #             ids.append(str(i))
    #             uris.append(frame_path)
    #         i += 1

    #     cap.release()
    #     cv2.destroyAllWindows()

    #     # Add images to Chroma collection
    #     self.frames_collection.add(
    #         ids=ids,
    #         uris=uris
    #     )

    #     os.removedirs(output_dir)


    def _save_video_frames(self, video_path: str, frame_interval: int = 30, batch_size: int = 32):
        cap = cv2.VideoCapture(video_path)
        
        # Store embeddings in batches
        i = 0
        while True:
            batch_end_frame = i + frame_interval * batch_size
            ids = []
            frames = []

            while i < batch_end_frame and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    return
                if i % frame_interval == 0:
                    ids.append(str(i))
                    frames.append(frame)
                i += 1

            if ids:
                # Embed batch frames
                embedded_frames = self.embedding_function(frames)

                # Add embedded frames to Chroma collection
                self.frames_collection.add(
                    ids=ids,
                    embeddings=embedded_frames
                )

                print(f'Added batch {batch_end_frame // (frame_interval * batch_size)}')

            # End of video
            if not cap.isOpened():
                break

        cap.release()
        cv2.destroyAllWindows()

        print(self.frames_collection.count())


    def query(self, search_terms):
        results = self.frames_collection.query(
            query_texts=[f'{search_terms}'],
            include=['uris', 'distances']

        )
        paths_with_distances = zip(results['uris'][0], map(str, results['distances'][0]))
        return paths_with_distances
    

    def search(self, video_path, search_terms):
        self._save_video_frames(video_path)

        # return self.query(search_terms)
        frames = self.frames_collection.get(
            include=['embeddings']
        )
        return frames