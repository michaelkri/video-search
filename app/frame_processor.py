import cv2
import numpy as np
import base64
from vector_database import Collection


class FrameProcessor:
    def __init__(self, frames_collection: Collection, video_path: str):
        self.video_path = video_path
        self.frames_collection = frames_collection


    def serialize_frame(self, frame_array: np.ndarray) -> str:
        '''
        Convert a NumPy array to a base64 string so it can be saved in the collection
        '''
        serialized_frame = base64.b64encode(frame_array)
        return serialized_frame.decode()
    

    def deserialize_frame(self, serialized_frame: str):
        '''
        Convert a serialized frame to the NumPy array it represents
        '''
        serialized_frame_bytes = serialized_frame.encode()
        decoded_frame = base64.b64decode(serialized_frame_bytes)
        return decoded_frame

    
    def add_frames_to_collection(self, frame_interval: int = 30, batch_size: int = 64):
        '''
        Adds the video to the frame collection in batches
        '''
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        
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
                embedded_frames = self.frames_collection.embed(frames)

                # Serialize frames so they can be saved in the collection
                serialized_frames = [self.serialize_frame(frame) for frame in frames]

                # Add embedded frames to Chroma collection
                self.frames_collection.add(
                    ids=ids,
                    embeddings=embedded_frames,
                    documents=serialized_frames
                )

                print(f'Added batch {batch_end_frame // (frame_interval * batch_size)}')

            # End of video
            if not cap.isOpened():
                break

        cap.release()
        cv2.destroyAllWindows()
