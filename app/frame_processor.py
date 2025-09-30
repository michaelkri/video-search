import cv2
import numpy as np
from vector_database import Collection
import os


class FrameProcessor:
    # Directory to save frames to
    FRAMES_DIR = 'frames'


    def __init__(self, frames_collection: Collection, video_path: str, save_path: str = 'frames'):
        self.video_path = video_path
        self.frames_collection = frames_collection
        self.save_path = save_path


    def __del__(self):
        os.rmdir(self.save_path)

    
    def add_frames_to_collection(self, frame_interval: int = 30, batch_size: int = 64):
        '''
        Adds the video to the frame collection in batches
        '''
        # Open video file
        cap = cv2.VideoCapture(self.video_path)

        # Create directory for frames
        os.makedirs(FrameProcessor.FRAMES_DIR, exist_ok=True)
        
        # Store embeddings in batches
        i = 0
        while True:
            batch_end_frame = i + frame_interval * batch_size
            
            ids_batch = []
            frames_batch = []
            metadatas_batch = []

            while i < batch_end_frame and cap.isOpened():
                ret, frame = cap.read()
                
                # Failed to read frame
                if not ret:
                    return
                
                # Frame to be included in database
                if i % frame_interval == 0:
                    # Save frame to disk
                    frame_path = os.path.join(FrameProcessor.FRAMES_DIR, f'frame_{i}.jpg')
                    cv2.imwrite(frame_path, frame)

                    # Set Chroma entry information
                    ids_batch.append(str(i))
                    frames_batch.append(frame)
                    metadatas_batch.append({
                        "image_path": frame_path
                    })

                i += 1

            if ids_batch:
                # Embed batch frames
                embedded_frames_batch = self.frames_collection.embed(frames_batch)
                
                # Add embedded frames to Chroma collection
                self.frames_collection.add(
                    ids=ids_batch,
                    embeddings=embedded_frames_batch,
                    metadatas=metadatas_batch
                )

                print(f'Added batch {batch_end_frame // (frame_interval * batch_size)}')

            # End of video
            if not cap.isOpened():
                break

        cap.release()
        cv2.destroyAllWindows()
