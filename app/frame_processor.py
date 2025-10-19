import cv2
from vector_database import Collection
import os


class FrameProcessor:
    def __init__(self, frames_collection: Collection, save_path: str = 'frames'):
        self.processed_videos = set()
        self.frames_collection = frames_collection
        self.save_path = save_path

    
    def add_frames_to_collection(self, video_path: str, frame_interval: int = 30, batch_size: int = 64) -> None:
        '''
        Adds the video to the frame collection in batches
        '''
        video_name = os.path.basename(video_path)

        # Already saved video frames
        if video_path in self.processed_videos:
            print(f'Video {video_path} already processed')
            return

        # Open video file
        cap = cv2.VideoCapture(video_path)

        # Create directory for frames
        os.makedirs(self.save_path, exist_ok=True)
        
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
                    break
                
                # Frame to be included in database
                if i % frame_interval == 0:
                    # Save frame to disk
                    frame_path = os.path.join(self.save_path, f'{video_name}_frame_{i}.jpg')
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
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        # Mark video as processed
        self.processed_videos.add(video_path)
        print(f'Added {video_path} to processed list')
