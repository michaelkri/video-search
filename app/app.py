import chromadb
import gradio as gr
import os
import shutil
import uuid
from yt_dlp import YoutubeDL
from vector_database import Collection
from frame_processor import FrameProcessor


FRAMES_FOLDER = 'frames'
VIDEOS_FOLDER = 'videos'


def main():
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    os.makedirs(VIDEOS_FOLDER, exist_ok=True)

    # Create a Chroma collection to hold frames
    client = chromadb.Client()
    frames_collection = Collection(client, 'frames')
    frames_processor = FrameProcessor(frames_collection, save_path=FRAMES_FOLDER)


    def search_frames(video_path, search_query):
        frames_processor.add_frames_to_collection(video_path=video_path)
        results = frames_collection.query(search_query=search_query)
        results_image_paths = [metadata['image_path'] for metadata in results['metadatas'][0]]
        return results_image_paths[:3]


    def process_and_search(upload, url, query):
        if upload:
            # Copy uploaded file
            video_path = os.path.join(VIDEOS_FOLDER, os.path.basename(upload))
            shutil.copy(upload, video_path)
        else:
            # Download video from YouTube
            ydl_options = {
                'outtmpl': os.path.join(VIDEOS_FOLDER, '%(id)s.%(ext)s'),
                'noplaylist': True
            }
            video_url = url.strip()
            with YoutubeDL(ydl_options) as ydl:
                info = ydl.extract_info(video_url, download=True)
                video_filename = f"{info['id']}.{info['ext']}"
                video_path = os.path.join(VIDEOS_FOLDER, video_filename)

        return search_frames(video_path, query)
        

    try:
        with gr.Blocks(theme=gr.themes.Citrus()) as app:
            gr.Markdown("# Video Search")

            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Upload Video"):
                            video_upload_input = gr.Video(label="Upload your video file")
                        with gr.Tab("Video URL"):
                            video_url_input = gr.Textbox(label="Enter YouTube video URL", placeholder="https://www.youtube.com/watch?v=...")

                    text_input = gr.Textbox(label="Search Query")
                    search_button = gr.Button("Search", variant="primary")

                with gr.Column(scale=2):
                    output_gallery = gr.Gallery(
                        label="Search Results",
                        object_fit="contain",
                        height="auto",
                        columns=4
                    )

            search_button.click(
                fn=process_and_search,
                inputs=[video_upload_input, video_url_input, text_input],
                outputs=output_gallery
            )

        app.launch()
    finally:
        # Delete directory containing frames
        shutil.rmtree(FRAMES_FOLDER)


if __name__ == "__main__":
    main()