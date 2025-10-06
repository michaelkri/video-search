import chromadb
import gradio as gr
import os
import shutil
from vector_database import Collection
from frame_processor import FrameProcessor


# Folder to store frames in
FRAMES_FOLDER = 'frames'


def main():
    # Create a Chroma collection to hold frames
    client = chromadb.Client()
    frames_collection = Collection(client, 'frames')

    # Create a directory to save video frames
    os.makedirs(FRAMES_FOLDER, exist_ok=True)

    # Used to process video frames
    frames_processor = FrameProcessor(frames_collection)
    
    def search_frames(video_path, search_query):
        
        frames_processor.add_frames_to_collection(video_path=video_path)
        
        results = frames_collection.query(search_query=search_query)

        results_image_paths = [metadata['image_path'] for metadata in results['metadatas'][0]]

        return results_image_paths[:3]

    try:
        with gr.Blocks(theme=gr.themes.Citrus()) as app:
            gr.Markdown("# Video Search")

            with gr.Row():
                with gr.Column(scale=1):
                    # Group all inputs in the first column
                    video_input = gr.Video(label="Upload Video")
                    text_input = gr.Textbox(label="Search Query")
                    search_button = gr.Button("Search", variant="primary")

                with gr.Column(scale=2):
                    # The second column is dedicated to the output
                    output_gallery = gr.Gallery(
                        label="Search Results",
                        object_fit="contain",
                        height="auto",
                        columns=4
                    )

            search_button.click(
                fn=search_frames,
                inputs=[video_input, text_input],
                outputs=output_gallery
            )

        app.launch()
    finally:
        # Delete directory containing frames
        shutil.rmtree(FRAMES_FOLDER)


if __name__ == "__main__":
    main()