import chromadb
from vector_database import Collection
from frame_processor import FrameProcessor
import gradio as gr


def main():
    # Create a Chroma collection to hold frames
    client = chromadb.Client()
    frames_collection = Collection(client, 'frames')


    def search_frames(video_path, search_query):
        frames_processor = FrameProcessor(frames_collection, video_path=video_path)
        
        frames_processor.add_frames_to_collection()
        
        results = frames_collection.query(search_query=search_query)

        results_image_paths = [metadata['image_path'] for metadata in results['metadatas'][0]]

        return results_image_paths[:3]


    with gr.Blocks(theme=gr.themes.Citrus()) as demo:
        gr.Markdown("# Video Search")
        with gr.Row():
            search_query_input = gr.Textbox(label="Search Query")
        search_button = gr.Button("Search")
        video_input = gr.Video(label="Video")

        output_gallery = gr.Gallery(
            label="Results",
            object_fit="cover",
            columns=4,
            rows=2
        )

        search_button.click(
            fn=search_frames,
            inputs=[video_input, search_query_input],
            outputs=output_gallery
        )

    demo.launch()


if __name__ == "__main__":
    main()