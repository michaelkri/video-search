from vector_database import VectorDatabase
import gradio as gr


def main():
    db = VectorDatabase()

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
            fn=db.search,
            inputs=[video_input, search_query_input],
            outputs=output_gallery
        )

    demo.launch()


if __name__ == "__main__":
    main()