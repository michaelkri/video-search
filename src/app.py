from vector_database import VectorDatabase
import gradio as gr


db = VectorDatabase()

db.add(
    ids=['1', '2'],
    uris=[
        '/workspaces/video-search/images/car.jpg',
        '/workspaces/video-search/images/house.png'
    ]
)

def search_db(term):
    results = db.query(
        query_texts=[f'{term}'],
        include=['uris', 'distances']
    )
    return results


with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    gr.Markdown("# Video Search")
    with gr.Row():
        search_query_input = gr.Textbox(label="Search Query")
    search_button = gr.Button("Search")

    output_json = gr.JSON(label="Results")

    search_button.click(
        fn=search_db,
        inputs=[search_query_input],
        outputs=output_json
    )

demo.launch()