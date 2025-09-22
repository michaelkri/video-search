from vector_database import VectorDatabase
import gradio as gr


db = VectorDatabase()

db.add(
    ids=['1', '2', '3', '4'],
    uris=[
        '/workspaces/video-search/images/car.jpg',
        '/workspaces/video-search/images/house.png',
        '/workspaces/video-search/images/bear.jpg',
        '/workspaces/video-search/images/lake.jpg'
    ]
)

def search_db(term):
    results = db.query(
        query_texts=[f'{term}'],
        include=['uris', 'distances']
    )

    paths_with_distances = zip(results['uris'][0], map(str, results['distances'][0]))
    return paths_with_distances


with gr.Blocks(theme=gr.themes.Citrus()) as demo:
    gr.Markdown("# Video Search")
    with gr.Row():
        search_query_input = gr.Textbox(label="Search Query")
    search_button = gr.Button("Search")

    output_gallery = gr.Gallery(
        label="Results",
        object_fit="cover",
        columns=4,
        rows=2
    )

    search_button.click(
        fn=search_db,
        inputs=[search_query_input],
        outputs=output_gallery
    )

demo.launch()