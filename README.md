# Semantic Video Search App
### Search within video content using natural language

<!-- Demo GIF -->

## Architecture
<!-- Diagram -->
<img src="assets/diagram.svg" />

### Storage
1. A video can be uploaded in one of two ways:
    - YouTube video link: The video is downloaded and saved.
    - Direct file upload.
2. The frames of the video are extracted using OpenCV and stored as individual images.
3. An embedding is created out of each frame using the OpenCLIP model.
4. Each embedding is saved in a ChromaDB collection (vector database equivalent to a table), along with the corresponding frame as metadata.

### Search
1. A natural language search query is received as input.
2. An embedding for the query is created using OpenCLIP.
3. The embedding is used to query the ChromaDB collection of frame embeddings, to find the three most similar stored embeddings.
4. The frames corresponding to the embeddings found in the previous step are fetched and displayed.

Simple caching is implemented, so subsequent searches within a video previously uploaded in this session are faster (as the **Storage** part is skipped).


## Tech Stack
- **ML:** OpenCLIP
- **Database:** ChromaDB
- **Frontend:** Gradio
- **YouTube Scraping:** yt-dlp


## Local Setup
1. Clone the repository:
```sh
git clone https://github.com/michaelkri/video-search.git
```

2. Build and run the Docker container:
```sh
docker build -t video-search .
docker run -p 7860:7860 video-search
```