# Book Search Demo with Azure AI Search

This demo showcases how to create a simple application that enables uploading text files (like books) and searching them using Azure AI Search with vector embeddings.

## Features

- Upload text files (.txt) to be indexed
- Generate vector embeddings for text chunks
- Create and manage search indexes in Azure AI Search
- Search using hybrid search (keyword + vector search)
- Simple web UI for uploading and searching

## Setup

1. Make sure you have Python 3.8+ installed

2. Clone this repository and navigate to the project directory

3. Create a virtual environment and activate it:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

5. Create a `.env` file with your Azure credentials:
   ```
   AIPROJECT_CONNECTION_STRING=your_ai_project_connection_string
   AISEARCH_INDEX_NAME=your_ai_search_index_name
   EMBEDDINGS_MODEL=text-embedding-ada-002  # or text-embedding-3-large
   ```

## Running the Application

Start the FastAPI server:

```
python app.py
```

The application will be available at http://localhost:8000

## Usage

1. Open your browser and go to http://localhost:8000
2. Use the Upload tab to upload .txt files to the search index
3. Switch to the Search tab to search across your indexed books
4. Select the desired index from the dropdown
5. Enter your search query and click "Search"

## Notes

- Processing large files may take some time, but the upload endpoint will return immediately as the processing happens in the background
- The application creates a unique index for each book file uploaded
- For best results, use high-quality text files with good formatting
# classics-server
