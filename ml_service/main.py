import os
import logging
from flask import Flask, request, jsonify
import submission_handler
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

UPLOAD_FOLDER = 'temp_docs'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- The Single Endpoint for the Submission ---
@app.route('/hackrx/run', methods=['POST'])
def hackrx_run():
    """
    Handles the entire RAG pipeline by running our async handler.
    """
    data = request.get_json()
    if not data or 'documents' not in data or 'questions' not in data:
        return jsonify({"error": "Request must include 'documents' URL and 'questions' list."}), 400

    doc_url = data['documents']
    questions = data['questions']
    
    logging.info(f"Received request for document: {doc_url} with {len(questions)} questions.")

    try:
        # Run the asynchronous handler from our synchronous Flask route
        answers = asyncio.run(submission_handler.handle_submission(doc_url, questions, app.config['UPLOAD_FOLDER']))
        
        response_data = {"answers": answers}
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"An error occurred during submission handling: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == "__main__":
    app.run(port=8000, debug=True)


# --- Endpoint to Process a Query (from Day 8) ---
# @app.route('/process_query', methods=['POST'])
# def process_query():
#     INDEX_NAME = "polisy-search"
#     data = request.get_json()
#     if not data or 'raw_query' not in data:
#         return jsonify({"error": "Invalid request: 'raw_query' is required."}), 400
    
#     raw_query = data['raw_query']
#     logging.info(f"Received query for processing: \"{raw_query}\"")

#     try:
#         structured_query = query_parser.get_structured_query(raw_query)
#         if not structured_query:
#             return jsonify({"error": "Failed to parse query."}), 500

#         search_results = search.perform_search(
#             raw_query=raw_query,
#             structured_query=structured_query,
#             index_name=INDEX_NAME
#         )
#         if not search_results:
#             return jsonify({"error": "No relevant documents found."}), 404

#         final_answer = answer_generator.generate_answer(
#             raw_query=raw_query, 
#             search_results=search_results,
#             query_language="en" # Assuming English for now
#         )
#         if not final_answer:
#             return jsonify({"error": "Failed to generate a final answer."}), 500
        
#         return jsonify(final_answer.model_dump())

#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         return jsonify({"error": "An internal server error occurred."}), 500

# # --- NEW Endpoint to Ingest a Single File ---
# @app.route('/ingest-file', methods=['POST'])
# def ingest_file():
#     INDEX_NAME = "polisy-search"
#     if 'document' not in request.files:
#         return jsonify({"error": "No file part in the request"}), 400
    
#     file = request.files['document']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file:
#         filename = secure_filename(file.filename)
#         # Save the file temporarily to a dedicated folder
#         save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(save_path)
#         logging.info(f"File '{filename}' saved temporarily for processing.")

#         try:
#             # --- Run the Ingestion Pipeline for this single file ---
#             logging.info(f"1. Ingesting '{filename}'...")
#             # We modify the ingest_documents function to take a specific path
#             ingested_docs = document_parser.ingest_documents(specific_file=save_path)
            
#             logging.info("2. Chunking document...")
#             chunked_docs = document_parser.chunk_documents(ingested_docs)
            
#             logging.info("3. Generating embeddings...")
#             final_data = document_parser.generate_embeddings(chunked_docs)
            
#             logging.info(f"4. Uploading {len(final_data)} vectors to Pinecone...")
#             search.pc.Index(INDEX_NAME).upsert(
#                 vectors=[{
#                     "id": f"{filename}_chunk_{i}", # Create a unique ID
#                     "values": item["embedding"],
#                     "metadata": {"text": item["chunk_text"], "source": filename}
#                 } for i, item in enumerate(final_data)],
#                 batch_size=100
#             )
            
#             # Clean up the temporary file
#             os.remove(save_path)
            
#             return jsonify({"message": f"Successfully ingested and processed {filename}"}), 200

#         except Exception as e:
#             logging.error(f"Failed to process file {filename}: {e}")
#             # Clean up even if there's an error
#             if os.path.exists(save_path):
#                 os.remove(save_path)
#             return jsonify({"error": f"Failed to process file {filename}"}), 500
    
#     return jsonify({"error": "An unknown error occurred"}), 500


# if __name__ == "__main__":
#     # Create uploads directory if it doesn't exist
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     # Run the Flask server
#     app.run(port=5000, debug=True)
