import os
import logging
from flask import Flask, request, jsonify
import submission_handler
import asyncio
import search
import document_parser
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # ❗ Important to define
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Set 100MB max upload size

INDEX_NAME = "polisy-search"

UPLOAD_FOLDER = 'temp_docs'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- The Single Endpoint for the Submission ---
@app.route('/hackrx/run', methods=['POST'])
def run_rag():
    data = request.get_json()
    if not data or 'document_url' not in data or 'questions' not in data:
        return jsonify({"error": "Request must include 'document_url' and 'questions' list."}), 400

    doc_url = data['document_url']
    questions = data['questions']
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        return jsonify({"error": "'questions' must be a list of strings."}), 400

    logging.info(f"Received request for document: {doc_url} with {len(questions)} questions.")

    try:
        # Run the asynchronous handler from our synchronous Flask route
        answers = asyncio.run(submission_handler.handle_submission(doc_url, questions, app.config['UPLOAD_FOLDER']))
        response_data = {"answers": answers}
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"An error occurred during submission handling: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500




# --- NEW Endpoint to Ingest a Single File ---
@app.route('/ingest-file', methods=['POST'])
def ingest_file():
    print(f"Received content length: {request.content_length} bytes")
    if 'document' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['document']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Save file
        file.save(save_path)
        logging.info(f"File '{filename}' saved for processing.")

        # Step 1: Parse
        ingested_docs = document_parser.ingest_documents(specific_file=save_path)
        if not ingested_docs:
            raise ValueError("No documents parsed.")

        # Step 2: Chunk
        chunked_docs = document_parser.chunk_documents(ingested_docs)
        if not chunked_docs:
            raise ValueError("No chunks created.")

        # Step 3: Embeddings
        final_data = document_parser.generate_embeddings(chunked_docs)
        if not final_data or not isinstance(final_data, list):
            raise ValueError("Embedding generation failed or returned wrong format.")

        # Step 4: Pinecone Upsert
        vectors = [{
            "id": f"{filename}_chunk_{i}",
            "values": item["embedding"],
            "metadata": {"text": item["chunk_text"], "source": filename}
        } for i, item in enumerate(final_data)]

        if not vectors or any(type(v["values"]) is str for v in vectors):
            raise ValueError("Invalid vector format passed.")

        search.pc.Index(INDEX_NAME).upsert(vectors=vectors, batch_size=100)
        logging.info(f"Uploaded {len(vectors)} vectors.")

        return jsonify({
            "message": f"Successfully processed {filename}",
            "chunks_uploaded": len(vectors)
        }), 200

    except Exception as e:
        logging.error(f"[Ingest Error] {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Ensure file is always cleaned up
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    # Run the Flask server
    app.run(port=5000, debug=True)