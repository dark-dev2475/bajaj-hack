import logging
from flask import Flask, request, jsonify
import query_parser
import search
import answer_generator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/process_query', methods=['POST'])
def process_query():
    """
    This is the main endpoint for the Python ML service.
    It receives a query, runs the full RAG pipeline, and returns the answer.
    """
    INDEX_NAME = "polisy-search"
    
    # Receive JSON with the query from the Node.js server [cite: user-provided source]
    data = request.get_json()
    if not data or 'raw_query' not in data:
        return jsonify({"error": "Invalid request: 'raw_query' is required."}), 400
    
    raw_query = data['raw_query']
    logging.info(f"Received query for processing: \"{raw_query}\"")

    try:
        # Day 4: Parse the query
        structured_query = query_parser.get_structured_query(raw_query)
        if not structured_query:
            return jsonify({"error": "Failed to parse query."}), 500

        # Day 5: Retrieve relevant documents
        search_results = search.perform_search(
            raw_query=raw_query,
            structured_query=structured_query,
            index_name=INDEX_NAME
        )
        if not search_results:
            return jsonify({"error": "No relevant documents found."}), 404

        # Day 6: Generate final answer
        final_answer = answer_generator.generate_answer(
            raw_query=raw_query, 
            search_results=search_results,
            query_language="en" # Assuming English for now, can be enhanced
        )
        if not final_answer:
            return jsonify({"error": "Failed to generate a final answer."}), 500
        
        # Return the final JSON response [cite: user-provided source]
        return jsonify(final_answer.model_dump())

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    # Run the Flask server on port 5000
    app.run(port=5000, debug=True)