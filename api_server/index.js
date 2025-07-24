// api_server/index.js
const express = require('express');
const axios = require('axios');

const app = express();
const PORT = 3000;
const PYTHON_API_URL = 'http://localhost:5000/process_query';

// Middleware to parse JSON bodies
app.use(express.json());

// Define the main REST endpoint: POST /evaluate [cite: user-provided source]
app.post('/evaluate', async (req, res) => {
    console.log("Received request on /evaluate");

    // 1. Receive JSON with query [cite: user-provided source]
    const { query } = req.body;
    if (!query) {
        return res.status(400).json({ error: 'Invalid request: "query" is required.' });
    }

    try {
        // 2. Call the Python microservice [cite: user-provided source]
        console.log(`Forwarding query to Python service: "${query}"`);
        const pythonResponse = await axios.post(PYTHON_API_URL, {
            raw_query: query
        });

        // 3. Return the JSON response from the Python service [cite: user-provided source]
        console.log("Successfully received response from Python service.");
        res.json(pythonResponse.data);

    } catch (error) {
        console.error("Error calling Python service:", error.message);
        // Forward the error status and message if available
        if (error.response) {
            res.status(error.response.status).json(error.response.data);
        } else {
            res.status(500).json({ error: 'An internal server error occurred.' });
        }
    }
});
app.listen(PORT, () => {
    console.log(`Node.js API server listening on port ${PORT}`);
});