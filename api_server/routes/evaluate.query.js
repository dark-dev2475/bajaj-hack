// route for processing evaluation queries
const express = require('express');
const router = express.Router();  


router.post('/evaluate', async (req, res) => {
    console.log("Received request on /evaluate");
    const { query } = req.body;
    if (!query) {
        return res.status(400).json({ error: 'Request body must contain a "query" field.' });
    }

    try {
        console.log(`Forwarding query to Python service: "${query}"`);
        const pythonResponse = await axios.post(`${PYTHON_API_URL}/process_query`, {
            raw_query: query
        });
        res.json(pythonResponse.data);
    } catch (error) {
        console.error("Error calling Python service:", error.message);
        const status = error.response ? error.response.status : 500;
        const data = error.response ? error.response.data : { error: 'Internal server error.' };
        res.status(status).json(data);
    }
});

// Export the router
module.exports = router;

