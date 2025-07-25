const express = require('express');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;
const PYTHON_API_URL = 'http://localhost:5000';

// Multer Configuration for File Uploads
const upload = multer({ dest: 'uploads/' });

app.use(express.json());

// --- Endpoint to Evaluate a Query ---
app.post('/evaluate', async (req, res) => {
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

// --- Endpoint to Ingest Multiple Documents ---
app.post('/ingest-documents', upload.array('documents'), async (req, res) => {
    console.log(`Received ${req.files.length} file(s) for ingestion.`);
    
    if (!req.files || req.files.length === 0) {
        return res.status(400).json({ error: 'No files were uploaded.' });
    }

    const results = [];
    for (const file of req.files) {
        try {
            const formData = new FormData();
            formData.append('document', fs.createReadStream(file.path), file.originalname);

            console.log(`Sending file ${file.originalname} to Python service...`);
            
            // Note the updated URL to match our new blueprint structure
            const response = await axios.post(`${PYTHON_API_URL}/ingest-file`, formData, {
                headers: { ...formData.getHeaders() }
            });

            results.push({ file: file.originalname, status: 'Success', message: response.data.message });
        } catch (error) {
            console.error(`Failed to ingest ${file.originalname}:`, error.message);
            results.push({ file: file.originalname, status: 'Error', message: error.message });
        } finally {
            fs.unlinkSync(file.path);
        }
    }

    res.status(200).json({
        message: 'Ingestion process completed.',
        results: results
    });
});


app.listen(PORT, () => {
    console.log(`Node.js API server listening on port ${PORT}`);
    if (!fs.existsSync('uploads')){
        fs.mkdirSync('uploads');
    }
});