
const express = require('express');
const axios = require('axios');
const router=express.Router();
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');

const PYTHON_API_URL = 'http://localhost:5000';


// Multer Configuration for File Uploads
const upload = multer({ dest: 'uploads/' });




router .post('/ingest-documents', upload.array('documents'), async (req, res) => {
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

module.exports = router;