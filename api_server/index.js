const express = require('express');
const axios = require('axios');
const multer = require('multer');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const evaluateQuery = require('./routes/evaluate.query'); // Assuming this is the path to your evaluate query route
const ingestDoc =require('./routes/ingestion.route')
const app = express();
const PORT = 3000;



app.use(express.json());

// --- Endpoint to Evaluate a Query ---
app.use('/api', evaluateQuery);

// --- Endpoint to Ingest Multiple Documents ---
app.use('/api',ingestDoc);


app.listen(PORT, () => {
    console.log(`Node.js API server listening on port ${PORT}`);
    if (!fs.existsSync('uploads')){
        fs.mkdirSync('uploads');
    }
});