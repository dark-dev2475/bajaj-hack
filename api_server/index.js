// api_server/index.js
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('Hello from the Node.js API Server! 🟩');
});

app.listen(port, () => {
  console.log(`Server listening at http://localhost:${port}`);
});
