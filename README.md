# Hack Bajaj Project

This repository contains the codebase for the Hack Bajaj project, which is organized into multiple services:

## Folders

- **api_server/**: Node.js backend server.
- **ml_service/**: Python-based machine learning service.

## Setup

### API Server

1. Navigate to `api_server/`
2. Install dependencies:
   ```
   npm install
   ```
3. Create a `.env` file for environment variables.

### ML Service

1. Navigate to `ml_service/`
2. (Recommended) Create a virtual environment:
   ```
   python -m venv env
   ```
3. Activate the environment and install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file for environment variables.

## Notes

- Each service has its own `.gitignore` for best practices.
- Do not commit sensitive information (API keys, credentials, etc.).

## License

[Specify your license]