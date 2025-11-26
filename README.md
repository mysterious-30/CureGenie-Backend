# Backend API Service Setup

This Python backend service handles barcode reading from Student ID cards using the `read_barcode` function from `Database.py`.

## Prerequisites

```bash
pip install fastapi uvicorn pillow numpy pyzbar supabase
```

## Running the Service

1. Make sure your Supabase credentials are set in `Database.py`
2. Start the FastAPI service:

**For local access only:**
```bash
# From the Backend directory
uvicorn app:app --reload --port 8000
```

**For network access (to use from phone):**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Or use Python directly:

```bash
python app.py
```

The service will be available at:
- Local: `http://localhost:8000`
- Network: `http://YOUR_LAPTOP_IP:8000` (when using `--host 0.0.0.0`)

## Environment Variables

In your Next.js project, set the Python backend URL:

```env
# .env.local
PYTHON_BACKEND_URL=http://localhost:8000
```

For production, update this to your deployed Python service URL.

## API Endpoints

### POST `/api/read-barcode`

Reads barcode from a base64-encoded image.

**Request:**
```json
{
  "image": "base64_encoded_image_string",
  "format": "image/jpeg"
}
```

**Response:**
```json
{
  "barcode": "STUDENT-12345",
  "success": true,
  "message": "Barcode verified successfully"
}
```

### GET `/health`

Health check endpoint.

## Integration with Next.js

The Next.js frontend (`/api/read-barcode`) will automatically forward requests to this Python service when the `PYTHON_BACKEND_URL` environment variable is set.

