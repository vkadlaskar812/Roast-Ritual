# Roast & Ritual

An AI-powered cafe ordering app built with FastAPI and the Gemini API. The active chat experience is in `app_agent.py`; SQLite is used locally, and Cloud SQL can be configured for persistent production data.

Developed for Google's Builder Pop-Up Vibecode event.

## Run locally

1. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to a key from Google AI Studio.
2. Create and activate a virtual environment, install dependencies, and start the chat app:

   ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   python -m uvicorn app_agent:app --reload
   ```

3. Open `http://127.0.0.1:8000`.

Do not commit `.env`; it contains secrets. The local SQLite database is created automatically.

## Deploy to Cloud Run

Create a Secret Manager secret named `gemini-api-key` and add the Gemini API key as an enabled version. Grant the Cloud Run runtime service account the `roles/secretmanager.secretAccessor` role on that secret. From this project directory, deploy with:

```bash
gcloud run deploy roast-ritual \
  --source . \
  --region asia-south1 \
  --project=PROJECT_ID \
  --allow-unauthenticated \
  --set-secrets=GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars=GEMINI_MODEL=gemini-3.8-flash
```

Replace `PROJECT_ID` with your Google Cloud project ID. This service is public, so monitor Gemini usage and quota. SQLite in a Cloud Run container is not persistent across restarts or instances; use Cloud SQL when persistent shared order history is needed.
