# Gemini Image Generator

Streamlit app that calls the latest Gemini image-generation endpoint (`models/gemini-2.0-flash-exp`) via the Google GenAI SDK. Enter your Gemini API key and a prompt to create images on demand.

## Setup

1. Create a virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the app

Launch Streamlit and open the provided local URL in your browser:

```bash
streamlit run app.py
```

Enter your Gemini API key and prompt. The app will display the generated image (and any text notes). API keys are only used for your local session and are not stored.
