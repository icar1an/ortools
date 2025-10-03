### API key setup

You only need two keys:
- 1 Gemini key (`GEMINI_API_KEY`)
- 1 Google Maps key (`GOOGLE_MAPS_API_KEY`)

- Create a file named `.env` in this directory. Do not commit it.
- Add your keys (copy the snippet below):

```
GEMINI_API_KEY=your_gemini_key_here
GOOGLE_MAPS_API_KEY=your_maps_key_here
```

- Optional: install dotenv for local runs so the script auto-loads `.env`:

```
pip install python-dotenv
```

- The script will fail fast with a clear message if keys are missing when actually needed.

Notes
- Real keys should only live in environment variables or a local `.env` ignored by git (already configured in `.gitignore`).
- Never hardcode secrets in code or commit them to the repository.
