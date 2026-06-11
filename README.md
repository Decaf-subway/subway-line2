# DeCafe Split Apps

This branch separates the project into two independent Streamlit apps.

## Directories

- `line2/`: Seoul subway line 2 app, models, data, and Docker Space files.
- `line1_8/`: Seoul subway line 1-8 app, models, data, and Docker Space files.

Each directory is intended to be uploaded as a separate Hugging Face Docker Space repository.

## Local Run

```powershell
cd line2
streamlit run subway_app.py
```

```powershell
cd line1_8
streamlit run subway_app.py
```

## Hugging Face

Upload the contents of each directory as the root of its own Space.
Both directories include a `Dockerfile` and `README.md` with `sdk: docker` and `app_port: 7860`.
