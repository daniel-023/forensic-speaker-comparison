# Demo README

This folder is reserved for the Streamlit demo workflow.

## Current Status

The Streamlit entrypoint is in the project root: `app.py`.
`demo/` stores demo docs/instructions and optional demo-only support files.

## Recommended Layout

- `app.py`: Streamlit entrypoint (project root)
- `demo/requirements.txt`: demo-only dependencies (optional)
- `demo/samples/`: sample pairs for no-upload demo mode
- `demo/outputs/`: demo outputs/artifacts
- `demo/AGENTS.md`: demo-scoped agent instructions

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r demo/requirements.txt
```

## Run

```bash
streamlit run app.py
```

Run this command from the project root:
`/Users/daniel/Documents/VSCode/forensic-speaker-comparison`

In the app, select `Use sample pair` to run preloaded files from `demo/samples/`.

## Notes

- Keep demo-specific code and outputs inside `demo/`.
- Prefer writing generated files to `demo/outputs/`.
