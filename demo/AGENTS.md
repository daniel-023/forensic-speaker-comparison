# AGENTS.md (Demo Scope)

## Scope

These instructions apply only to work inside `demo/`.
Do not treat this file as repo-wide policy.

## Purpose

`demo/` is reserved for the Streamlit demo workflow of this project.
The Streamlit app entrypoint is `../app.py` at the project root.
Work here should prioritize:
- reliable demo behavior,
- simple reproducible setup,
- clear user-facing outputs.

## Change Boundaries

- Allowed:
  - create/update files under `demo/` for the Streamlit demo.
  - update `../app.py` when the user explicitly requests demo app changes.
  - add small helper modules under `demo/`.
  - add demo-specific docs under `demo/README.md`.
- Not allowed without explicit user request:
  - modify files outside `demo/`.
  - move or rename dataset, training, inference, or evaluation scripts in the project root.
  - introduce large refactors across the repository.

## Streamlit Requirements

- Keep startup simple (single command where possible).
- Avoid long blocking operations on each interaction.
- Cache heavyweight model loading where appropriate.
- Handle upload/runtime errors with clear messages.
- Preserve deterministic behavior when randomization is used (fixed seed if applicable).

## Dependencies

- Reuse existing project dependencies when possible.
- If demo-only dependencies are needed, prefer a `demo/requirements.txt`.
- Do not silently upgrade core ML packages unless requested.

## I/O and Artifacts

- Keep demo outputs inside `demo/outputs/` unless user requests otherwise.
- Avoid writing temporary artifacts to project root.
- If large/generated files are produced, mention whether they should be gitignored.

## Validation Before Completion

Before marking work complete, run lightweight checks:
- syntax check for modified Python files,
- smoke run command for the demo (if app exists),
- confirm output paths are correct.

If runtime dependencies are missing, report exactly what is missing and the command to install.

## Communication

- State assumptions briefly.
- For behavior changes, explain:
  - what changed,
  - why,
  - how to run it.
- If requested task conflicts with these rules, follow user request and call out the override explicitly.
