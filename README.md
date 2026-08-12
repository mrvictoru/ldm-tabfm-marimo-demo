# TabFM LDM demo

This workspace contains a marimo notebook that recreates the IBM LDM-style fraud triage workflow with:

- the public Hugging Face dataset `CiferAI/Cifer-Fraud-Detection-Dataset-AF`
- Google's pretrained `google/tabfm-1.0.0-pytorch` classification model
- reactive marimo controls for what-if transaction editing
- similar-case retrieval over sampled historical transactions

## Run locally

```powershell
pip install -r requirements.txt
marimo edit ldm_tabfm_marimo_demo.py
```

## Molab notes

Open `ldm_tabfm_marimo_demo.py` in molab or marimo, install the dependencies from `requirements.txt`, and run the notebook. The first run downloads the TabFM weights and samples a notebook-sized slice of the Cifer dataset from Hugging Face.

## Presentation assets

Open `index.html` in a browser to view the HTML intro slide. The speaker notes are in `speaker_notes.md`.
