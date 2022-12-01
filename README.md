# Phrase Extraction

## Purpose
This repository contains a part of Elsa Fernandas implementation for the Bachelor's Thesis *Phrase Extraction for In-Depth Comparison of Regulatory Documents and Their Realizations*.


## Setup
This project uses Python 3.9 and [spaCy](https://spacy.io/usage) v3.4, but it should be compatible with similar versions as well. It is recommended to use a virtual environment such as Anaconda. To download and install spaCy and the required pretrained transformer model, - run the following:
```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_trf
```
- create the folders "input", "result" and "models"

- for the Input: put each sentence into a .txt file (one sentence per file) in the `input/realization_document/` and `input/regulatory_document/` directories. You can use the notebook `transform_gs_to_single_txt.ipynb` to transform the excel from step 1 of the other script into this format

- run create_model.py to create the model

- run the constraint component extraction with the path to the above saved model:
```
python extract_docs.py <model_path>
```

- copy output from results back to other script
