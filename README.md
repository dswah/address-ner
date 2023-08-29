---
license: apache-2.0
datasets:
- oscar-corpus/OSCAR-2301
language:
- de
metrics:
- f1
library_name: transformers
pipeline_tag: token-classification
---
# address-ner

## Address NER for Legal German Text

# Installation
`pip install .`
  or
`pip install -e .` for an editable installation

You will need an globally accessible `DVC` installation.  
You can install DVC as a binary with `pipx` by doing:
```bash
pip install pipx
pipx install dvc[s3]
```

# Running the project
This whole project is versioned as a DVC pipeline: from gathering external data, cleaning it, and training a model.  
You can reproduce the entire pipeline by doing  
`dvc repro`

# Running Specific Stages of the Pipeline
You can check the pipeline stages by doing
`dvc dag`

or by looking inside `dvc.yaml`.
