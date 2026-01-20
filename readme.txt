anaconda terminal from file repository:

conda create -n ner-ui python=3.11 -y
conda activate ner-ui
pip install -r requirements.txt

# required because you call spacy.load("en_core_web_sm")
python -m spacy download en_core_web_sm

# (optional) generate data if not already present
python export_bc5cdr_to_json.py
python make_spacy_docbins_from_frozen.py

# train full model
python -m spacy train configs/config.cfg --output output --paths.train data/train.spacy --paths.dev data/dev.spacy
python -m spacy evaluate output/model-best data/test.spacy -o results/metrics_full.json

# learning curve subsets (examples)
python make_subset_spacy.py --size 50  --seed 0
python -m spacy train configs/config.cfg --output curve/runs/run_50_seed0 --paths.train curve/subsets/train_50_seed0.spacy --paths.dev data/dev.spacy
python -m spacy evaluate curve/runs/run_50_seed0/model-best data/test.spacy -o results/metrics_b50_seed0.json

# active learning selection
python scripts/al_one_step_800.py
python -m spacy train configs/config.cfg --output al/run_al_800 --paths.train al/train_al_800.spacy --paths.dev data/dev.spacy
python -m spacy evaluate al/run_al_800/model-best data/test.spacy -o al/metrics_al_800.json






#UI


pip install streamlit-component-lib


cd ner_annotator/frontend

npm init -y

npm install --save react react-dom streamlit-component-lib

npm install --save-dev typescript parcel @types/react @types/react-dom


streamlit run app.py


