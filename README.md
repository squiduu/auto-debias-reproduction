# Auto-Debias
This is a reproduction source code of [Auto-Debias](https://github.com/Irenehere/Auto-Debias) containing SEAT and GLUE.
## Requirements
```
python==3.7.10
numpy==1.21.5
torch==1.11.0
transformers==4.14.1
datasets==2.3.2
tqdm==4.64.0
dataclasses==0.8
scipy==1.7.3
scikit-learn==1.0.2
```
## Setup
This environment is available in Ubuntu 20.04 LTS, and it is not tested on other OS.
```
git clone https://github.com/squiduu/auto-debias-reproduction.git
cd auto-debias-reproduction/
conda create -n auto_debias python==3.7.10
conda activate auto_debias
```
Install all of the above requirements after setup.
## Debiasing
### Generating biased prompts
```
cd auto_debias/
bash generate_prompts.sh
```
Then you wil get the generated prompts file in `./data/debias/prompts_{model_name_or_path}_{bias_type}`
### Debiasing models
```
bash auto_debias.sh
```
Then you will get the debiased checkpoint in `./auto_debias/out/{model_name}_{run_name}_{bias_type}_epoch:{epoch}`
## Evaluation
### SEAT
```
cd ../seat/
bash run_seat.sh
```
### GLUE
```
cd ../glue/
bash run_glue.sh
```
