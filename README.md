### ``Wandb`` project: 
https://wandb.ai/shkodnik-mipt/SBER_LORA?nw=nwusershkodnik

# PEFT for Sber

## Requirements 
    python 3.11.7

## Dependancies installation
Plese follow these steps (required!):

0. If you don't have python of specific version:
    - Install conda
    - Run:
      ```bash
      conda create --name python3.11.7 python=3.11.7
      conda activate python3.11.7
      ```

1. Create virtual env
```bash
python3.11 -m venv venv
source venv/bin/activate
```

2. Update pip to latest
```bash
pip install pip==24.1.1
```

3. Install rest of packages from requirements.txt
```bash
pip install -r requirements.txt
```

## Usage
### Train Scripts
Up to this point, experiments were based on Meta-Llama-3.1-8B from unsloth library (see https://huggingface.co/unsloth/Meta-Llama-3.1-8B)

1. GLUE + deberta-v3-base
```bash
./scripts/run_glue_deberta.sh
```
