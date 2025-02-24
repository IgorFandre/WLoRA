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
