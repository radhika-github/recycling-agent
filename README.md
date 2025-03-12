# ♻️ recycling-data

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd recycling-data
```

### 2. Create a Virtual Environment and Install Dependencies
It's recommended to use a Python virtual environment to manage dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install Additional Packages (If Needed)
If you encounter the following error:
```
ModuleNotFoundError: No module named 'llama_index.experimental'
```
Install the missing packages manually:
```bash
pip install llama-index-experimental
pip install llama-index-embeddings-huggingface
```

### 4. Hugging Face Login
Before running the project, log in to Hugging Face to access the required models:
```bash
huggingface-cli login
```

## ▶️ Running the Project
Once everything is set up, run the main script:
```bash
python3 main.py
```