# Hunyuan-MT Batch Translation on AWS

This repository provides scripts for **batch translation** using **Hunyuan-MT-7B** model from Tencent. It is optimized for **GPU-enabled AWS instances** (e.g., G6e.xlarge with L40S GPU).

- Model References:
  - [Hunyuan-MT Paper](https://www.arxiv.org/pdf/2509.05209)
  - [Preliminary Ranking of WMT25 MT Systems](https://www.linkedin.com/posts/kocmitom_preliminary-ranking-of-wmt25-general-machine-activity-7364948264888049664-5nAG?utm_source=share&utm_medium=member_desktop&rcm=ACoAADlBqKABh8F94bzPPJCIBmwIqJsD4FPP238)
  - [Hunyuan-MT-7B HuggingFace](https://huggingface.co/tencent/Hunyuan-MT-7B)
  - [Code Repository](https://github.com/Tencent-Hunyuan/Hunyuan-MT)

---

## 1️⃣ Create & Activate Virtual Environment
```bash
python3 -m venv hunyuan_venv
source hunyuan_venv/bin/activate
pip install --upgrade pip
```

## 2️⃣ Install Required Libraries & Download Models
```bash
# Install PyTorch GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Transformers and related libraries
pip install transformers==4.56.0
pip install accelerate bitsandbytes huggingface_hub[cli]

# Download Hunyuan-MT Models to your workspace
# Hunyuan-MT-7B
hf download tencent/Hunyuan-MT-7B --local-dir ~/hunyuan_model_workspace/model/Hunyuan-MT-7B

```

## 3️⃣ Verify Installation
```python
import torch, transformers
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

## 4️⃣ Usage
### Python Example
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./Hunyuan-MT-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

messages = [
    {"role": "user", "content": "Translate the following segment into Chinese, without additional explanation.\n\nIt’s on the house."}
]

tokenized_chat = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt"
)

outputs = model.generate(
    tokenized_chat.to(model.device),
    max_new_tokens=2048,
    top_k=20,
    top_p=0.6,
    repetition_penalty=1.05,
    temperature=0.7
)

output_text = tokenizer.decode(outputs[0])
print(output_text)
```

### Batch Translation Example
```bash
python3 batch_translate_for-server-use_with-loading-time.py \
  --input /path/to/input.txt \
  --output /path/to/output.txt \
  --source Chinese \
  --target English
```

## 5️⃣ AWS Server Setup
```bash
# Connect to server
ssh -i "OT-Access-Key.pem" ubuntu@<AWS-Public-IP>

# Create workspace
mkdir -p ~/hunyuan_model_workspace/model
mkdir -p ~/hunyuan_model_workspace/script
chmod -R 755 ~/hunyuan_model_workspace

# Download models (requires huggingface_hub)
hf download tencent/Hunyuan-MT-7B --local-dir ~/hunyuan_model_workspace/model/Hunyuan-MT-7B
hf download tencent/Hunyuan-MT-Chimera-7B --local-dir ~/hunyuan_model_workspace/model/Hunyuan-MT-Chimera-7B
```

## 6️⃣ Performance on AWS G6e.xlarge
| Scenario | Model Loading Time (seconds) | Translation Time per Sentence (seconds) | Notes |
|----------|------------------------|----------------------------------|-------|
| Cold Start | 9.41 | 22.84 | First CUDA + kernel compilation |
| Warm Model | 5.78 (avg) | 1.54 (avg) | Subsequent inferences on GPU |
| Production | ~9.4 (load once) | ~1.5 | Keep model loaded in memory for real-time translation |

> Recommendation: Use a persistent server (FastAPI/Flask/Gradio) to avoid repeated cold starts.

