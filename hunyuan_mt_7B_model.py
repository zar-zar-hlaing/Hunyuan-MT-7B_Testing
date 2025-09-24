from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path to the local model; Assign with your downloaded model path;
MODEL_PATH = "/home/ubuntu/hunyuan_model_workspace/model/Hunyuan-MT-7B"


# Load tokenizer and model just once
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)

model.eval()
