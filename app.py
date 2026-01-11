from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app = FastAPI()

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "Sambit-Mishra/vekkam-v0"

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base, ADAPTER)
model.eval()

class Req(BaseModel):
    prompt: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: Req):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=200)
    return {"text": tokenizer.decode(out[0], skip_special_tokens=True)}