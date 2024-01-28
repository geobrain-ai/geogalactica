import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_path = "/home/lzhou/download/0617/qomolangma-30b-sft-geosignal-main"

input_text = "How did Earth and other planets form? Were planets formed in situ?"

my_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

print(tokenizer.decode(my_model.generate(tokenizer(input_text, return_tensors="pt")["input_ids"], max_length=128, do_sample=True, top_p=0.95)[0].tolist()))
