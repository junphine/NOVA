import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.phi import PhiForCausalLM

torch.set_default_device("cuda")
ROOT_PATH = "/data/nlp/models/BAAI/nova-d48w1024-osp480/"
model = PhiForCausalLM.from_pretrained(ROOT_PATH+"text_encoder", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(ROOT_PATH+"tokenizer", trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)

inputs = tokenizer('''Write a detailed analogy between mathematics and a lighthouse.''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
