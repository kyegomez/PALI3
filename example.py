import torch
from pali3.ul2 import Pali3
from pali3.tokenizer import tokenize


text = tokenize("Hello, world!").astype("int16")
print(f"text tokens {text}")

text = torch.from_numpy(text)
print(f"Text tensors: {text.shape}")

model = Pali3()

img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 256, (1, 1024))
mask = torch.ones(1, 1024).bool()
output_text = torch.randint(0, 256, (1, 1024))

result = model.process(img, text, output_text, mask)
print(result)
