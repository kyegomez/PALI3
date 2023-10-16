import torch
from pali3.main import Pali3

model = Pali3()

img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 256, (1, 1024))
mask = torch.ones(1, 1024).bool()
output_text = torch.randint(0, 256, (1, 1024))

result = model.process(img, prompt, output_text, mask)
print(result)
