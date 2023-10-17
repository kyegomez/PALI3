[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Pali3
![pali](pali.png)

"Figure 1: Overview of the PaLI-3 (5B) model: images are encoded into visual tokens individually
by the contrastively pretrained 2B SigLIP vision model. Along with a query, these visual tokens
are passed to an 3B encoder-decoder UL2 Transformer which produces the desired answer."


Vit trained with siglip loss -> embeddings -> ul2 -> text tokens

text -> tokenizer -> embeddings -> ul2 -> text tokens

[ARXVIV PAPER LINK](https://arxiv.org/pdf/2310.09199v1.pdf)

--------

## Installation

`pip install pali3`

-------

## Usage:

```python
import torch
from pali3.main import Pali3

model = Pali3()

img = torch.randn(1, 3, 256, 256)
prompt = torch.randint(0, 256, (1, 1024))
mask = torch.ones(1, 1024).bool()
output_text = torch.randint(0, 256, (1, 1024))

result = model.process(img, prompt, output_text, mask)
print(result)


```


-------

## Architecture

Here is the ASCII representation of the model architecture and the stages of training:

```
Model Architecture:

Image Input
    |
    V
Contrastive Vision Encoder (ViT-G/14)
    |
    V
Transformer Encoder
    |
    V
Transformer Decoder
    |
    V
Text Output

Stages of Training:

Stage 0: Unimodal pretraining
    |
    V
Stage 1: Multimodal training
    |
    V
Stage 2: Resolution increase
    |
    V
Task specialization (transfer)

```


# Model Training Phases
The model architecture consists of a contrastive vision encoder (ViT-G/14) that encodes the image into tokens. These tokens are passed to a transformer encoder and then to a transformer decoder that generates a text output.

The training procedure consists of multiple stages:

-   Stage 0: Unimodal pretraining. The image encoder is pretrained contrastively on image-text pairs from the web, following the SigLIP training protocol. The text encoder-decoder is a 3B UL2 model trained following the mixture of denoisers procedure.

-   Stage 1: Multimodal training. The image encoder is combined with the text encoder-decoder and trained on a multimodal task and data mixture, keeping the image encoder frozen and using its native resolution.

-   Stage 2: Resolution increase. The resolution of the model is increased by fine-tuning the whole model with a short curriculum of increasing resolutions.

-   Task specialization (transfer). Finally, for each individual task, the model is fine-tuned with frozen ViT image encoder on the task's training data.

Please note that this is a high-level representation and the actual implementation might involve more details and complexities.



------

# Vit Architecture
Here are the ASCII diagrams for the ViT (Vision Transformer)

```
ViT (Vision Transformer):

Image Input
    |
    V
Patch Extraction
    |
    V
Linear Embedding
    |
    V
Positional Encoding
    |
    V
Transformer Encoder Blocks (Multiple Layers)
    |
    V
Classification Head (Optional)
    |
    V
Output (Image Embeddings)

```

The ViT starts with patch extraction from the input image. These patches are then linearly embedded and positional encodings are added. The resulting sequence of patch embeddings is passed through multiple layers of transformer encoders. Optionally, a classification head can be added at the end to get class probabilities for image classification tasks. The output of the ViT is the image embeddings.

-------

# UL2 Encoder/Decoder Transformer
```
Encoder-Decoder Architecture:

Input (Image + Text Tokens)
    |
    V
Transformer Encoder
    |
    V
Encoder Output (Context for Decoder)
    |
    V
Transformer Decoder
    |
    V
Output (Generated Text)

```

The encoder-decoder architecture starts with the input, which is a combination of image and text tokens in this case. The input is passed through a transformer encoder, which generates a context for the decoder. The transformer decoder then uses this context to generate the output text.


# Dataset Strategy
Here is a table summarizing the key datasets mentioned in the paper along with their metadata and source links:

- Made with claude so links could be fake

| Dataset | Type | Size | Tasks | Source |
|-|-|-|-|-|
| ImageNet-22k | Image Classification | 14M images, 21,841 classes | Pretraining | https://github.com/google-research-datasets/ImageNet-21k-P |
| MS COCO | Image Captioning, VQA | 330K images, 80 object categories | Evaluation | https://cocodataset.org | 
| Flickr30k | Image Captioning | 31K images | Evaluation | https://www.kaggle.com/dataset/flickr30k |
| VQAv2 | Visual QA | 204K images, 1.1M questions | Evaluation | https://visualqa.org/download.html |  
| GQA | Visual QA | 22M graph-based questions | Evaluation | https://cs.stanford.edu/people/dorarad/gqa/download.html |
| RefCOCO/RefCOCO+ | Referring Expression | 19,994/19,992 images | Evaluation | https://github.com/lichengunc/refer |
| TextCaps | Image Captioning | 31,014 images | Evaluation | https://textvqa.org/textcaps |
| TextVQA | Visual QA | 28,408 images | Evaluation | https://textvqa.org/index.html |
| STVQA | Visual QA | 249,991 QA pairs | Evaluation | https://tvqa.cs.unc.edu/ |
| OCR-VQA | Visual QA | 45,336 images | Evaluation | https://ocrvqa.cloudcv.org/ |
| DocVQA | Visual QA | 5,000 document images | Evaluation | https://github.com/doc-vqa/docvqa |
| InfographiVQA | Visual QA | 10,047 infographic images | Evaluation | https://github.com/doc-vqa/InfoVQA |
| WebLI | Image-Text Pairs | 72M image-text pairs in 100+ languages | Pretraining | https://laion.ai/blogs/webli/ |
| JFT-300M | Image Classification | 303M images, 18,291 classes | Pretraining | https://github.com/google-research-datasets/jft300m |
| CrossModal-3600 | Image-Text Retrieval | 31K images, 3600 lang-image pairs | Evaluation | https://laion.ai/crossmodal-3600/ |

-----

# License
MIT

# Todo

- [x] Implement sig_lip vit model with training recipe
- [x] Implement the text tokenizer, maybe use token monster 
- [x] Implement the UL2 Transformer Encoder and Decoder
- [ ] Implement the pooling layer after vit then linear
- [ ] Implement the prepending the visual token embeddings to the text embeddings
- [ ] Implement training scripts for the full pali3 model


# Citation

```bibtex
@misc{2310.09199,
Author = {Xi Chen and Xiao Wang and Lucas Beyer and Alexander Kolesnikov and Jialin Wu and Paul Voigtlaender and Basil Mustafa and Sebastian Goodman and Ibrahim Alabdulmohsin and Piotr Padlewski and Daniel Salz and Xi Xiong and Daniel Vlasic and Filip Pavetic and Keran Rong and Tianli Yu and Daniel Keysers and Xiaohua Zhai and Radu Soricut},
Title = {PaLI-3 Vision Language Models: Smaller, Faster, Stronger},
Year = {2023},
Eprint = {arXiv:2310.09199},
}
```



