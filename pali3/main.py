import torch
from torch import nn
from pali3.ul2 import ViTransformerWrapper, Encoder, UL2
from transformers import AutoTokenizer


###########


class PrependTokens(nn.Module):
    """

    # Initialize models
    vit_model = ViTModel()
    text_embedding = TextEmbedding("bert-base-uncased")

    # Initialize PrependVisualTokens
    prepend_visual_tokens = PrependVisualTokens(vit_model, text_embedding)

    # Process image and text
    img = torch.randn(1, 3, 256, 256)  # dummy image
    text = "This is a sample text"
    combined_tokens = prepend_visual_tokens.process(img, text)

    """

    def __init__(
        self,
        vit,
        text_embedding,
    ):
        super().__init__()
        self.vit = vit
        self.text_embedding = text_embedding

    def forward(self, x):
        visual_tokens = self.vit.process(x)
        text_tokens = self.text_embedding.process(x)
        combined_tokens = torch.cat((visual_tokens, text_tokens), dim=1)
        return combined_tokens


class VitModel:
    def __init__(
        self, image_size=256, patch_size=32, dim=512, depth=6, heads=8, *args, **kwargs
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        self.depth = depth
        self.heads = heads

        self.vit = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(dim=dim, depth=depth, heads=heads),
        )
        # adaptive avg pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_projection = nn.Linear(dim, dim)

    def process(self, img):
        if img is None:
            raise ValueError("Input image cannot be None")
        if img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(
                "Input image must have the shape [*, 3, {}, {}]".format(
                    self.image_size, self.image_size
                )
            )
        vit_output = self.vit(img, return_embeddings=True)
        pooled_output = self.pool(vit_output)
        projected_output = self.linear_projection(pooled_output)
        return projected_output


x = torch.randn(1, 3, 256, 256)
model = VitModel()
model.process(x).shape


class Pali3:
    def __init__(
        self,
        model_name=None,
        image_size=256,
        patch_size=32,
        dim=512,
        depth=6,
        heads=8,
        enc_num_tokens=256,
        enc_max_seq_len=1024,
        dec_num_tokens=256,
        dec_max_seq_len=1024,
        enc_depth=6,
        enc_heads=8,
        dec_depth=6,
        dec_heads=8,
        seq_len=1024,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vit_model = VitModel(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
        )

        self.pali_model = UL2(
            dim=dim,
            enc_num_tokens=enc_num_tokens,
            enc_depth=enc_depth,
            enc_heads=enc_heads,
            enc_max_seq_len=enc_max_seq_len,
            dec_num_tokens=dec_num_tokens,
            dec_depth=dec_depth,
            dec_heads=dec_heads,
            dec_max_seq_len=dec_max_seq_len,
        )

    def process(self, img, prompt, output, mask):
        img_embeds = self.vit_model.process(img)

        result = self.pali_model(
            prompt, output, mask=mask, src_prepend_embeds=img_embeds
        )
        return result

    def generate(self, text, mask=None, attn_mask=None, model_name=None):
        if model_name:
            self.model_name = model_name

        if not self.model_name:
            raise ValueError(
                "model_name must be specidfied either in the class constructor or in the generate method"
            )

        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        inputs = self.tokenizer.encode(text, return_tensors="pt")
        seq_out_start = torch.zeros(1, 1).long()
        result = self.pali_model.generate(
            inputs, seq_out_start, self.seq_len, mask, attn_mask
        )
        result_text = self.tokenizer.decode(result[0], skip_special_tokens=True)
        return result_text
