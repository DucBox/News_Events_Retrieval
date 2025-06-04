import torch
import clip
from PIL import Image
from typing import Union


class CLIPEmbedder:

    def __init__(self, model_name: str = "ViT-L/14", device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def encode(self, inp: Union[Image.Image, str]):
        if isinstance(inp, Image.Image):
            return self.encode_image(inp)
        if isinstance(inp, str):
            return self.encode_text(inp)
        raise TypeError(f"Unsupported input: {type(inp)}")

    @torch.no_grad()
    def encode_image(self, img: Image.Image):
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(tensor)
        return (emb / emb.norm(p=2, dim=-1, keepdim=True)).squeeze(0)

    @torch.no_grad()
    def encode_text(self, text: str):
        toks = clip.tokenize([text]).to(self.device)
        emb = self.model.encode_text(toks)
        return (emb / emb.norm(p=2, dim=-1, keepdim=True)).squeeze(0)
