import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(
        self,
        hf_name: str,
        hf_path: str | None = None,
        max_length: int = 77,
        torch_dtype=torch.float16,
    ):
        super().__init__()
        self.is_t5 = "t5" in hf_name.lower()
        if self.is_t5:
            self.tokenizer = T5Tokenizer.from_pretrained(hf_name if hf_path is None else hf_path)
            self.model = T5EncoderModel.from_pretrained(
                hf_name if hf_path is None else hf_path, torch_dtype=torch_dtype
            )
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(hf_name if hf_path is None else hf_path)
            self.model = CLIPTextModel.from_pretrained(
                hf_name if hf_path is None else hf_path, torch_dtype=torch_dtype
            )
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, text: list[str]):
        text_inputs = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        if self.is_t5:
            prompt_embeds = self.model(text_input_ids.to(self.model.device))[0]
        else:
            prompt_embeds = self.model(text_input_ids.to(self.model.device), output_hidden_states=True).hidden_states[-2]
        return prompt_embeds
