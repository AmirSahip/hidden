"""Frozen CheMLT-F DeBERTa encoder plus a small transformer decoder."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from transformers import DebertaTokenizerFast, DebertaV2ForMaskedLM

ENCODER_DIR = Path(__file__).resolve().parents[1] / "Pre-Training" / "PubchemModelDeberta"


class SinusoidalPositions(nn.Module):
    def __init__(self, width: int, length: int) -> None:
        super().__init__()
        position = torch.arange(length).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.encoding[: tokens.size(1)]


class RetroDecoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, layers: int = 2, width: int = 768, heads: int = 8, feedforward: int = 2048, max_length: int = 192) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length
        self.embed = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.positions = SinusoidalPositions(width, max_length)
        block = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(block, num_layers=layers)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.output.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor, memory_padding: torch.Tensor) -> torch.Tensor:
        # True marks positions the token is not allowed to see. Same dtype as the padding masks.
        causal = torch.triu(torch.ones(tokens.size(1), tokens.size(1), dtype=torch.bool, device=tokens.device), diagonal=1)
        hidden = self.decoder(
            self.positions(self.embed(tokens)),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tokens.eq(self.pad_id),
            memory_key_padding_mask=memory_padding,
        )
        return self.output(hidden)


class RetroModel(nn.Module):
    def __init__(self, encoder: DebertaV2ForMaskedLM, pad_id: int, layers: int = 2) -> None:
        super().__init__()
        self.encoder = encoder.deberta
        self.decoder = RetroDecoder(encoder.config.vocab_size, pad_id, layers=layers)
        with torch.no_grad():
            self.decoder.embed.weight.copy_(encoder.deberta.embeddings.word_embeddings.weight)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden, attention_mask.eq(0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, decoder_tokens: torch.Tensor) -> torch.Tensor:
        memory, memory_padding = self.encode(input_ids, attention_mask)
        return self.decoder(decoder_tokens, memory, memory_padding)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, bos_id: int, eos_id: int, max_new_tokens: int = 80) -> torch.Tensor:
        self.decoder.eval()
        memory, memory_padding = self.encode(input_ids, attention_mask)
        tokens = torch.full((input_ids.size(0), 1), bos_id, dtype=torch.long, device=input_ids.device)
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        for _ in range(max_new_tokens):
            if bool(finished.all()) or tokens.size(1) >= self.decoder.max_length:
                break
            step = self.decoder(tokens, memory, memory_padding)[:, -1].argmax(dim=-1)
            step = torch.where(finished, torch.full_like(step, self.decoder.pad_id), step)
            finished = finished | step.eq(eos_id)
            tokens = torch.cat([tokens, step.unsqueeze(1)], dim=1)
        return tokens


def load_tokenizer(path: Path = ENCODER_DIR) -> DebertaTokenizerFast:
    return DebertaTokenizerFast.from_pretrained(path)


def load_model(path: Path = ENCODER_DIR, layers: int = 2) -> tuple[DebertaTokenizerFast, RetroModel]:
    tokenizer = load_tokenizer(path)
    encoder = DebertaV2ForMaskedLM.from_pretrained(path)
    model = RetroModel(encoder, pad_id=tokenizer.pad_token_id, layers=layers)
    return tokenizer, model
