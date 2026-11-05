"""Train on local ORD reactions, then fine-tune on USPTO-50K train.

Does not download ORD and does not read the USPTO test split.
Starts from the decoder that already learned to write SMILES.
Leaves decoder_latest.pt unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import train_spelling as spelling
from model import load_model
from smoke import fits, read_rows

ORD = Path(__file__).resolve().parent / "data" / "ord.jsonl"
TRAIN = Path(__file__).resolve().parent / "data" / "train.jsonl"
START = Path(__file__).resolve().parent / "checkpoints" / "decoder_pubchem.pt"
ORD_EPOCHS = 1
USPTO_EPOCHS = 3
SEED = 42


def main() -> None:
    if not ORD.exists():
        raise SystemExit(
            "data/ord.jsonl is not here. Export ORD first as one JSON object per line "
            "with fields product and reactants, canonical SMILES, reactants joined by '.'. "
            "This script does not download it."
        )
    if spelling.free_gb() < spelling.MIN_FREE_GB:
        raise SystemExit(f"free space is below {spelling.MIN_FREE_GB} GB; training was not started")
    spelling.LOG = Path(__file__).resolve().parent / "train_ord_log.jsonl"
    batch = int(os.environ.get("E0_BATCH", "32"))
    torch.manual_seed(SEED)
    tokenizer, model = load_model()
    state = torch.load(START, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(state["decoder"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    reactions = [row for row in read_rows(ORD) if fits(row, tokenizer)]
    uspto = [row for row in read_rows(TRAIN) if fits(row, tokenizer)]
    spelling.BATCH = batch
    spelling.log({"event": "start", "device": str(device), "ord_rows": len(reactions), "uspto_train": len(uspto), "batch": batch, "downloads": "none"})
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    ord_loss = spelling.train_stage(model, reactions, tokenizer, optimizer, loss_fn, device, "ord", ORD_EPOCHS)
    spelling.save_checkpoint(model, "decoder_ord.pt", {"stage": "ord", "epoch": ORD_EPOCHS, "loss": ord_loss})
    uspto_loss = spelling.train_stage(model, uspto, tokenizer, optimizer, loss_fn, device, "uspto", USPTO_EPOCHS)
    path = spelling.save_checkpoint(model, "decoder_ord_uspto.pt", {"stage": "uspto", "epoch": USPTO_EPOCHS, "loss": uspto_loss})
    spelling.log({"event": "saved", "checkpoint": str(path)})


if __name__ == "__main__":
    main()
