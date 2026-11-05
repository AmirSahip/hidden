"""Teach the decoder to write SMILES, then fine-tune on USPTO-50K train.

The PubChem file is a local sample. The held-out USPTO test is not read here.
The baseline checkpoint decoder_latest.pt is left unchanged.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import load_model
from smoke import BATCH, MAX_LENGTH, ReactionSet, collate, fits, read_rows

MOL = Path(__file__).resolve().parent / "data" / "pubchem_sample.jsonl"
TRAIN = Path(__file__).resolve().parent / "data" / "train.jsonl"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG = Path(__file__).resolve().parent / "train_spelling_log.jsonl"
MOL_EPOCHS = 2
USPTO_EPOCHS = 3
MOL_ROWS = 100_000
SEED = 42
MIN_FREE_GB = 8.0


def free_gb() -> float:
    return shutil.disk_usage(CHECKPOINT_DIR).free / (1024 ** 3)


def log(record: dict) -> None:
    record["time"] = time.strftime("%H:%M:%S")
    record["free_gb"] = round(free_gb(), 2)
    print(json.dumps(record), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def save_checkpoint(model, name: str, meta: dict) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / name
    payload = {"decoder": model.decoder.state_dict(), **meta}
    torch.save(payload, path)
    return path


def run_epoch(model, loader, optimizer, loss_fn, device, stage: str, epoch: int) -> float:
    model.decoder.train()
    total = 0.0
    seen = 0
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        if step % 500 == 0 and free_gb() < MIN_FREE_GB:
            path = save_checkpoint(model, "decoder_spelling_partial.pt", {"stage": stage, "epoch": epoch, "step": step, "loss": total / max(seen, 1)})
            log({"event": "stopped_low_disk", "stage": stage, "epoch": epoch, "step": step, "checkpoint": str(path)})
            raise SystemExit(0)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        logits = model(input_ids, attention_mask, target_ids[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * input_ids.size(0)
        seen += input_ids.size(0)
        if step % 500 == 0:
            log({"event": "step", "stage": stage, "epoch": epoch, "step": step, "loss": round(total / seen, 4), "rows": seen, "seconds": round(time.time() - started, 1)})
    return total / max(seen, 1)


def train_stage(model, rows, tokenizer, optimizer, loss_fn, device, stage: str, epochs: int) -> float:
    loader = DataLoader(
        ReactionSet(rows, tokenizer),
        batch_size=BATCH,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    loss = 0.0
    for epoch in range(1, epochs + 1):
        loss = run_epoch(model, loader, optimizer, loss_fn, device, stage, epoch)
        log({"event": "epoch", "stage": stage, "epoch": epoch, "loss": round(loss, 4), "rows": len(rows), "max_length": MAX_LENGTH})
    return loss


def main() -> None:
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"free space is below {MIN_FREE_GB} GB; training was not started")
    torch.manual_seed(SEED)
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    molecules = [row for row in read_rows(MOL) if fits(row, tokenizer)][:MOL_ROWS]
    reactions = [row for row in read_rows(TRAIN) if fits(row, tokenizer)]
    log({"event": "start", "device": str(device), "molecules": len(molecules), "uspto_train": len(reactions), "mol_epochs": MOL_EPOCHS, "uspto_epochs": USPTO_EPOCHS, "downloads": "pubchem sample already local"})
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    mol_loss = train_stage(model, molecules, tokenizer, optimizer, loss_fn, device, "pubchem", MOL_EPOCHS)
    mol_path = save_checkpoint(model, "decoder_pubchem.pt", {"stage": "pubchem", "epoch": MOL_EPOCHS, "loss": mol_loss})
    log({"event": "saved", "checkpoint": str(mol_path)})
    uspto_loss = train_stage(model, reactions, tokenizer, optimizer, loss_fn, device, "uspto", USPTO_EPOCHS)
    uspto_path = save_checkpoint(model, "decoder_pubchem_uspto.pt", {"stage": "uspto", "epoch": USPTO_EPOCHS, "loss": uspto_loss})
    log({"event": "saved", "checkpoint": str(uspto_path)})


if __name__ == "__main__":
    main()
