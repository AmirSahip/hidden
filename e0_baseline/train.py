"""Train the decoder on the USPTO-50K training reactions.

Uses local files only. Does not read test_heldout.jsonl.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import load_model
from smoke import MAX_LENGTH, ReactionSet, collate, fits, read_rows

TRAIN = Path(__file__).resolve().parent / "data" / "train.jsonl"
VAL = Path(__file__).resolve().parent / "data" / "val.jsonl"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
LOG = Path(__file__).resolve().parent / "train_log.jsonl"
EPOCHS = 3
BATCH = 4
SEED = 42
MIN_FREE_GB = 8.0


def free_gb() -> float:
    return shutil.disk_usage(CHECKPOINT_DIR).free / (1024 ** 3)


def save_checkpoint(model, epoch: int, step: int, loss: float) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / "decoder_latest.pt"
    torch.save({"epoch": epoch, "step": step, "loss": loss, "decoder": model.decoder.state_dict()}, path)
    return path


def log(record: dict) -> None:
    record["time"] = time.strftime("%H:%M:%S")
    print(json.dumps(record), flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_epoch(model, loader, optimizer, loss_fn, device, epoch: int) -> float:
    model.decoder.train()
    total = 0.0
    seen = 0
    started = time.time()
    for step, batch in enumerate(loader, start=1):
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
            current = total / seen
            free = free_gb()
            log({"event": "step", "epoch": epoch, "step": step, "loss": round(current, 4), "rows": seen, "seconds": round(time.time() - started, 1), "free_gb": round(free, 2)})
            if free < MIN_FREE_GB:
                path = save_checkpoint(model, epoch, step, current)
                log({"event": "stopped_low_disk", "free_gb": round(free, 2), "checkpoint": str(path)})
                raise SystemExit(0)
    return total / max(seen, 1)


def samples(model, tokenizer, rows, device) -> list[dict]:
    model.decoder.eval()
    batch = collate([ReactionSet(rows[:4], tokenizer)[i] for i in range(min(4, len(rows)))], tokenizer.pad_token_id)
    generated = model.generate(batch["input_ids"].to(device), batch["attention_mask"].to(device), tokenizer.cls_token_id, tokenizer.sep_token_id)
    preview = []
    for index, tokens in enumerate(generated.tolist()):
        body = tokens[1:]
        if tokenizer.sep_token_id in body:
            body = body[: body.index(tokenizer.sep_token_id)]
        preview.append(
            {
                "product": batch["product"][index],
                "recorded_reactants": batch["reactants"][index],
                "generated": tokenizer.decode(body, skip_special_tokens=True).replace(" ", ""),
            }
        )
    return preview


def main() -> None:
    torch.manual_seed(SEED)
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_rows = [row for row in read_rows(TRAIN) if fits(row, tokenizer)]
    val_rows = [row for row in read_rows(VAL) if fits(row, tokenizer)]
    log({"event": "start", "device": str(device), "train_rows": len(train_rows), "val_rows": len(val_rows), "max_length": MAX_LENGTH, "epochs": EPOCHS, "downloads": "none"})
    loader = DataLoader(ReactionSet(train_rows, tokenizer), batch_size=BATCH, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        loss = run_epoch(model, loader, optimizer, loss_fn, device, epoch)
        path = save_checkpoint(model, epoch, step=0, loss=loss)
        log({"event": "epoch", "epoch": epoch, "loss": round(loss, 4), "checkpoint": str(path), "val_preview": samples(model, tokenizer, val_rows, device)})


if __name__ == "__main__":
    main()
