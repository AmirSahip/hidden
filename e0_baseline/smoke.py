"""Train the decoder on a few hundred training reactions and print sample outputs.

This does not read the held-out test file and is not an E0 result.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import load_model

DATA = Path(__file__).resolve().parent / "data" / "train.jsonl"
REPORT = Path(__file__).resolve().parent / "smoke_report.json"
MAX_LENGTH = 160
SAMPLES = 256
EPOCHS = 2
BATCH = 4
SEED = 42


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ReactionSet(Dataset):
    def __init__(self, rows: list[dict], tokenizer) -> None:
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        source = self.tokenizer(row["product"], truncation=True, max_length=MAX_LENGTH, add_special_tokens=True)
        target = self.tokenizer(row["reactants"], truncation=True, max_length=MAX_LENGTH - 2, add_special_tokens=False)
        target_ids = [self.tokenizer.cls_token_id, *target["input_ids"], self.tokenizer.sep_token_id]
        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "target_ids": target_ids,
            "product": row["product"],
            "reactants": row["reactants"],
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    def pad(key: str, fill: int) -> torch.Tensor:
        width = max(len(item[key]) for item in batch)
        rows = [item[key] + [fill] * (width - len(item[key])) for item in batch]
        return torch.tensor(rows, dtype=torch.long)

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": pad("attention_mask", 0),
        "target_ids": pad("target_ids", pad_id),
        "product": [item["product"] for item in batch],
        "reactants": [item["reactants"] for item in batch],
    }


def fits(row: dict, tokenizer) -> bool:
    source = tokenizer(row["product"], add_special_tokens=True, truncation=False)
    target = tokenizer(row["reactants"], add_special_tokens=False, truncation=False)
    return len(source["input_ids"]) <= MAX_LENGTH and len(target["input_ids"]) + 2 <= MAX_LENGTH


def round_trip(rows: list[dict], tokenizer) -> dict:
    checked = rows[:200]
    failures = []
    for row in checked:
        for key in ("product", "reactants"):
            encoded = tokenizer(row[key], add_special_tokens=False, truncation=False)
            decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True).replace(" ", "")
            if decoded != row[key]:
                failures.append({"field": key, "source": row[key], "decoded": decoded})
                break
    return {"checked": len(checked), "failures": len(failures), "examples": failures[:5]}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    rows = [row for row in read_rows(DATA) if fits(row, tokenizer)]
    random.shuffle(rows)
    chosen = rows[:SAMPLES]
    report = {"device": str(device), "tokenizer_round_trip": round_trip(chosen, tokenizer), "train_rows_available": len(rows), "smoke_rows": len(chosen)}
    if report["tokenizer_round_trip"]["failures"]:
        REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit("tokenizer did not round-trip SMILES; smoke training was not started")

    loader = DataLoader(ReactionSet(chosen, tokenizer), batch_size=BATCH, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    losses = []
    model.decoder.train()
    for epoch in range(EPOCHS):
        total = 0.0
        seen = 0
        for batch in loader:
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
        losses.append(total / seen)
        print(f"epoch {epoch + 1} loss {losses[-1]:.4f}", flush=True)

    preview = next(iter(loader))
    generated = model.generate(
        preview["input_ids"][:4].to(device),
        preview["attention_mask"][:4].to(device),
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
    )
    examples = []
    for index, tokens in enumerate(generated.tolist()):
        body = tokens[1:]
        if tokenizer.sep_token_id in body:
            body = body[: body.index(tokenizer.sep_token_id)]
        examples.append(
            {
                "product": preview["product"][index],
                "recorded_reactants": preview["reactants"][index],
                "generated": tokenizer.decode(body, skip_special_tokens=True).replace(" ", ""),
            }
        )
    report.update({"epoch_loss": losses, "examples": examples})
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()
