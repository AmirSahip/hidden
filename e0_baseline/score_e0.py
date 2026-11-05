"""Greedy top-1 precursor recovery on the held-out USPTO-50K test split.

Reads test_heldout.jsonl for scoring only. Does not train and does not download.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch

from model import load_model
from prepare_data import canonicalize
from smoke import read_rows

TEST = Path(__file__).resolve().parent / "data" / "test_heldout.jsonl"
CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "decoder_latest.pt"
REPORT = Path(__file__).resolve().parent / "e0_greedy_report.json"
BATCH = 8
MAX_NEW_TOKENS = 160
MIN_FREE_GB = 8.0


def free_gb() -> float:
    return shutil.disk_usage(REPORT.parent).free / (1024 ** 3)


def encode_batch(rows: list[dict], tokenizer) -> dict:
    encoded = [tokenizer(row["product"], truncation=True, max_length=160, add_special_tokens=True) for row in rows]
    width = max(len(item["input_ids"]) for item in encoded)
    pad_id = tokenizer.pad_token_id
    input_ids = [item["input_ids"] + [pad_id] * (width - len(item["input_ids"])) for item in encoded]
    attention = [item["attention_mask"] + [0] * (width - len(item["attention_mask"])) for item in encoded]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def generated_smiles(tokens: list[int], tokenizer) -> str:
    body = tokens[1:]
    if tokenizer.sep_token_id in body:
        body = body[: body.index(tokenizer.sep_token_id)]
    return tokenizer.decode(body, skip_special_tokens=True).replace(" ", "")


def main() -> None:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else CHECKPOINT
    report_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REPORT
    if free_gb() < MIN_FREE_GB:
        raise SystemExit(f"free space is below {MIN_FREE_GB} GB; scoring was not started")
    tokenizer, model = load_model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.decoder.load_state_dict(state["decoder"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.decoder.eval()
    rows = read_rows(TEST)
    exact = 0
    canonical_hits = 0
    invalid = 0
    examples = []
    started = time.time()
    for start in range(0, len(rows), BATCH):
        if free_gb() < MIN_FREE_GB:
            report_path.write_text(json.dumps({"stopped_low_disk": True, "scored": start, "free_gb": round(free_gb(), 2)}, indent=2), encoding="utf-8")
            raise SystemExit(0)
        chunk = rows[start : start + BATCH]
        batch = encode_batch(chunk, tokenizer)
        tokens = model.generate(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for row, sequence in zip(chunk, tokens.tolist()):
            prediction = generated_smiles(sequence, tokenizer)
            canonical = canonicalize(prediction)
            if canonical is None:
                invalid += 1
            elif canonical == row["reactants"]:
                canonical_hits += 1
            if prediction == row["reactants"]:
                exact += 1
            if len(examples) < 8:
                examples.append({"product": row["product"], "recorded_reactants": row["reactants"], "generated": prediction, "canonical": canonical})
        done = min(start + BATCH, len(rows))
        if done % 400 == 0 or done == len(rows):
            print(json.dumps({"scored": done, "canonical_top1": canonical_hits, "invalid": invalid, "free_gb": round(free_gb(), 2), "seconds": round(time.time() - started, 1)}), flush=True)
    report = {
        "split": "test_heldout",
        "rows": len(rows),
        "decode": "greedy",
        "checkpoint": str(checkpoint),
        "checkpoint_stage": state.get("stage"),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_loss": state.get("loss"),
        "exact_string_top1": exact,
        "canonical_multiset_top1": canonical_hits,
        "canonical_top1_rate": round(canonical_hits / len(rows), 4),
        "invalid_smiles": invalid,
        "free_gb": round(free_gb(), 2),
        "seconds": round(time.time() - started, 1),
        "examples": examples,
        "note": "Greedy top-1 only. This is not a beam-search top-k or MRR.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("rows", "canonical_multiset_top1", "canonical_top1_rate", "invalid_smiles", "free_gb")}), flush=True)


if __name__ == "__main__":
    main()
