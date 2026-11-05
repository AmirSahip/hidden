"""Canonicalize the author USPTO-50K split for the decoder baseline.

Train and validation are the only rows a later training run may read.
The test file is written so the same canonicalization exists, and is marked held out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

SOURCE = Path(
    r"C:\Users\amirs\OneDrive\Рабочий стол\4year_Fall\Senior"
    r"\retropath_four_person_experiments_2026-10-15 (2)"
    r"\retropath_four_person_experiments_2026-10-15"
    r"\data\reactiont5_author_uspto50k"
)
OUT = Path(__file__).resolve().parent / "data"
SPLITS = ("train", "val", "test")


def canonicalize(smiles: str) -> str | None:
    parts = [part for part in str(smiles).split(".") if part]
    if not parts:
        return None
    canonical = []
    for part in parts:
        molecule = Chem.MolFromSmiles(part)
        if molecule is None:
            return None
        canonical.append(Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True))
    return ".".join(sorted(canonical))


def convert(split: str) -> dict:
    frame = pd.read_csv(SOURCE / f"{split}.csv", keep_default_na=False)
    rows = []
    dropped = 0
    for index, row in frame.iterrows():
        product = canonicalize(row["PRODUCT"])
        reactants = canonicalize(row["REACTANT"])
        if product is None or reactants is None:
            dropped += 1
            continue
        rows.append(
            {
                "source_index": int(index),
                "split": split,
                "product": product,
                "reactants": reactants,
            }
        )
    destination = OUT / ("test_heldout.jsonl" if split == "test" else f"{split}.jsonl")
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"split": split, "source_rows": int(len(frame)), "kept": len(rows), "dropped": dropped, "path": str(destination)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(SOURCE), "test_status": "held out; do not train on test_heldout.jsonl", "splits": [convert(split) for split in SPLITS]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
