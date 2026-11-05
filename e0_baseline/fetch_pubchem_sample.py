"""Save a small PubChem SMILES sample and stop. The full dump is not stored."""

from __future__ import annotations

import gzip
import json
import urllib.request
from pathlib import Path

URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
OUT = Path(__file__).resolve().parent / "data" / "pubchem_sample.jsonl"
KEEP = 100_000
MAX_CHARS = 140


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "CheMLT-F-e0-sample/1.0"})
    kept = 0
    seen = 0
    with urllib.request.urlopen(request, timeout=120) as response, gzip.GzipFile(fileobj=response) as stream, OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for raw in stream:
            seen += 1
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            smiles = line.split()[-1]
            if not smiles or len(smiles) > MAX_CHARS or "." in smiles:
                continue
            handle.write(json.dumps({"product": smiles, "reactants": smiles}, ensure_ascii=False) + "\n")
            kept += 1
            if kept >= KEEP:
                break
            if kept % 20000 == 0:
                print(json.dumps({"kept": kept, "seen": seen}), flush=True)
    print(json.dumps({"kept": kept, "seen": seen, "bytes": OUT.stat().st_size, "path": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
