"""Convert Open Reaction Database dumps into e0_baseline/data/ord.jsonl.

Prints the ORD dump size first, then downloads only if a local copy is missing.
Each output line is {"product": "...", "reactants": "..."} with canonical
isomeric SMILES and reactants joined by ".", matching data/train.jsonl.

Does not download vmun/CheMLT-F Datasets.
Does not read data/test_heldout.jsonl for training; if that file is present
next to the output, matching products are dropped from ORD to avoid leakage.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ORD_HF_ID = "open-reaction-database/ord-data"
ORD_HF_URL = f"https://huggingface.co/datasets/{ORD_HF_ID}"
SIZE_API = f"https://huggingface.co/api/datasets/{ORD_HF_ID}/tree/main/data?recursive=true"


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_out() -> Path:
    env = os.environ.get("E0_ORD_OUT")
    if env:
        return Path(env).expanduser().resolve()
    here = script_dir()
    for candidate in (
        here / "e0_baseline" / "data" / "ord.jsonl",
        here / "data" / "ord.jsonl",
        Path.cwd() / "e0_baseline" / "data" / "ord.jsonl",
        Path.cwd() / "data" / "ord.jsonl",
    ):
        if candidate.parent.is_dir() or candidate.parent.parent.is_dir():
            return candidate
    return here / "e0_baseline" / "data" / "ord.jsonl"


def resolve_ord_root() -> Path:
    env = os.environ.get("ORD_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = script_dir()
    for candidate in (
        here / "ord-data",
        here.parent / "ord-data",
        Path.cwd() / "ord-data",
        Path.home() / "ord-data",
    ):
        if (candidate / "data").is_dir():
            return candidate
    return here / "ord-data"


def http_json(url: str):
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "CheMLT-F-e0-ord/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def report_size() -> dict:
    entries = http_json(SIZE_API)
    files = [
        item
        for item in entries
        if item.get("type") == "file" and str(item.get("path", "")).endswith(".parquet")
    ]
    total = sum(int(item.get("size") or 0) for item in files)
    info = {
        "source": ORD_HF_URL,
        "parquet_files": len(files),
        "bytes": total,
        "gib": round(total / (1024 ** 3), 3),
        "note": "download starts only after this size is printed",
    }
    print(json.dumps({"event": "ord_size", **info}), flush=True)
    return info


def ensure_deps() -> None:
    missing = []
    for name, module in (
        ("rdkit", "rdkit"),
        ("pyarrow", "pyarrow"),
        ("huggingface_hub", "huggingface_hub"),
        ("ord-schema", "ord_schema"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "missing packages: "
            + ", ".join(missing)
            + ". Install with: pip install rdkit pyarrow huggingface_hub 'ord-schema'"
        )


def download_ord(root: Path, expected_bytes: int) -> Path:
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data"
    if data_dir.is_dir() and any(data_dir.rglob("*.parquet")):
        local = sum(path.stat().st_size for path in data_dir.rglob("*.parquet"))
        print(
            json.dumps(
                {
                    "event": "ord_local",
                    "path": str(root),
                    "bytes": local,
                    "gib": round(local / (1024 ** 3), 3),
                }
            ),
            flush=True,
        )
        return root

    print(
        json.dumps(
            {
                "event": "ord_download_start",
                "path": str(root),
                "expected_bytes": expected_bytes,
                "expected_gib": round(expected_bytes / (1024 ** 3), 3),
            }
        ),
        flush=True,
    )
    snapshot_download(
        repo_id=ORD_HF_ID,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=["data/**/*.parquet", "README.md", "LICENSE*", "CITATION*"],
    )
    local = sum(path.stat().st_size for path in (root / "data").rglob("*.parquet"))
    print(
        json.dumps(
            {
                "event": "ord_download_done",
                "path": str(root),
                "bytes": local,
                "gib": round(local / (1024 ** 3), 3),
            }
        ),
        flush=True,
    )
    return root


def canonicalize(smiles: str) -> str | None:
    from rdkit import Chem

    parts = [part for part in str(smiles).split(".") if part]
    if not parts:
        return None
    canonical: list[str] = []
    for part in parts:
        molecule = Chem.MolFromSmiles(part)
        if molecule is None:
            return None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical.append(Chem.MolToSmiles(molecule, isomericSmiles=True, canonical=True))
    return ".".join(sorted(canonical))


def compound_smiles(compound) -> str | None:
    from ord_schema.proto import reaction_pb2

    preferred = (
        reaction_pb2.CompoundIdentifier.CXSMILES,
        reaction_pb2.CompoundIdentifier.SMILES,
    )
    by_type = {identifier.type: identifier.value for identifier in compound.identifiers}
    for kind in preferred:
        value = by_type.get(kind)
        if value:
            return value.split("|", 1)[0].strip() or None
    return None


def extract_pair(reaction) -> tuple[str, str] | None:
    from ord_schema.proto import reaction_pb2

    roles = reaction_pb2.ReactionRole
    reactants: list[str] = []
    products: list[str] = []

    for value in reaction.inputs.values():
        for component in value.components:
            if component.reaction_role != roles.REACTANT:
                continue
            smiles = compound_smiles(component)
            if smiles:
                reactants.append(smiles)

    for outcome in reaction.outcomes:
        for product in outcome.products:
            if product.reaction_role not in (roles.PRODUCT, roles.UNSPECIFIED, 0):
                continue
            smiles = compound_smiles(product)
            if smiles:
                products.append(smiles)

    if not reactants or not products:
        return None
    product = canonicalize(".".join(products))
    reactant = canonicalize(".".join(reactants))
    if product is None or reactant is None:
        return None
    if product == reactant:
        return None
    return product, reactant


def heldout_products(out: Path) -> set[str]:
    path = out.parent / "test_heldout.jsonl"
    if not path.is_file():
        return set()
    blocked: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            product = row.get("product")
            if product:
                blocked.add(product)
    print(
        json.dumps({"event": "heldout_loaded", "path": str(path), "products": len(blocked)}),
        flush=True,
    )
    return blocked


def convert(root: Path, out: Path) -> dict:
    from ord_schema import datasets
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    out.parent.mkdir(parents=True, exist_ok=True)
    blocked = heldout_products(out)
    seen: set[tuple[str, str]] = set()
    kept = 0
    dropped = 0
    heldout_hits = 0
    files = sorted((root / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files under {root / 'data'}")

    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index, path in enumerate(files, start=1):
            view = datasets.load_dataset(str(path))
            for reaction in view.reactions:
                pair = extract_pair(reaction)
                if pair is None:
                    dropped += 1
                    continue
                product, reactants = pair
                if product in blocked:
                    heldout_hits += 1
                    continue
                key = (product, reactants)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                handle.write(
                    json.dumps(
                        {"product": product, "reactants": reactants},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
            if index % 5 == 0 or index == len(files):
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "files_done": index,
                            "files_total": len(files),
                            "kept": kept,
                            "dropped": dropped,
                            "heldout_skipped": heldout_hits,
                        }
                    ),
                    flush=True,
                )

    summary = {
        "event": "done",
        "source": str(root),
        "out": str(out),
        "files": len(files),
        "kept": kept,
        "dropped": dropped,
        "heldout_skipped": heldout_hits,
        "bytes": out.stat().st_size if out.is_file() else 0,
    }
    summary_path = out.with_name("ord_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "Usage: python convert_ord.py\n"
            "Env: ORD_ROOT=path/to/ord-data  E0_ORD_OUT=path/to/ord.jsonl\n"
            "Prints ORD size, downloads open-reaction-database/ord-data if needed,\n"
            "writes e0_baseline/data/ord.jsonl. Never touches CheMLT-F Datasets."
        )
        return
    size = report_size()
    ensure_deps()
    root = download_ord(resolve_ord_root(), size["bytes"])
    convert(root, resolve_out())


if __name__ == "__main__":
    main()
