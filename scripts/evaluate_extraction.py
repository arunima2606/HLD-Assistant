"""Precision / recall / F1 of entity extraction against a ground-truth JSON.

Usage:  python -m scripts.evaluate_extraction [--pdf data/sample_hld_v1.0.pdf] [--truth data/ground_truth_v1.0.json]
Uses whatever LLM settings are in the environment (MOCK_LLM=1 -> deterministic table parsing only).
Runs in a throwaway data dir so your real projects are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.textutil import norm_key  # noqa: E402


def truth_keys(gt: dict) -> dict[str, dict[str, dict]]:
    return {
        "component": {norm_key(n): {} for n in gt["components"]},
        "interface": {norm_key(n): {} for n in gt["interfaces"]},
        "port": {f"{norm_key(p['component'])}.{norm_key(p['name'])}": {"direction": p["direction"], "interface": p["interface"]}
                 for p in gt["ports"]},
        "signal": {norm_key(s["name"]): {"sender": s["sender"], "receiver": s["receiver"]} for s in gt["signals"]},
        "dependency": {f"{norm_key(d['from'])}->{norm_key(d['to'])}": {"dep_type": d["type"]} for d in gt["dependencies"]},
    }


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def evaluate(pdf: Path, truth: Path) -> dict[str, dict]:
    from app import pipeline
    from app.db import Document, Entity, Project, session_scope

    gt = truth_keys(json.loads(Path(truth).read_text()))
    with session_scope() as s:
        proj = Project(name="eval")
        s.add(proj)
        s.commit()
        doc = pipeline.register_document(s, proj.id, Path(pdf).name, "eval", Path(pdf))
        doc_id = doc.id
    pipeline.process_document(doc_id)
    with session_scope() as s:
        status = s.get(Document, doc_id)
        if status.status != "ready":
            raise SystemExit(f"Processing failed: {status.message}")
        ents = s.query(Entity).filter_by(document_id=doc_id).all()
        got: dict[str, dict[str, dict]] = {t: {} for t in gt}
        for e in ents:
            got.setdefault(e.type, {})[e.key] = e.attrs

    report = {}
    for typ, want in gt.items():
        have = got.get(typ, {})
        tp = len(set(want) & set(have))
        p, r, f = prf(tp, len(set(have) - set(want)), len(set(want) - set(have)))
        hits = [(k, a) for k, a in want.items() if k in have]
        ok = sum(all(norm_key(str(have[k].get(f, ""))) == norm_key(str(v)) for f, v in a.items()) for k, a in hits)
        report[typ] = {"precision": p, "recall": r, "f1": f, "attr_accuracy": round(ok / len(hits), 3) if hits else 0.0,
                       "missing": sorted(set(want) - set(have)), "extra": sorted(set(have) - set(want))}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "data" / "sample_hld_v1.0.pdf"))
    ap.add_argument("--truth", default=str(ROOT / "data" / "ground_truth_v1.0.json"))
    a = ap.parse_args()
    if not Path(a.pdf).exists() or not Path(a.truth).exists():  # first run: create the sample data
        from scripts.make_sample_hld import generate
        generate(ROOT / "data")
    os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="hld_eval_")
    rep = evaluate(Path(a.pdf), Path(a.truth))
    print(f"{'type':<12}{'precision':>10}{'recall':>9}{'f1':>7}{'attr_acc':>10}")
    for t, m in rep.items():
        print(f"{t:<12}{m['precision']:>10}{m['recall']:>9}{m['f1']:>7}{m['attr_accuracy']:>10}")
        if m["missing"]:
            print(f"   missing: {m['missing']}")
        if m["extra"]:
            print(f"   extra:   {m['extra']}")


if __name__ == "__main__":
    main()
