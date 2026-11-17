"""X-MDD inference: predict learners' actual phones and score MDD.

With no arguments it runs the Arabic example in examples/arabic/:

    python -m xmdd.infer --shots 0 4

Your own data:

    python -m xmdd.infer --model-dir pretrained/X-MDD --language en \
        --manifest my_test.jsonl --support-manifest my_demos.jsonl \
        --constrain-phones my_phones.txt --shots 4 --out-dir results/en
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from xmdd.data import load_manifest, select_demos
from xmdd.metrics import score_unaligned
from xmdd.model import XMDD

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "arabic"
TRAINING_LANGUAGES = {"zh", "en"}  # the phone vocabulary covers these


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default="pretrained/X-MDD",
                        help="released model folder (hf download walston/X-MDD --local-dir pretrained/X-MDD)")
    parser.add_argument("--language", default="ar", help="language id: zh, en, ar, or a new one")
    parser.add_argument("--language-name", default=None,
                        help="language name for a language not in config.yaml, e.g. Spanish")
    parser.add_argument("--manifest", default=str(EXAMPLE / "queries.jsonl"),
                        help="JSONL of the utterances to evaluate")
    parser.add_argument("--support-manifest", default=str(EXAMPLE / "demos.jsonl"),
                        help="JSONL to draw demonstrations from; may be the --manifest itself "
                             "(a query is never its own demonstration)")
    parser.add_argument("--constrain-phones", default=None, metavar="TXT_OR_JSONL",
                        help="phone set decoding is restricted to (.txt, one per line, or a manifest); "
                             "default: unconstrained decoding; 'none' also disables constraints")
    parser.add_argument("--shots", type=int, nargs="+", default=[0],
                        help="numbers of in-context demonstrations, each evaluated separately")
    parser.add_argument("--exclude-same-canonical", action="store_true",
                        help="never use a demonstration that reads the same canonical text as the query")
    parser.add_argument("--exclude-same-speaker", action="store_true",
                        help="never use a demonstration of the query's speaker")
    parser.add_argument("--raw-phones", action=argparse.BooleanOptionalAction, default=None,
                        help="keep every output symbol instead of only phones of the training vocabulary "
                             "(default: on for languages other than zh/en)")
    parser.add_argument("--out-dir", default=None,
                        help="write metrics.json and predictions.jsonl here (default: print only)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    args = parser.parse_args()
    raw_phones = args.raw_phones if args.raw_phones is not None else args.language not in TRAINING_LANGUAGES

    print(f"Loading model from {args.model_dir} ...", flush=True)
    load_started = perf_counter()
    model = XMDD.from_pretrained(args.model_dir, device=args.device, attn_implementation=args.attn_implementation)
    model_load_s = perf_counter() - load_started
    print(f"Model load: {model_load_s:.3f}s", flush=True)
    if args.constrain_phones is not None and args.constrain_phones.lower() != "none":
        model.set_phone_constraint(args.constrain_phones)
    queries = load_manifest(args.manifest, args.language, model.cfg, args.language_name)
    support = load_manifest(args.support_manifest, args.language, model.cfg, args.language_name)
    seed = int(model.cfg["icl"].get("selection_seed", 42))
    print(f"{len(queries)} queries, {len(support)} support utterances", flush=True)

    results, rows = {}, []
    for shots in args.shots:
        scored = []
        for query in queries:
            demos = select_demos(query, support, shots, seed, args.exclude_same_canonical,
                                 args.exclude_same_speaker)
            print(f"  {query['id']}: requested={shots} demos={len(demos)}, running ...", flush=True)
            predicted, latency = model.predict(query, demos, raw_phones=raw_phones, return_latency=True)
            scored.append({"canonical_phones": query["canonical_phones"], "actual_phones": query["actual_phones"],
                           "predicted_phones": predicted})
            rows.append({"id": query["id"], "language": args.language, "shots": shots,
                         "demo_ids": [d["id"] for d in demos], "latency": latency, **scored[-1]})
            print(f"  {query['id']} ({shots}-shot): {' '.join(predicted)}", flush=True)
            print(f"    demos={latency['num_demos']} audios={latency['num_audios']} "
                  f"tokens={latency['prompt_tokens']}+{latency['generated_tokens']} "
                  f"preprocess={latency['preprocess_s']:.3f}s transfer={latency['transfer_s']:.3f}s "
                  f"AuT={latency['aut_s']:.3f}s projector={latency['projector_s']:.3f}s "
                  f"LLM={latency['llm_s']:.3f}s "
                  f"(prefill={latency['llm_prefill_s']:.3f}s decode={latency['llm_decode_s']:.3f}s) "
                  f"generation={latency['generation_s']:.3f}s total={latency['total_s']:.3f}s", flush=True)
        summary = score_unaligned(scored)
        results[f"{args.language}_{shots}shot"] = summary
        print(f"{args.language} {shots}-shot: PER={summary['PER']:.2f} P={summary['Precision']:.2f} "
              f"R={summary['Recall']:.2f} F1={summary['F1']:.2f} DER={summary['DER']:.2f}", flush=True)
    latency_summary = {"model_load_s": model_load_s, "by_shots": {}}
    for shots in args.shots:
        samples = [r["latency"] for r in rows if r["shots"] == shots]
        if samples:
            latency_summary["by_shots"][str(shots)] = {
                "num_queries": len(samples),
                "mean": {k: sum(s[k] for s in samples) / len(samples) for k in samples[0]},
            }
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        (out / "predictions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        (out / "latency.json").write_text(json.dumps(latency_summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}/metrics.json, {out}/predictions.jsonl and {out}/latency.json")


if __name__ == "__main__":
    main()
