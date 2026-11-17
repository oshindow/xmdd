"""Constrained decoding to a fixed phone set."""
from __future__ import annotations

import json


def read_symbols(source):
    """Phone symbols from a list, a .txt (one per line) or a JSONL manifest (canonical + actual phones)."""
    if not isinstance(source, str):
        return sorted(set(source))
    with open(source, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if source.endswith(".txt"):
        return sorted(set(lines))
    return sorted({p for line in lines for row in [json.loads(line)]
                   for p in row["canonical_phones"] + row["actual_phones"]})


def make_phone_constraint(tokenizer, source):
    """Returns f(prompt_length) -> the prefix_allowed_tokens_fn of generate().

    Before <asr_text> (the language prefix) decoding is free; after it the output must be a
    sequence of symbols: the first in its bare BPE form, later ones space-prefixed, then the end
    of the turn. A symbol may span several BPE tokens: the allowed tokens come from a token trie,
    and a new symbol (or the end) may only start once the current one is complete.
    """
    symbols = read_symbols(source)

    def trie(texts):
        root = {}
        for text in texts:
            node = root
            for token in tokenizer.encode(text, add_special_tokens=False):
                node = node.setdefault(token, {})
            node[None] = True  # a symbol ends here
        return root
    bare, spaced = trie(symbols), trie(" " + s for s in symbols)
    asr_text = tokenizer.convert_tokens_to_ids("<asr_text>")
    stops = [t for t in {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")} if t is not None]
    everything = list(range(len(tokenizer)))
    print(f"phone-constrained decoding: {len(symbols)} symbols", flush=True)

    def allowed_after(tokens):
        node = bare
        for token in tokens:
            if token in node:
                node = node[token]
            elif node.get(None) and token in spaced:
                node = spaced[token]
            else:  # cannot happen under the constraint; restart at a symbol boundary
                node = {None: True}
        allowed = [t for t in node if t is not None]
        if node.get(None):
            allowed += list(spaced) + stops
        return allowed

    def factory(prompt_length):
        def allowed(_batch_id, ids):
            new = ids[prompt_length:].tolist()
            if asr_text not in new:
                return everything if len(new) < 8 else [asr_text]
            after = len(new) - 1 - new[::-1].index(asr_text)
            return allowed_after(new[after + 1:])
        return allowed
    return factory
