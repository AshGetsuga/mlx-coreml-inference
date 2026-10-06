# LSTM Head vs Transformer: Context Drift Experiment

## Motivation

A transformer's decoding limitation is structural: one forward pass produces exactly
one new token, because every new position needs causal self-attention over the full
prefix (including itself) before it can be scored. Speculative decoding methods
(Medusa, EAGLE, blockwise parallel decoding, classic draft-model speculative decoding)
work around this by proposing several candidate future tokens cheaply, then verifying
them with one real forward pass of the full model.

This experiment explores a specific candidate mechanism for the "propose" step: an
LSTM head bolted onto a frozen transformer trunk, used to generate several tokens
autoregressively from a single transformer forward pass, without re-running attention
for each new token.

## Architecture

```
[N tokens] --embed--> [N, 768] --Transformer stack--> [N, 768]  (causal self-attention, contextualized)
                                          |
                                          v
                                [N, 768] --LSTM--> LM head --> [N, VOCAB]
                                          |
                                take argmax at position N --> token_{N+1}
```

To continue generating beyond position N without another transformer pass:

1. Look up `token_{N+1}`'s embedding via the transformer's own embedding table -> `e_{N+1}` (raw, uncontextualized).
2. Append `e_{N+1}` to the LSTM's running sequence and step the LSTM forward.
3. Take the LM head's argmax at the new position -> `token_{N+2}`. Repeat.

This buys `k` extra tokens per transformer forward pass, at the cost of the LSTM
running purely on its own recurrent state for everything past position N.

## Core Problem: Representation Mismatch

Positions `1..N` feed the LSTM **attention-contextualized** vectors `h_i` — each a
mixture of arbitrary earlier tokens via self-attention. Positions `N+1..N+k` feed the
LSTM **raw embeddings** `e_i` — no mixing, no long-range information at all. The
LSTM's fixed-size hidden state has to:

- approximate what attention would have computed, using only its recurrent state, and
- do so with an input type (`e_i`) it may never have seen during training (if trained
  purely teacher-forced on `h_1..h_N`).

The second point matters: if the LSTM never sees raw embeddings as input during
training, then at inference time step `N+1` isn't just "somewhat degraded" — it's
out-of-distribution for the LSTM. That would conflate two distinct failure sources:

1. **Representational limitation** — an LSTM's fixed-size hidden state cannot
   substitute for direct attention over long-range context (the thing we actually want
   to measure).
2. **Training artifact** — the LSTM was never trained to consume its own raw-embedding
   continuations, so it fails for reasons unrelated to (1).

## Fix for (2): Truncated-Context Training

Train the LSTM head with scheduled sampling over the *context boundary*, not just the
usual scheduled sampling over predicted vs. ground-truth tokens:

- Pick a random cut point `i < N` per training example.
- Feed `h_1..h_i` (real transformer output) for the prefix.
- Feed `e_{i+1}..e_N` (raw embeddings, teacher-forced token identities) for the
  suffix — exactly the regime the LSTM will face at inference past the last real
  transformer pass.
- Backprop through the LM head loss as usual.

This makes the LSTM's training distribution match its inference-time usage, so the
drift we measure later is attributable to (1) and not (2).

## Related Work (positioning)

| Method | How it avoids/handles the mismatch |
|---|---|
| **Blockwise Parallel Decoding** (Stern et al., 2018) | Separate FFN heads per offset, predicting directly from `h_N`. No recurrence, no feedback — sidesteps the mismatch entirely. |
| **Medusa** (Cai et al., 2024) | Same idea, modernized: independent heads per offset from `h_N`, tree-based candidate verification. Heads don't condition on each other, so acceptance rate drops for longer drafts, but there's no compounding drift. |
| **EAGLE** (Li et al., 2024) | Chains predictions like an RNN would, but replaces the LSTM with a single causal-attention layer over the growing draft sequence — so new tokens still get *real* (if shallow) attention instead of a raw embedding. This is the closest existing fix to the problem identified here. |
| **This experiment** | Keep the LSTM (pure recurrence, no attention at all past position N) specifically *because* the goal is to characterize how much that recurrence-only bottleneck costs, not to eliminate it. |

## Experiment Design

**Two conditions, same base model:**

- **A — Transformer baseline (ground truth):** re-run the full transformer with real
  causal attention at every generated position. This defines the true conditional
  distribution `P_T(x_{N+k} | x_1 .. x_{N+k-1})`.
- **B — Transformer + LSTM head:** one transformer forward pass over `x_1..x_N`, then
  `k` LSTM-only steps as described above, producing `P_L(x_{N+k} | h_1..h_N, e_{N+1}..e_{N+k-1})`.

**Metric — per-offset KL divergence ("drift curve"):**

```
D(k) = KL( P_T(x_{N+k} | full context) || P_L(x_{N+k} | LSTM continuation) )
```

Compute `D(k)` for `k = 1, 2, 3, ...` and plot against `k`. This directly answers the
original question — how many "free" tokens can the LSTM head generate before its
proposal distribution diverges too far from what the transformer would actually say.

**Secondary metrics** (cheaper to compute, useful for sanity-checking the KL curve):

- Top-1 token match rate between `P_T` and `P_L` at each offset `k`.
- Perplexity of condition B's generated continuation under condition A (i.e., feed B's
  tokens back through the real transformer and score them).

**Expected shape of the result:** `D(k)` should be small at `k=1` (the LSTM's
prediction of `token_{N+1}` is derived from a fully attention-contextualized `h_N`, so
it should be close to the transformer baseline) and grow — likely non-linearly, due to
compounding error in the LSTM's hidden state — as `k` increases. The offset at which
`D(k)` crosses some acceptance threshold is the practical answer to "how many tokens
can this drafting scheme safely propose."

## Open Questions / Next Steps

- Base transformer + tokenizer to use (referenced LLaMA embeddings previously —
  confirm which checkpoint/size).
- Dataset: `Language_Modelling/LSTMvsTransformer/archive` already has a prepared
  Cornell Movie-Dialogues corpus from an earlier LSTM-vs-Transformer exploration —
  worth checking whether it's suitable here, or whether a different corpus is needed.
- LSTM head size/depth — needs to be big enough to be a meaningful drafter but small
  enough that the "cheap draft" premise of speculative decoding still holds.
- Decide whether the LSTM head trains jointly with the transformer or on top of a
  frozen pretrained transformer (frozen trunk is closer to how Medusa/EAGLE operate in
  practice, and isolates the LSTM's limitation more cleanly).
