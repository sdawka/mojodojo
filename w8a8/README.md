# w8a8

Wikitext-2 perplexity of Llama 3.2 3B in MAX with W8A8 (int8 weights and activations) on every projection, using the same custom architecture as `fpga/maxop` and the llama-perplexity method from `quality/` (ctx 512, BOS per chunk, second half of each chunk scored, 144,384 tokens, all prefill).

| Run | ppl | vs bf16 |
| --- | --- | --- |
| bf16 | 10.3189 ± 0.0736 | — |
| W8A8 int8 (all projections) | 10.3883 ± 0.0741 | +0.67% |
| W8A8 fake-quant (all projections) | 10.3899 ± 0.0741 | +0.69% |

Raw per-token NLL is in `results/ppl_raw/*.nll.npy`; logs in `logs/`. Timings are not authoritative.

```sh
cd fpga/maxop
W8A8_MODE=int8 W8A8_ROUTE=all:all pixi run python ../../w8a8/ppl_max.py --name llama3b-max-w8a8-int8-all
```
