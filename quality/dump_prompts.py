"""Write the alpha.py prompt set as text files (chat template rendered, leading BOS text removed because
llama.cpp adds BOS itself) for llama-speculative.
  --family llama (default): Llama 3.2 template -> quality/prompts/*.txt
  --family gemma:           Gemma 3 template   -> quality/prompts/gemma/*.txt
usage: uv run --project /Users/sdawka/Code/llming/mlx python /Users/sdawka/Code/llming/quality/dump_prompts.py [--family gemma]"""
import argparse
import os
import sys
sys.path.insert(0, "/Users/sdawka/Code/llming/quality")
from alpha import BENCH, CHAT, ROOT
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--family", default="llama", choices=["llama", "gemma"])
a = ap.parse_args()
if a.family == "gemma":
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/gemma3/mlx-gemma-3-4b-it-4bit")
    out = f"{ROOT}/quality/prompts/gemma"
else:
    tok = AutoTokenizer.from_pretrained(f"{ROOT}/models/mlx-Llama-3.2-3B-Instruct-4bit")
    out = f"{ROOT}/quality/prompts"
os.makedirs(out, exist_ok=True)
# llama.cpp's -f strips one trailing newline (common/arg.cpp). For Gemma, pad so the prompt llama.cpp sees is
# byte-identical to the MLX one (the Llama files predate this and are left as they were measured).
pad = (lambda t: t + "\n" if t.endswith("\n") else t) if a.family == "gemma" else (lambda t: t)
open(f"{out}/raw_bench512.txt", "w").write(pad(BENCH))
for name, msg in CHAT.items():
    s = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, tokenize=False)
    s = s.removeprefix(tok.bos_token)
    open(f"{out}/{name}.txt", "w").write(pad(s))
