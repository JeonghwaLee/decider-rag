"""Export a q+D+k decider for vLLM pooling serving: the LoRA is merged into the Qwen3 backbone and written as a
Qwen3ForCausalLM checkpoint (lm_head unchanged, unused), so `vllm serve --runner pooling --convert embed` with LAST pooling and no
normalisation returns the backbone's last-token hidden state (after the final norm, as Router.forward). The MLP head is saved as
head.pt and applied by the client (vllm_decider.py). Usage: export_decider_vllm.py --ckpt <decider dir> --out <dir>"""

import argparse, os, sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from e2e_ent2 import load_decider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dec = load_decider(a.ckpt)
    enc = dec.enc.merge_and_unload()
    sd = {k: v.to(torch.bfloat16) for k, v in enc.state_dict().items()}
    lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16)
    missing, unexpected = lm.model.load_state_dict(sd, strict=True), None
    os.makedirs(a.out, exist_ok=True)
    lm.save_pretrained(a.out)
    AutoTokenizer.from_pretrained("Qwen/Qwen3-4B").save_pretrained(a.out)
    torch.save({k: v.float().cpu() for k, v in dec.head.state_dict().items()}, f"{a.out}/head.pt")
    print("saved", a.out, list(dec.head.state_dict()))


if __name__ == "__main__":
    main()
