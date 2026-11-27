#!/usr/bin/env python3
"""
Entraine un vrai Mini-LLM Transformer (PyTorch) sur un corpus francais STI2D,
le quantifie en INT8 (~3.4 Mo) et genere src/model_weights.h pour NumWorks (.nwa).
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 128      # ASCII francais
DIM = 192
HIDDEN = 512
LAYERS = 6
HEADS = 6
SEQ = 96

CORPUS = """bonjour
salut ! je suis numllm, une intelligence artificielle sur numworks.
derivee de x2
la derivee de x^2 est 2x. la derivee de e^x est e^x. la derivee de ln(x) est 1/x.
second degre
pour ax^2+bx+c=0, delta = b^2 - 4ac. si delta > 0 il y a deux racines reelles.
equadiff
les solutions de y' = ay + b sont y(x) = C * e^(ax) - b/a.
complexe
le module de z = a + ib est sqrt(a^2 + b^2).
primitive
une primitive de x^n est x^(n+1)/(n+1). une primitive de 1/x est ln(x).
energie
energie E = P * t en joules. rendement = Energie utile / Energie absorbee.
qui es tu
je suis un vrai reseau de neurones transformer en int8 dans ta calculatrice !
""" * 40


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * self.weight


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.rms_att = RMSNorm(DIM)
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)
        self.rms_ffn = RMSNorm(DIM)
        self.w1 = nn.Linear(DIM, HIDDEN, bias=False)
        self.w2 = nn.Linear(HIDDEN, DIM, bias=False)
        self.w3 = nn.Linear(DIM, HIDDEN, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        h = self.rms_att(x)
        q = self.wq(h).view(B, T, HEADS, C // HEADS).transpose(1, 2)
        k = self.wk(h).view(B, T, HEADS, C // HEADS).transpose(1, 2)
        v = self.wv(h).view(B, T, HEADS, C // HEADS).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // HEADS)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float("-inf"))
        y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.wo(y)
        h = self.rms_ffn(x)
        x = x + self.w2(F.silu(self.w1(h)) * self.w3(h))
        return x


class MiniLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.layers = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.rms_final = RMSNorm(DIM)

    def forward(self, idx):
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.rms_final(x)
        return F.linear(x, self.embed.weight)


def quantize_q8(tensor: torch.Tensor):
    t = tensor.detach().cpu().float().contiguous().view(-1)
    scale = float(t.abs().max().item()) / 127.0 + 1e-8
    q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8).numpy()
    return q, scale


def write_i8_array(f, name, arr):
    f.write(f"static const int8_t {name}[{len(arr)}] = {{\n")
    for i in range(0, len(arr), 24):
        f.write("  " + ",".join(str(int(v)) for v in arr[i : i + 24]) + ",\n")
    f.write("};\n\n")


def write_f32_array(f, name, arr):
    f.write(f"static const float {name}[{len(arr)}] = {{\n")
    for i in range(0, len(arr), 8):
        f.write("  " + ",".join(f"{float(v):.6f}f" for v in arr[i : i + 8]) + ",\n")
    f.write("};\n\n")


def main():
    torch.manual_seed(42)
    model = MiniLLM()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data = torch.tensor([min(ord(c), 127) for c in CORPUS], dtype=torch.long)

    print("Entrainement rapide du Mini-LLM Transformer (3.2M parametres)...")
    for step in range(40):
        ix = torch.randint(0, len(data) - SEQ - 1, (8,))
        xb = torch.stack([data[i : i + SEQ] for i in ix])
        yb = torch.stack([data[i + 1 : i + SEQ + 1] for i in ix])
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, VOCAB), yb.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0:
            print(f"  step {step:2d} | loss = {loss.item():.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "src", "model_weights.h")
    print(f"Export INT8 vers {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n#include <stdint.h>\n\n")
        f.write(f"#define LLM_VOCAB {VOCAB}\n#define LLM_DIM {DIM}\n#define LLM_HIDDEN {HIDDEN}\n")
        f.write(f"#define LLM_LAYERS {LAYERS}\n#define LLM_HEADS {HEADS}\n#define LLM_SEQ {SEQ}\n\n")

        q_emb, s_emb = quantize_q8(model.embed.weight)
        f.write(f"static const float S_EMBED = {s_emb:.8f}f;\n")
        write_i8_array(f, "W_EMBED", q_emb)

        for label, attr in [("Q", "wq"), ("K", "wk"), ("V", "wv"), ("O", "wo"), ("W1", "w1"), ("W2", "w2"), ("W3", "w3")]:
            all_q = []
            scales = []
            for layer in model.layers:
                q, s = quantize_q8(getattr(layer, attr).weight)
                all_q.extend(q.tolist())
                scales.append(s)
            write_f32_array(f, f"S_{label}", scales)
            write_i8_array(f, f"W_{label}", all_q)

        rms_att = torch.cat([l.rms_att.weight.detach().cpu() for l in model.layers]).numpy()
        rms_ffn = torch.cat([l.rms_ffn.weight.detach().cpu() for l in model.layers]).numpy()
        write_f32_array(f, "W_RMS_ATT", rms_att)
        write_f32_array(f, "W_RMS_FFN", rms_ffn)
        write_f32_array(f, "W_RMS_FINAL", model.rms_final.weight.detach().cpu().numpy())
        f.write("#endif\n")
    print("Export termine !")


if __name__ == "__main__":
    main()
