"""
A Transformer built from scratch in PyTorch ("Attention Is All You Need", 2017),
trained on a toy copy task: the model reads a random sequence of digits and must
output the same sequence.

Parts, in build order:
  1. Scaled dot-product attention
  2. Multi-head attention
  3. Sinusoidal positional encoding
  4. Feed-forward block + encoder/decoder layers
  5. The full encoder-decoder Transformer
  6. Copy-task data, training loop, greedy decoding
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. SCALED DOT-PRODUCT ATTENTION
# ---------------------------------------------------------------------------
# Every token makes a Query ("what am I looking for?"), a Key ("what do I
# contain?") and a Value ("what do I hand over if someone attends to me?").
# Attention scores = Q.K^T. We divide by sqrt(d_k) because dot products grow
# with dimension, which would push softmax into saturated regions with tiny
# gradients. Softmax turns scores into weights; the output is a weighted sum
# of Values. A mask sets forbidden positions to -inf so softmax gives them 0.
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)   # (B, H, Tq, Tk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights                          # (B, H, Tq, d_k)


# ---------------------------------------------------------------------------
# 2. MULTI-HEAD ATTENTION
# ---------------------------------------------------------------------------
# One attention pattern is limiting, so we run H of them in parallel, each in
# its own lower-dimensional subspace (d_k = d_model / H). Different heads can
# specialise (e.g. "attend to previous token", "attend to same position").
# Implementation trick: one big linear layer produces all heads at once, then
# we reshape (B, T, d_model) -> (B, H, T, d_k). Outputs are concatenated and
# mixed by a final linear layer.
# The same module does self-attention (q, k, v from the same sequence) and
# cross-attention (q from decoder, k/v from encoder).
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_k = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split(self, x):                                  # (B,T,D) -> (B,H,T,d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None):
        q, k, v = self.split(self.w_q(x_q)), self.split(self.w_k(x_kv)), self.split(self.w_v(x_kv))
        out, self.last_weights = scaled_dot_product_attention(q, k, v, mask)
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, -1)   # concat heads
        return self.w_o(out)


# ---------------------------------------------------------------------------
# 3. POSITIONAL ENCODING
# ---------------------------------------------------------------------------
# Attention is permutation-invariant: it has no idea about word order. So we
# add a position-dependent vector to each embedding. The paper uses sines and
# cosines of geometrically spaced frequencies:
#   PE[pos, 2i]   = sin(pos / 10000^(2i/d))
#   PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
# Each position gets a unique pattern, and relative offsets correspond to
# linear transformations of it, which makes relative position easy to learn.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)                   # saved, but not trained

    def forward(self, x):                                # x: (B, T, D)
        return x + self.pe[: x.size(1)]


# ---------------------------------------------------------------------------
# 4. FEED-FORWARD + LAYERS
# ---------------------------------------------------------------------------
# Attention moves information BETWEEN positions; the feed-forward network
# processes each position independently (two linear layers with a ReLU).
# Each sub-layer is wrapped in a residual connection + LayerNorm, which keeps
# gradients flowing and training stable. (Post-norm, as in the original paper.)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.n1(x + self.drop(self.attn(x, x, src_mask)))   # self-attention
        return self.n2(x + self.drop(self.ff(x)))


# The decoder layer has THREE sub-layers:
#  (a) masked self-attention: position t may only see positions <= t, so the
#      model can't cheat by peeking at the answer it is supposed to predict;
#  (b) cross-attention: queries from the decoder, keys/values from the encoder
#      output. This is how the decoder "looks at" the input sequence;
#  (c) feed-forward.
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, src_mask, tgt_mask):
        x = self.n1(x + self.drop(self.self_attn(x, x, tgt_mask)))
        x = self.n2(x + self.drop(self.cross_attn(x, enc, src_mask)))
        return self.n3(x + self.drop(self.ff(x)))


# ---------------------------------------------------------------------------
# 5. THE FULL TRANSFORMER
# ---------------------------------------------------------------------------
# tokens -> embedding (scaled by sqrt(d_model)) -> + positions -> N layers.
# A final linear layer maps decoder states to vocabulary logits.
class Transformer(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=4, d_ff=128, n_layers=2, dropout=0.1, pad=0):
        super().__init__()
        self.pad, self.d_model = pad, d_model
        self.src_emb = nn.Embedding(vocab, d_model)
        self.tgt_emb = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.enc = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.dec = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.out = nn.Linear(d_model, vocab)

    def encode(self, src):
        src_mask = (src != self.pad)[:, None, None, :]           # hide padding
        x = self.drop(self.pos(self.src_emb(src) * math.sqrt(self.d_model)))
        for layer in self.enc:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt, enc, src_mask):
        T = tgt.size(1)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=tgt.device))  # lower-triangular
        x = self.drop(self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model)))
        for layer in self.dec:
            x = layer(x, enc, src_mask, causal)
        return self.out(x)

    def forward(self, src, tgt):
        enc, src_mask = self.encode(src)
        return self.decode(tgt, enc, src_mask)

    @torch.no_grad()
    def greedy_decode(self, src, max_len, bos=0):
        # Generate one token at a time, feeding each prediction back in.
        enc, src_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), bos, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(ys, enc, src_mask)[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
        return ys[:, 1:]


# ---------------------------------------------------------------------------
# 6. TOY COPY TASK + TRAINING
# ---------------------------------------------------------------------------
# Token 0 is reserved for padding/BOS; real tokens are 1..9.
# decoder input  = [BOS, s1, s2, ..., sN]   (shifted right)
# decoder target = [s1,  s2, ..., sN]       (teacher forcing)
VOCAB, SEQ_LEN = 10, 8


def make_batch(batch_size):
    src = torch.randint(1, VOCAB, (batch_size, SEQ_LEN))
    bos = torch.zeros(batch_size, 1, dtype=torch.long)
    return src, torch.cat([bos, src[:, :-1]], dim=1), src


def sanity_checks():
    # (1) our attention should match PyTorch's built-in implementation
    q, k, v = (torch.randn(2, 4, 5, 16) for _ in range(3))
    ours, _ = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(ours, F.scaled_dot_product_attention(q, k, v), atol=1e-5)
    # (2) causal masking: changing a FUTURE token must not change earlier outputs
    m = Transformer(VOCAB).eval()
    src = torch.randint(1, VOCAB, (1, SEQ_LEN))
    t1 = torch.randint(1, VOCAB, (1, SEQ_LEN)); t2 = t1.clone(); t2[0, -1] = (t1[0, -1] % 9) + 1
    assert torch.allclose(m(src, t1)[:, :-1], m(src, t2)[:, :-1], atol=1e-5)
    print("sanity checks passed: attention matches F.scaled_dot_product_attention, causal mask leaks nothing\n")


if __name__ == "__main__":
    sanity_checks()
    model = Transformer(VOCAB)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    # Warmup then cosine decay: Transformers are touchy early on, so the learning
    # rate ramps up over the first steps (as in the original paper), then decays.
    STEPS, WARMUP = 4000, 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / WARMUP, 0.5 * (1 + math.cos(math.pi * s / STEPS))))

    for step in range(1, STEPS + 1):
        model.train()
        src, tgt_in, tgt_out = make_batch(64)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt_out.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 400 == 0 or step == 1:
            model.eval()
            src, _, tgt = make_batch(500)                # fresh, never-seen sequences
            pred = model.greedy_decode(src, SEQ_LEN)
            acc = (pred == tgt).all(dim=1).float().mean().item()
            print(f"step {step:4d} | loss {loss.item():.4f} | exact-copy accuracy (greedy) {acc:6.1%}")

    print("\nExamples on unseen sequences:")
    model.eval()
    src, _, tgt = make_batch(5)
    pred = model.greedy_decode(src, SEQ_LEN)
    for s, p in zip(src, pred):
        print(f"  input {s.tolist()}  ->  output {p.tolist()}  {'OK' if torch.equal(s, p) else 'WRONG'}")

    # Peek inside: which source position does each output step attend to?
    cross = model.dec[-1].cross_attn.last_weights        # (B, H, T, T) from the last decode pass
    print("\nCross-attention of last decoder layer (head-averaged, example 0).")
    print("Row = output step, argmax = source position attended to:")
    print("  ", cross[0].mean(0).argmax(-1).tolist(), "(a diagonal 0..7 means it learned to copy position-by-position)")
