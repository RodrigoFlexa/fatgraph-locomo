import json, re, string, itertools
from collections import Counter
from nltk.stem import PorterStemmer; st = PorterStemmer().stem

def norm(s):
    s = (s or "").replace(",", "")
    s = re.sub(r"\b(a|an|the|and)\b", " ", s.lower())
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())

def f1(p, g):
    pt = [st(w) for w in norm(p).split()]; gt = [st(w) for w in norm(g).split()]
    c = Counter(pt) & Counter(gt); n = sum(c.values())
    if not n: return 0.0
    pr, rc = n/len(pt), n/len(gt)
    return 2*pr*rc/(pr+rc)

def best_window(p, g):                      # teto de "só cortar o excesso"
    t = p.split()
    return max((f1(" ".join(t[i:j]), g)
                for i in range(len(t)) for j in range(i+1, len(t)+1)), default=0.0)

rows = [json.loads(l) for l in open("results/L2-slots/predictions.jsonl")]
for cat in (4, 1):
    sub = [r for r in rows if r["category"] == cat and r["recall"]["recall_context"] >= 0.999]
    cur = sum(r["f1"] for r in sub)/len(sub)
    top = sum(best_window(r["prediction"], r["gold"]) for r in sub)/len(sub)
    zero = sum(1 for r in sub if best_window(r["prediction"], r["gold"]) < 0.1)
    print(f"cat={cat} n={len(sub)}  F1 atual={cur:.3f}  teto só-formato={top:.3f}  "
          f"irrecuperáveis={zero} ({zero/len(sub):.0%})")