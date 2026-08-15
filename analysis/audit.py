"""
Reproduces defect_report.md findings S1-06, S1-07, and S2-04:
  1. length-channel test (does query length predict the label?)
  2. TF-IDF bag-of-words floor
  3. correlation between activation score and query length
  4. honest val/test threshold protocol (200 random splits), vs the
     reported test-set-optimized accuracy

Run from the repo root:
    python3 analysis/audit.py

Requires: data/vectors/all_layers_proper.pkl, dataset_2000.pkl,
train_test_split.pkl, llama_layer_results.pkl (or pass --data-dir).
"""
import argparse
import numpy as np
from inspect_pkl import StubUnpickler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pointbiserialr

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument('--data-dir', default='data/vectors/',
                  help='directory containing the .pkl artifacts (default: data/vectors/, '
                       'i.e. run this from the repo root)')
U = _ap.parse_args().data_dir
if not U.endswith('/'):
    U += '/'

def load(f):
    with open(U + f, 'rb') as fh:
        return StubUnpickler(fh).load()

proper = load('all_layers_proper.pkl')
g0 = proper['gpt2'][0]['results']
queries = g0['queries']
y = np.array(g0['true_labels'])
print(f"test set: n={len(y)}  positive rate={y.mean():.3f}  (1 = dangerous)")

# ---------- 1. LENGTH CHANNEL ----------
char_len = np.array([len(q) for q in queries], float)
word_len = np.array([len(q.split()) for q in queries], float)
X_len = np.c_[char_len, word_len]

print("\n" + "=" * 62)
print("1. LENGTH CHANNEL  (no model, no activations, just string length)")
print("=" * 62)
print(f"  mean chars  safe={char_len[y==0].mean():6.1f}   dangerous={char_len[y==1].mean():6.1f}")
print(f"  mean words  safe={word_len[y==0].mean():6.1f}   dangerous={word_len[y==1].mean():6.1f}")
r, p = pointbiserialr(y, char_len)
print(f"  point-biserial r(label, char_len) = {r:+.3f}   p = {p:.2e}")

cv = StratifiedKFold(5, shuffle=True, random_state=0)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
for name, X in [('char_len only', char_len[:, None]), ('word_len only', word_len[:, None]), ('both', X_len)]:
    s = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print(f"  LogReg on {name:14s} -> {s.mean()*100:5.2f}% (+/- {s.std()*100:.2f})")

# ---------- 2. TF-IDF FLOOR ----------
print("\n" + "=" * 62)
print("2. TF-IDF FLOOR  (bag of words, no model)")
print("=" * 62)
tfidf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                      LogisticRegression(max_iter=2000))
s = cross_val_score(tfidf, queries, y, cv=cv, scoring='accuracy')
print(f"  5-fold CV on the same 1800 test items -> {s.mean()*100:5.2f}% (+/- {s.std()*100:.2f})")

split = load('train_test_split.pkl')
tr_q = split['train']['safe'] + split['train']['dangerous']
tr_y = np.r_[np.zeros(len(split['train']['safe'])), np.ones(len(split['train']['dangerous']))]
tfidf.fit(tr_q, tr_y)
print(f"  trained on the SAME 200 train items -> {(tfidf.predict(queries) == y).mean()*100:5.2f}% on the 1800")

# ---------- 3. DOES THE ACTIVATION SCORE ENCODE LENGTH? ----------
print("\n" + "=" * 62)
print("3. CORRELATION OF ACTIVATION SCORE WITH RAW STRING LENGTH")
print("=" * 62)
for layer in [0, 1, 6, 12, 18, 23]:
    sc = np.array(proper['gpt2'][layer]['results']['scores'])
    rl = np.corrcoef(sc, char_len)[0, 1]
    ry = np.corrcoef(sc, y)[0, 1]
    print(f"  gpt2 L{layer:<2d}  r(score, char_len) = {rl:+.3f}    r(score, label) = {ry:+.3f}")

# ---------- 4. HONEST THRESHOLD PROTOCOL ----------
print("\n" + "=" * 62)
print("4. THRESHOLD ON VAL, SCORED ON HELD-OUT TEST  (200 random splits)")
print("=" * 62)

def sweep_best(scores, labels):
    order = np.argsort(scores)
    s_sorted, l_sorted = scores[order], labels[order]
    cum = np.cumsum(l_sorted)
    n, tot = len(labels), l_sorted.sum()
    k = np.arange(n + 1)
    below_pos = np.r_[0, cum]
    acc = (below_pos + (n - k) - (tot - below_pos)) / n
    i = int(np.argmax(acc))
    thr = s_sorted[i - 1] if 0 < i < n else (s_sorted[0] - 1 if i == 0 else s_sorted[-1] + 1)
    return thr, acc[i]

def honest(scores, labels, n_rep=200, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_rep):
        idx = rng.permutation(len(labels))
        v, t = idx[:len(idx) // 2], idx[len(idx) // 2:]
        thr, _ = sweep_best(scores[v], labels[v])
        out.append(((scores[t] <= thr).astype(int) == labels[t]).mean())
    return np.mean(out), np.std(out)

print(f"  {'layer':>5} {'reported':>9} {'default t=0':>12} {'honest val/test':>18}")
rows = []
for layer in sorted(proper['gpt2'].keys()):
    d = proper['gpt2'][layer]
    sc = np.array(d['results']['scores'])
    m, sd = honest(sc, y)
    rows.append((layer, d['accuracy_optimized'], d['accuracy_default'], m, sd))
for layer, rep, dflt, m, sd in rows:
    star = '  <-- canonical' if layer == 0 else ''
    print(f"  {layer:>5} {rep*100:8.2f}% {dflt*100:11.2f}% {m*100:12.2f}% +/-{sd*100:4.2f}{star}")

best = max(rows, key=lambda r: r[3])
print(f"\n  best layer under honest protocol: L{best[0]} at {best[3]*100:.2f}%")

print("\n  --- llama ---")
ll = load('llama_layer_results.pkl')
for layer in sorted(ll.keys()):
    d = ll[layer]
    sc = np.array(d['results']['scores'])
    m, sd = honest(sc, y)
    print(f"  {layer:>5} {d['accuracy_optimized']*100:8.2f}% {d['accuracy_default']*100:11.2f}% {m*100:12.2f}% +/-{sd*100:4.2f}")
