"""
time_decay_backtest.py — 시간가중(time-decay) 실험
==================================================
가설(문헌): 오래된 경기를 지수적으로 down-weight하면 팀의 '현재 강도'를 더
잘 반영해 예측이 좋아질 수 있다.
  - Dixon & Coles (1997), JRSS-C: 과거 경기 지수감쇠 가중 도입.
  - Ley, Van de Wiele & Van Eetvelde (2019), Statistical Modelling:
    단일 강도 + 시간가중 모델이 가장 경쟁력 있다고 보고.

방법: prob_model.py와 '완전히 동일한' walk-forward(학습 1990~2023.12,
검증 2024.01~)에서, LogisticRegression에 sample_weight만 추가한다.
  weight_i = 0.5 ** (age_i / halflife),  age = (검증시작일 - 경기일) 일수
반감기(halflife)를 여러 값으로 스윕하고, 균등가중(현행=halflife→∞)과 Brier 비교.

절대 원칙 준수: 시간순 분리 유지, 학습은 경기 전 elo_diff_pre만 사용,
운영 코드/산출물은 건드리지 않음(실험 전용). 채택은 Brier 개선 확인 후에만.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

hist = pd.read_csv('data/elo_history.csv')
comp = hist[hist['tournament'] != 'Friendly'].copy()
comp = comp[comp['date'] >= '1990-01-01'].copy()
comp['date'] = pd.to_datetime(comp['date'])

def brier(probs, actual, order):
    onehot = np.zeros_like(probs)
    for i, a in enumerate(actual):
        onehot[i, order.index(a)] = 1
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))


def split(cut, test_start, test_end=None):
    tr = comp[comp['date'] <= cut].copy()
    te = comp[comp['date'] >= test_start].copy()
    if test_end is not None:
        te = te[te['date'] <= test_end].copy()
    return tr, te


def run(tr, te, halflife_years, test_start):
    age = (pd.Timestamp(test_start) - tr['date']).dt.days.values.astype(float)
    w = None if halflife_years is None else 0.5 ** (age / (halflife_years * 365.25))
    m = LogisticRegression(max_iter=1000)
    m.fit(tr[['elo_diff_pre']].values, tr['outcome'].values, sample_weight=w)
    order = list(m.classes_)
    p = m.predict_proba(te[['elo_diff_pre']].values)
    b = brier(p, te['outcome'].values, order)
    ll = log_loss(te['outcome'].values, p, labels=order)
    acc = np.mean(np.array(order)[p.argmax(1)] == te['outcome'].values)
    eff = float(len(tr)) if w is None else w.sum()
    return b, ll, acc, eff


HLS = [None, 8, 5, 3, 2, 1.5, 1, 0.75, 0.5]

# 두 개의 독립 walk-forward 창에서 시간가중을 교차검증한다.
SPLITS = [
    ('2020~2023', '2019-12-31', '2020-01-01', '2023-12-31'),
    ('2024~현재', '2023-12-31', '2024-01-01', None),
]

results = {}  # split_name -> {hl: brier}
for sname, cut, ts, tend in SPLITS:
    tr, te = split(cut, ts, tend)
    print(f'\n[{sname}]  학습 {len(tr)}경기 · 검증 {len(te)}경기  (시간순)')
    print(f'{"반감기":>10} | {"Brier":>8} | {"LogLoss":>8} | {"적중률":>6} | {"유효표본":>8}')
    print('-' * 58)
    results[sname] = {}
    for hl in HLS:
        b, ll, acc, eff = run(tr, te, hl, ts)
        results[sname][hl] = b
        label = '균등(현행)' if hl is None else f'{hl}년'
        print(f'{label:>10} | {b:8.4f} | {ll:8.4f} | {acc:6.1%} | {eff:8.0f}')

print('\n' + '=' * 58)
print('교차검증 요약 (현행 균등 대비 Brier 변화, 음수=개선):')
for sname in results:
    base = results[sname][None]
    best_hl = min((hl for hl in HLS if hl is not None),
                  key=lambda h: results[sname][h])
    d = results[sname][best_hl] - base
    print(f'  {sname}: 균등 {base:.4f} → 최적 {best_hl}년 {results[sname][best_hl]:.4f}  (Δ {d:+.4f})')

# 두 창 모두에서 개선되는 공통 반감기가 있는지
both = {hl: max(results[s][hl] - results[s][None] for s in results)
        for hl in HLS if hl is not None}
robust = {hl: d for hl, d in both.items() if d < 0}
print('-' * 58)
if robust:
    bh = min(robust, key=robust.get)
    print(f'두 창 모두 개선되는 반감기 존재 → 가장 안정적: {bh}년 '
          f'(두 창 중 최악 Δ {robust[bh]:+.4f})')
    print('판정: 시간가중 일관 개선 → 채택 검토 가치 있음(소폭).')
else:
    print('판정: 두 창 모두 개선되는 반감기 없음 → 노이즈, 현행 유지(기각).')
