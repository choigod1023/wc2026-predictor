"""
score_decay_backtest.py — 스코어 모델(Dixon-Coles) 시간가중 백테스트
====================================================================
prob_model과 별개 모델이므로 별도 검증이 필요(백테스트 우선 원칙).
score_model.py의 Dixon-Coles를 그대로 재현하되, Poisson GLM과 ρ 추정에
sample_weight=0.5^(age/halflife)만 추가. 두 독립 walk-forward 창에서
스코어 로그우도(↑)와 O/U 2.5 Brier(↓)를 균등가중과 비교.
"""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

MAXG = 10
_LOGFACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 100)))])


def pois_pmf(k, lam):
    k = np.asarray(k)
    lam = np.asarray(lam, dtype=float)
    return np.exp(k * np.log(lam) - lam - _LOGFACT[k])


def fit_dc(d, hs, as_, w):
    """가중 Dixon-Coles 적합. 반환: (mh, ma, rho)."""
    dd = d.reshape(-1, 1) / 100.0
    mh = PoissonRegressor(alpha=1e-6, max_iter=500).fit(dd, hs, sample_weight=w)
    ma = PoissonRegressor(alpha=1e-6, max_iter=500).fit(dd, as_, sample_weight=w)
    lh, la = mh.predict(dd), ma.predict(dd)
    ww = np.ones_like(hs, dtype=float) if w is None else w
    best = (-np.inf, 0.0)
    for rho in np.linspace(-0.2, 0.2, 81):
        t = _tau(hs, as_, lh, la, rho)
        p = pois_pmf(hs, lh) * pois_pmf(as_, la) * t
        ll = np.sum(ww * np.log(np.clip(p, 1e-12, None)))
        if ll > best[0]:
            best = (ll, rho)
    return mh, ma, best[1]


def _tau(x, y, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return np.clip(t, 1e-6, None)


def grid(lh, la, rho):
    x = pois_pmf(np.arange(MAXG)[:, None], lh)
    y = pois_pmf(np.arange(MAXG)[None, :], la)
    g = x * y
    g[0, 0] *= (1 - lh * la * rho)
    g[0, 1] *= (1 + lh * rho)
    g[1, 0] *= (1 + la * rho)
    g[1, 1] *= (1 - rho)
    g = np.clip(g, 0, None)
    return g / g.sum()


def evaluate(mh, ma, rho, te):
    dd = te['elo_diff_pre'].values.reshape(-1, 1) / 100.0
    lh, la = mh.predict(dd), ma.predict(dd)
    hs, as_ = te['home_score'].values, te['away_score'].values
    idx = np.add.outer(np.arange(MAXG), np.arange(MAXG))
    logp, ou_pred, ou_true = [], [], []
    for i in range(len(te)):
        g = grid(lh[i], la[i], rho)
        logp.append(np.log(max(g[min(hs[i], MAXG-1), min(as_[i], MAXG-1)], 1e-12)))
        ou_pred.append(g[idx > 2.5].sum())
        ou_true.append(1.0 if hs[i] + as_[i] > 2.5 else 0.0)
    ou_pred, ou_true = np.array(ou_pred), np.array(ou_true)
    return float(np.mean(logp)), float(np.mean((ou_pred - ou_true) ** 2))


# 데이터 (score_model.py와 동일 구성)
hist = pd.read_csv('data/elo_history.csv')
res = pd.read_csv('data/results.csv')
res = res[res['home_score'].notna()].copy()
res['home_score'] = res['home_score'].astype(int)
res['away_score'] = res['away_score'].astype(int)
df = hist.merge(res[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
                on=['date', 'home_team', 'away_team'], how='inner')
df = df[(df['tournament'] != 'Friendly') & (df['date'] >= '1990-01-01')].copy()
df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
df['date'] = pd.to_datetime(df['date'])

HLS = [None, 5, 3, 2, 1.5, 1]
SPLITS = [
    ('2020~2023', '2019-12-31', '2020-01-01', '2023-12-31'),
    ('2024~현재', '2023-12-31', '2024-01-01', None),
]

agg = {}
for sname, cut, ts, tend in SPLITS:
    tr = df[df['date'] <= cut]
    te = df[df['date'] >= ts]
    if tend:
        te = te[te['date'] <= tend]
    d, hs, as_ = tr['elo_diff_pre'].values, tr['home_score'].values, tr['away_score'].values
    age = (pd.Timestamp(ts) - tr['date']).dt.days.values.astype(float)
    print(f'\n[{sname}]  학습 {len(tr)} · 검증 {len(te)}')
    print(f'{"반감기":>10} | {"logLik↑":>9} | {"O/U2.5Brier↓":>12}')
    print('-' * 40)
    agg[sname] = {}
    for hl in HLS:
        w = None if hl is None else 0.5 ** (age / (hl * 365.25))
        mh, ma, rho = fit_dc(d, hs, as_, w)
        ll, oub = evaluate(mh, ma, rho, te)
        agg[sname][hl] = (ll, oub)
        label = '균등(현행)' if hl is None else f'{hl}년'
        print(f'{label:>10} | {ll:+9.4f} | {oub:12.4f}')

print('\n' + '=' * 52)
print('교차검증 (현행 균등 대비, +logLik/-Brier=개선):')
for s in agg:
    bll, boub = agg[s][None]
    # 두 지표 동시 개선되는 반감기
    cands = [hl for hl in HLS if hl and agg[s][hl][0] > bll and agg[s][hl][1] < boub]
    print(f'  {s}: 균등 logLik {bll:+.4f}/Brier {boub:.4f} · '
          f'동시개선 반감기 {cands or "없음"}')
both = [hl for hl in HLS if hl and all(
    agg[s][hl][0] > agg[s][None][0] and agg[s][hl][1] < agg[s][None][1] for s in agg)]
print('-' * 52)
print(f'판정: 두 창 모두 logLik·Brier 동시 개선 반감기 = {both or "없음"}')
print(' → 있으면 스코어 모델도 시간가중 채택 검토, 없으면 현행 유지(기각).')
