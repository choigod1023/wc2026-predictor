"""
score_model.py — 스코어(득점) 예측 모델 → 언더/오버 · 핸디캡
================================================================
승/무/패만으로는 언더오버(총 득점)와 핸디캡(득점차)을 줄 수 없다.
그래서 '몇 대 몇'의 분포를 모델링한다. 두 모델을 등록해 비교한다.

  M1 독립 포아송(Independent Poisson)
     log λ_home = a0 + a1·d ,  log λ_away = b0 + b1·d   (d = elo_diff_pre)
     홈/원정 득점이 서로 독립인 포아송이라 가정. P(x,y)=Pois(x;λh)·Pois(y;λa)
  M2 Dixon-Coles
     같은 λ에 저점수(0:0,1:0,0:1,1:1) 의존성 보정 τ(x,y;ρ) 를 곱한다.
     축구 실측에서 0:0·1:1 이 독립가정보다 자주 나오는 현상을 교정.

검증(walk-forward, 학습 1990~2023 / 검증 2024~2026.6, 친선 제외):
  - 스코어 로그우도(실제 스코어라인에 모델이 부여한 확률의 평균 log; 높을수록 좋음)
  - 언더오버 2.5 Brier(이항; 총득점>2.5 예측의 정확도)
기준선: 학습기간 평균 득점(λ 고정) 포아송.

산출(JSON): 72경기 각각의 기대득점·Top스코어·O/U·핸디캡·근거설명.
데이터 누수 방지: 피처는 경기 전 elo_diff_pre 만 사용.
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

MAXG = 10          # 스코어 그리드 0..9
HOME_ADV = 100
OU_LINES = [1.5, 2.5, 3.5]
HCAP_LINES = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]  # 홈 관점 (마이너스=홈이 접어줌)

# ── 데이터: elo_diff_pre + 실제 스코어 결합 ───────────────────
hist = pd.read_csv('data/elo_history.csv')
res = pd.read_csv('data/results.csv')
res = res[res['home_score'].notna()].copy()
res['home_score'] = res['home_score'].astype(int)
res['away_score'] = res['away_score'].astype(int)
df = hist.merge(res[['date', 'home_team', 'away_team', 'home_score', 'away_score']],
                on=['date', 'home_team', 'away_team'], how='inner')
df = df[(df['tournament'] != 'Friendly') & (df['date'] >= '1990-01-01')].copy()
df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'])

train = df[df['date'] <= '2023-12-31']
test = df[df['date'] >= '2024-01-01']


# log(k!) 룩업 (학습 스코어에 두 자릿수 대승도 있어 넉넉히 0..99)
_LOGFACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 100)))])


def pois_pmf(k, lam):
    """log 안정 포아송 PMF. k는 정수(스칼라/배열), lam은 양수."""
    k = np.asarray(k)
    lam = np.asarray(lam, dtype=float)
    return np.exp(k * np.log(lam) - lam - _LOGFACT[k])


class IndepPoisson:
    name = '독립 포아송'
    short = 'poisson'

    def fit(self, d, hs, as_):
        d = d.reshape(-1, 1) / 100.0      # 스케일
        self.mh = PoissonRegressor(alpha=1e-6, max_iter=500).fit(d, hs)
        self.ma = PoissonRegressor(alpha=1e-6, max_iter=500).fit(d, as_)
        return self

    def lams(self, d):
        d = np.atleast_1d(d).reshape(-1, 1) / 100.0
        return self.mh.predict(d), self.ma.predict(d)

    def grid(self, lh, la):
        x = pois_pmf(np.arange(MAXG)[:, None], lh)      # (MAXG,1)
        y = pois_pmf(np.arange(MAXG)[None, :], la)      # (1,MAXG)
        g = x * y
        return g / g.sum()


class DixonColes(IndepPoisson):
    name = 'Dixon-Coles'
    short = 'dixon_coles'

    def fit(self, d, hs, as_):
        super().fit(d, hs, as_)
        lh, la = self.lams(d)
        best = (-np.inf, 0.0)
        for rho in np.linspace(-0.2, 0.2, 81):
            ll = self._loglik(hs, as_, lh, la, rho)
            if ll > best[0]:
                best = (ll, rho)
        self.rho = best[1]
        return self

    @staticmethod
    def _tau(x, y, lh, la, rho):
        t = np.ones_like(lh, dtype=float)
        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        t = np.where(m00, 1 - lh * la * rho, t)
        t = np.where(m01, 1 + lh * rho, t)
        t = np.where(m10, 1 + la * rho, t)
        t = np.where(m11, 1 - rho, t)
        return np.clip(t, 1e-6, None)

    def _loglik(self, hs, as_, lh, la, rho):
        p = pois_pmf(hs, lh) * pois_pmf(as_, la) * self._tau(hs, as_, lh, la, rho)
        return np.sum(np.log(np.clip(p, 1e-12, None)))

    def grid(self, lh, la):
        g = super().grid(lh, la)
        # τ 보정 (저점수 칸만)
        rho = self.rho
        g = g.copy()
        g[0, 0] *= (1 - lh * la * rho)
        g[0, 1] *= (1 + lh * rho)
        g[1, 0] *= (1 + la * rho)
        g[1, 1] *= (1 - rho)
        g = np.clip(g, 0, None)
        return g / g.sum()


class FlatPoisson(IndepPoisson):
    """기준선: 학습기간 평균 득점으로 λ 고정 (피처 무시)."""
    name = '기준선·평균득점'
    short = 'flat'

    def fit(self, d, hs, as_):
        self.lh0, self.la0 = float(np.mean(hs)), float(np.mean(as_))
        return self

    def lams(self, d):
        n = np.atleast_1d(d).shape[0]
        return np.full(n, self.lh0), np.full(n, self.la0)


# ── 검증 ──────────────────────────────────────────────────────
def evaluate(model):
    lh, la = model.lams(test['elo_diff_pre'].values)
    hs, as_ = test['home_score'].values, test['away_score'].values
    # 스코어 로그우도(실제 칸 확률)
    logp = []
    ou_pred, ou_true = [], []
    for i in range(len(test)):
        g = model.grid(lh[i], la[i])
        xi = min(hs[i], MAXG - 1)
        yi = min(as_[i], MAXG - 1)
        logp.append(np.log(max(g[xi, yi], 1e-12)))
        # O/U 2.5
        idx = np.add.outer(np.arange(MAXG), np.arange(MAXG))
        ou_pred.append(g[idx > 2.5].sum())
        ou_true.append(1.0 if (hs[i] + as_[i]) > 2.5 else 0.0)
    ou_pred = np.array(ou_pred); ou_true = np.array(ou_true)
    return {
        'name': model.name, 'short': model.short,
        'score_loglik': round(float(np.mean(logp)), 4),
        'ou25_brier': round(float(np.mean((ou_pred - ou_true) ** 2)), 4),
    }


print(f'학습 {len(train)} / 검증 {len(test)} 경기\n')
dtr = train['elo_diff_pre'].values
hs_tr, as_tr = train['home_score'].values, train['away_score'].values

leaderboard = []
fitted = {}
for cls in [IndepPoisson, DixonColes, FlatPoisson]:
    m = cls().fit(dtr, hs_tr, as_tr)
    fitted[cls.short] = m
    row = evaluate(m)
    leaderboard.append(row)
    extra = f"(ρ={m.rho:+.3f})" if hasattr(m, 'rho') else ''
    print(f"{row['name']:14s} logLik {row['score_loglik']:+.4f}  "
          f"O/U2.5 Brier {row['ou25_brier']:.4f}  {extra}")

# 스코어 로그우도 높은 순 정렬
leaderboard.sort(key=lambda r: -r['score_loglik'])
leaderboard[0]['best'] = True
json.dump(leaderboard, open('data/score_leaderboard.json', 'w'),
          ensure_ascii=False, indent=2)

# ── 72경기 예측 (전체기간 재학습한 대표=Dixon-Coles) ───────────
PRIMARY = DixonColes().fit(df['elo_diff_pre'].values,
                           df['home_score'].values, df['away_score'].values)

# 시뮬레이션(sim.py)이 임의 대진의 기대득점을 계산할 수 있도록 계수 export.
#   log λ_home = a0 + a1·(d/100) ,  log λ_away = b0 + b1·(d/100)  (d=elo_diff_pre)
params = {
    'a0': float(np.ravel(PRIMARY.mh.intercept_)[0]), 'a1': float(np.ravel(PRIMARY.mh.coef_)[0]),
    'b0': float(np.ravel(PRIMARY.ma.intercept_)[0]), 'b1': float(np.ravel(PRIMARY.ma.coef_)[0]),
    'rho': float(PRIMARY.rho), 'scale': 100.0,
}
json.dump(params, open('data/score_model_params.json', 'w'), indent=2)

ratings = pd.read_csv('data/elo_final.csv', index_col=0)['elo'].to_dict()
fixtures = pd.read_csv('data/results.csv')
wc = fixtures[(fixtures['date'] >= '2026-06-11') &
              (fixtures['tournament'] == 'FIFA World Cup')].copy()


def handicap_probs(grid):
    """홈 관점 핸디캡 라인별 커버 확률. line<0 이면 홈이 그만큼 접어줌."""
    diff = np.subtract.outer(np.arange(MAXG), np.arange(MAXG))  # home - away
    out = {}
    for line in HCAP_LINES:
        # 홈 +line 의 결과(home_margin + line > 0)
        cover = grid[(diff + line) > 0].sum()
        out[str(line)] = round(float(cover), 4)
    return out


def fair_handicap(lh, la):
    """홈이 -X 핸디를 줘도 50:50 이 되는 대략적 라인(기대 득점차 반올림 .5단위)."""
    sup = lh - la
    return round(sup * 2) / 2  # 0.5 단위


def ou_probs(grid):
    tot = np.add.outer(np.arange(MAXG), np.arange(MAXG))
    out = {}
    for L in OU_LINES:
        out[str(L)] = round(float(grid[tot > L].sum()), 4)
    return out


def top_scores(grid, n=5):
    flat = [(int(x), int(y), float(grid[x, y]))
            for x in range(MAXG) for y in range(MAXG)]
    flat.sort(key=lambda t: -t[2])
    return [{'h': x, 'a': y, 'p': round(p, 4)} for x, y, p in flat[:n]]


def rationale(home, away, d, lh, la, grid, ou, top, hcap):
    sup = lh - la
    ml = top[0]
    fav = home if sup > 0.15 else (away if sup < -0.15 else None)
    parts = []
    parts.append(
        f"경기 전 Elo 차이 {d:+.0f} → 기대 득점 {home} {lh:.2f} : {la:.2f} {away}."
    )
    if fav:
        parts.append(
            f"득점 우위 {abs(sup):.2f}골로 {fav} 우세. "
            f"최빈 스코어 {ml['h']}-{ml['a']} ({ml['p']*100:.0f}%)."
        )
    else:
        parts.append(
            f"기대 득점차 {abs(sup):.2f}골로 박빙. 최빈 스코어 {ml['h']}-{ml['a']} "
            f"({ml['p']*100:.0f}%)."
        )
    o25 = ou['2.5'] * 100
    parts.append(
        f"합산 기대 {lh+la:.2f}골 → 오버 2.5 {o25:.0f}% / 언더 {100-o25:.0f}%."
    )
    fh = fair_handicap(lh, la)
    if fh != 0:
        side = home if fh > 0 else away
        parts.append(f"적정 핸디캡 약 {side} -{abs(fh):.1f}.")
    return ' '.join(parts)


rows = []
for r in wc.sort_values('date').itertuples():
    d = ratings[r.home_team] + (0 if r.neutral else HOME_ADV) - ratings[r.away_team]
    lh, la = PRIMARY.lams(np.array([d]))
    lh, la = float(lh[0]), float(la[0])
    grid = PRIMARY.grid(lh, la)
    ou = ou_probs(grid)
    top = top_scores(grid)
    hcap = handicap_probs(grid)
    rows.append({
        'date': r.date, 'home': r.home_team, 'away': r.away_team,
        'eloDiff': round(d), 'lambdaHome': round(lh, 2), 'lambdaAway': round(la, 2),
        'expScore': f"{top[0]['h']}-{top[0]['a']}",  # 최빈(가장 가능성 큰) 스코어
        'topScores': top,
        'overUnder': ou,
        'handicap': hcap,
        'fairHandicap': fair_handicap(lh, la),
        'rationale': rationale(r.home_team, r.away_team, d, lh, la, grid, ou, top, hcap),
    })

json.dump(rows, open('data/score_predictions.json', 'w'),
          ensure_ascii=False, indent=2)
print(f"\n저장: score_leaderboard.json, score_predictions.json ({len(rows)}경기)")
print('예시:', rows[0]['home'], rows[0]['expScore'], '|', rows[0]['rationale'])
