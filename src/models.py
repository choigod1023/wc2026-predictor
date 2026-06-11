"""
models.py — 확률 모델 레지스트리 (여러 모델을 한 인터페이스로)
================================================================
목적: "하나의 모델"이 아니라 여러 후보 모델을 동일한 입력/출력 규약으로
정의해 두고, compare_models.py가 walk-forward Brier로 공정 비교하게 한다.

규약 (모든 모델 공통):
  - fit(elo_diff: np.ndarray, outcome: np.ndarray)  # outcome ∈ {'H','D','A'}
  - predict_proba(elo_diff: np.ndarray) -> (N,3) 배열, 열 순서 = [P(H), P(D), P(A)]
  - name: 사람이 읽는 이름

입력 피처는 모두 elo_diff_pre(경기 전, 홈 어드밴티지 반영) 하나로 통일한다.
(피처 추가 실험은 모델 안에서 elo_diff를 변환해 쓰는 식으로만 — 데이터 누수
방지를 위해 외부에서 미래 정보를 끌어오지 않는다.)

여기 모델을 "채택"하는 게 아니라 "후보로 등록"하는 것이다. 채택 여부는
walk-forward Brier 개선으로만 판정한다 (CLAUDE.md 백테스트 우선 원칙).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

HDA = ['H', 'D', 'A']


def _onehot_order(model_classes):
    """sklearn 클래스 순서를 [H,D,A] 인덱스로 매핑."""
    return [list(model_classes).index(c) for c in HDA]


class EloLogistic:
    """M1 (기준 모델). elo_diff 단일 피처 다항 로지스틱. 현재 운영 모델."""
    name = 'Elo-로지스틱 (기준)'
    short = 'elo_logit'

    def fit(self, elo_diff, outcome):
        self.m = LogisticRegression(max_iter=1000)
        self.m.fit(elo_diff.reshape(-1, 1), outcome)
        self.idx = _onehot_order(self.m.classes_)
        return self

    def predict_proba(self, elo_diff):
        p = self.m.predict_proba(elo_diff.reshape(-1, 1))
        return p[:, self.idx]


class EloLogisticAbs:
    """M2. elo_diff 와 |elo_diff| 두 피처. 접전(차이≈0)일수록 무승부가
    늘어나는 비선형성을 명시 피처로 잡으려는 시도."""
    name = 'Elo-로지스틱 +|diff|'
    short = 'elo_logit_abs'

    def _X(self, d):
        d = d.reshape(-1, 1)
        return np.hstack([d, np.abs(d)])

    def fit(self, elo_diff, outcome):
        self.m = LogisticRegression(max_iter=1000)
        self.m.fit(self._X(elo_diff), outcome)
        self.idx = _onehot_order(self.m.classes_)
        return self

    def predict_proba(self, elo_diff):
        p = self.m.predict_proba(self._X(elo_diff))
        return p[:, self.idx]


class Davidson:
    """M3. Davidson(1970) 무승부 모델 — 학습 파라미터 단 1개(ν).
    Elo 기대승률에서 무승부를 해석적으로 분해한다(로지스틱 회귀 불사용).

      g = 10^(elo_diff/400)        # 홈/원정 상대 승산
      분모 = g + 1/g·? → 표준형:
      P(H) = g / (g + 1 + ν·√g)
      P(A) = 1 / (g + 1 + ν·√g)
      P(D) = ν·√g / (g + 1 + ν·√g)

    ν 는 학습기간 무승부 비율에 로그우도를 최대화하도록 1차원 탐색으로 적합.
    """
    name = 'Davidson 무승부모델'
    short = 'davidson'

    def _probs(self, elo_diff, nu):
        g = np.power(10.0, elo_diff / 400.0)
        sg = np.sqrt(g)
        denom = g + 1.0 + nu * sg
        pH = g / denom
        pA = 1.0 / denom
        pD = nu * sg / denom
        return np.clip(np.stack([pH, pD, pA], axis=1), 1e-9, 1)

    def fit(self, elo_diff, outcome):
        y = np.array([HDA.index(o) for o in outcome])
        best_nu, best_ll = 1.0, -np.inf
        for nu in np.linspace(0.05, 3.0, 120):
            p = self._probs(elo_diff, nu)
            ll = np.sum(np.log(p[np.arange(len(y)), y]))
            if ll > best_ll:
                best_ll, best_nu = ll, nu
        self.nu = best_nu
        return self

    def predict_proba(self, elo_diff):
        return self._probs(elo_diff, self.nu)


class BaseRate:
    """기준선 1. 학습기간의 H/D/A 평균 비율을 항상 출력."""
    name = '기준선·비율(base-rate)'
    short = 'base_rate'

    def fit(self, elo_diff, outcome):
        self.rates = np.array([np.mean(outcome == c) for c in HDA])
        return self

    def predict_proba(self, elo_diff):
        return np.tile(self.rates, (len(elo_diff), 1))


class Uniform:
    """기준선 2. 항상 (1/3, 1/3, 1/3)."""
    name = '기준선·균등(1/3)'
    short = 'uniform'

    def fit(self, elo_diff, outcome):
        return self

    def predict_proba(self, elo_diff):
        return np.full((len(elo_diff), 3), 1 / 3)


# 비교에 쓸 모델 묶음 (순서 = 표시 순서)
REGISTRY = [EloLogistic, EloLogisticAbs, Davidson, BaseRate, Uniform]

# 우승 시뮬레이션에 돌릴 "진짜 모델"만 (기준선 제외)
SIM_MODELS = [EloLogistic, EloLogisticAbs, Davidson]
