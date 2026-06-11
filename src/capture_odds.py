"""
capture_odds.py — named API 의 경기별 3-way 배당을 closing_odds.csv 에 자동 적재
==================================================================================
이 프로젝트의 핵심 판정 데이터(마감 배당)를 수기로 적던 것을 자동화한다.
named API(키 불필요)는 월드컵 경기마다 소수 3-way 배당(WIN/DRAW/LOSS)을 준다.

원칙:
  - '마감(closing)' 배당은 킥오프 직전 값이 이상적이다. 이 스크립트를 킥오프
    전후로(예: 크론 10분 간격, 또는 매일 --update 시) 실행하면 그 시점의 배당을 채운다.
  - 멱등·비파괴: 이미 채워진 행은 덮어쓰지 않는다(한 번 기록된 마감배당 보존).
    --overwrite 를 주면 갱신한다(킥오프 직전 최신값으로 굳히고 싶을 때).
사용: python src/capture_odds.py [--overwrite]
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import sys
import ssl
import json
import datetime
import urllib.request
import pandas as pd

# macOS 등에서 시스템 인증서 누락으로 SSL 검증이 실패하는 경우가 잦다.
# certifi 가 있으면 그 번들을 쓰고, 없으면 (읽기 전용 공개 API이므로) 미검증으로 폴백.
try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _SSL = ssl._create_unverified_context()

WC_LEAGUE_ID = 639
BASE = 'https://sports-api.named.com/v1.0'

# named(한국어) → 영어(스케줄) 팀명 매핑 (본선 48팀)
KO2EN = {
    '가나': 'Ghana', '남아공': 'South Africa', '네덜란드': 'Netherlands',
    '노르웨이': 'Norway', '뉴질랜드': 'New Zealand', '대한민국': 'South Korea',
    '독일': 'Germany', '멕시코': 'Mexico', '모로코': 'Morocco', '미국': 'United States',
    '벨기에': 'Belgium', '보스니아 헤르체고비나': 'Bosnia and Herzegovina',
    '브라질': 'Brazil', '사우디아라비아': 'Saudi Arabia', '세네갈': 'Senegal',
    '스웨덴': 'Sweden', '스위스': 'Switzerland', '스코틀랜드': 'Scotland',
    '스페인': 'Spain', '아르헨티나': 'Argentina', '아이티': 'Haiti', '알제리': 'Algeria',
    '에콰도르': 'Ecuador', '오스트리아': 'Austria', '요르단': 'Jordan',
    '우루과이': 'Uruguay', '우즈베키스탄': 'Uzbekistan', '이라크': 'Iraq', '이란': 'Iran',
    '이집트': 'Egypt', '일본': 'Japan', '잉글랜드': 'England', '체코': 'Czech Republic',
    '카보베르데': 'Cape Verde', '카타르': 'Qatar', '캐나다': 'Canada',
    '코트디부아르': 'Ivory Coast', '콜롬비아': 'Colombia', '콩고민주공화국': 'DR Congo',
    '퀴라소': 'Curaçao', '크로아티아': 'Croatia', '튀니지': 'Tunisia', '튀르키예': 'Turkey',
    '파나마': 'Panama', '파라과이': 'Paraguay', '포르투갈': 'Portugal', '프랑스': 'France',
    '호주': 'Australia',
}


def _get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=20, context=_SSL) as r:
        return json.load(r)


def extract_odds(game):
    """representativeOdds.domestic → (WIN, DRAW, LOSS) 소수배당. 없으면 None."""
    ro = (game.get('representativeOdds') or {}).get('domestic') or {}
    odds = ro.get('odds') or (game.get('odds') or {}).get('domesticWinLoseOdds')
    if not isinstance(odds, list):
        return None
    # 배열에 여러 회차(proto round)가 [WIN,DRAW,LOSS] 묶음으로 반복된다.
    # 대표값 = 첫 번째 묶음. 타입별 '첫 등장' 값만 취한다(덮어쓰기 금지).
    by = {}
    for o in odds:
        if o.get('latestFlag') is not False:
            by.setdefault(o.get('type'), o.get('odds'))
    if all(by.get(k) for k in ('WIN', 'DRAW', 'LOSS')):
        return by['WIN'], by['DRAW'], by['LOSS']
    return None


def collect_odds():
    """개막일~오늘+1 의 월드컵 경기에서 배당이 있는 것을 모아
    {(home_en, away_en): (oWIN, oDRAW, oLOSS)} 반환 (home/away 는 named 표기 기준)."""
    today = datetime.date.today()
    start = datetime.date(2026, 6, 11)
    out = {}
    # popular-games(배당 포함)를 날짜별로 훑는다.
    d = start
    while d <= today + datetime.timedelta(days=1):
        try:
            data = _get(f'{BASE}/popular-games?date={d}&tomorrow-game-flag=true')
            for g in data.get('soccer', []):
                if g.get('league', {}).get('id') != WC_LEAGUE_ID:
                    continue
                od = extract_odds(g)
                if not od:
                    continue
                h = KO2EN.get(g['teams']['home']['name'])
                a = KO2EN.get(g['teams']['away']['name'])
                if h and a:
                    out[(h, a)] = od
        except Exception as e:
            print(f'  ! {d} 조회 실패: {e}')
        d += datetime.timedelta(days=1)
    return out


def main():
    overwrite = '--overwrite' in sys.argv
    df = pd.read_csv('data/closing_odds.csv')
    odds = collect_odds()
    print(f'named 배당 수집: {len(odds)}경기')

    filled = 0
    for i, row in df.iterrows():
        key = (row['home'], row['away'])
        rev = (row['away'], row['home'])
        has = pd.notna(row['odds_H']) and pd.notna(row['odds_D']) and pd.notna(row['odds_A'])
        if has and not overwrite:
            continue
        if key in odds:
            w, dr, l = odds[key]                      # 홈/무/원정 그대로
        elif rev in odds:
            w2, dr, l2 = odds[rev]                     # named 홈/원정이 반대 → 뒤집기
            w, l = l2, w2
        else:
            continue
        df.at[i, 'odds_H'] = w
        df.at[i, 'odds_D'] = dr
        df.at[i, 'odds_A'] = l
        filled += 1

    df.to_csv('data/closing_odds.csv', index=False)
    done = df[['odds_H', 'odds_D', 'odds_A']].notna().all(axis=1).sum()
    print(f'이번에 기록: {filled}경기 | 누적 배당 보유: {done}/{len(df)}경기')
    if filled:
        print('→ python src/evaluate.py 로 모델 vs 시장 채점 가능 (결과가 있는 경기 한정)')


if __name__ == '__main__':
    main()
