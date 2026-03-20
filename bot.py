import os
import sys
import json
import time
import requests
import pandas as pd
import anthropic
from pydantic import BaseModel
from typing import Literal
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("KIS_APP_KEY")
APP_SECRET = os.getenv("KIS_APP_SECRET")
ACCOUNT = os.getenv("KIS_ACCOUNT")
BASE_URL = os.getenv("KIS_URL")

TOKEN_CACHE_FILE = ".token_cache.json"

# 1. 접근토큰 발급 (캐시 사용 — KIS는 1분에 1회 제한)
def get_token():
    # 캐시된 토큰이 있고 아직 유효하면 재사용
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE) as f:
            cache = json.load(f)
        expires_at = datetime.fromisoformat(cache["expires_at"])
        if datetime.now() < expires_at:
            print("✅ 토큰 캐시 사용 중")
            return cache["access_token"]

    url = f"{BASE_URL}/oauth2/tokenP"
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    res = requests.post(url, json=body)
    data = res.json()
    if "access_token" not in data:
        raise RuntimeError(f"토큰 발급 실패: {data}")
    token = data["access_token"]
    # 만료 시각 저장 (KIS 토큰 유효기간 24시간, 여유있게 23시간으로)
    expires_at = datetime.now() + timedelta(hours=23)
    with open(TOKEN_CACHE_FILE, "w") as f:
        json.dump({"access_token": token, "expires_at": expires_at.isoformat()}, f)
    print("✅ 토큰 발급 성공!")
    return token

# 2. 주식 현재가 조회
def get_price(token, symbol):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-price"
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010100"
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": symbol
    }
    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    price = data["output"]["stck_prpr"]
    print(f"📈 {symbol} 현재가: {int(price):,}원")
    return int(price)

# 3. 일봉 데이터 조회 (최근 30 영업일 — KIS inquire-daily-price 기준)
def get_daily_ohlcv(token, symbol):
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHKST01010400"
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": symbol,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "1"
    }
    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    if data.get("rt_cd") != "0":
        raise RuntimeError(f"일봉 조회 실패: {data.get('msg1')}")

    rows = []
    for item in data["output"]:
        rows.append({
            "date": item["stck_bsop_date"],
            "open": float(item["stck_oprc"]),
            "high": float(item["stck_hgpr"]),
            "low": float(item["stck_lwpr"]),
            "close": float(item["stck_clpr"]),
            "volume": float(item["acml_vol"])
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df

# 4. RSI 계산 (기본 14일)
def calc_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

# 5. MACD 계산 (12, 26, 9)
def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

# 6. 볼린저밴드 계산 (20일, 2σ)
def calc_bollinger(df, period=20, std_dev=2):
    df["bb_mid"] = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = df["bb_mid"] + std_dev * std
    df["bb_lower"] = df["bb_mid"] - std_dev * std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])  # 0~1: 밴드 내 위치
    return df

# 7. 거래량 지표 계산
def calc_volume(df):
    df["vol_ma20"] = df["volume"].rolling(20).mean()          # 20일 평균 거래량
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]           # 거래량 비율 (1.5 = 평균의 150%)
    df["vol_ma5"] = df["volume"].rolling(5).mean()            # 5일 평균 거래량
    df["vol_trend"] = df["vol_ma5"] / df["vol_ma20"]          # 거래량 추세 (>1 이면 증가 추세)
    return df

# 8. 전체 지표 계산 후 최신값 반환
def get_indicators(token, symbol):
    df = get_daily_ohlcv(token, symbol)
    df = calc_rsi(df)
    df = calc_macd(df)
    df = calc_bollinger(df)
    df = calc_volume(df)

    latest = df.iloc[-1]
    indicators = {
        "symbol": symbol,
        "date": latest["date"],
        "close": latest["close"],
        "rsi": round(latest["rsi"], 2),
        "macd": round(latest["macd"], 2),
        "macd_signal": round(latest["macd_signal"], 2),
        "macd_hist": round(latest["macd_hist"], 2),
        "bb_upper": round(latest["bb_upper"], 2),
        "bb_mid": round(latest["bb_mid"], 2),
        "bb_lower": round(latest["bb_lower"], 2),
        "bb_pct": round(latest["bb_pct"], 3),
        "volume": int(latest["volume"]),
        "vol_ma20": round(latest["vol_ma20"], 0),
        "vol_ratio": round(latest["vol_ratio"], 2),    # 오늘 거래량 / 20일 평균
        "vol_trend": round(latest["vol_trend"], 2),    # 5일 평균 / 20일 평균
    }
    return indicators, df

# 8. Claude AI 매매 판단
# 응답 구조 정의 (Pydantic으로 타입 보장)
class TradeDecision(BaseModel):
    action: Literal["BUY", "SELL", "HOLD"]   # 매수 / 매도 / 관망
    confidence: int                            # 확신도 0~100
    target_price: int                          # 목표가 (원)
    stop_loss: int                             # 손절가 (원)
    reason: str                                # 판단 근거 (한국어)

def ask_claude(indicators: dict) -> TradeDecision:
    claude = anthropic.Anthropic()  # ANTHROPIC_API_KEY 환경변수 자동 사용

    prompt = f"""
당신은 한국 주식 시장 전문 퀀트 트레이더입니다.
아래 기술적 지표를 분석하고 매매 판단을 내려주세요.

=== 종목 정보 ===
종목코드: {indicators['symbol']}
기준일: {indicators['date']}
현재가: {indicators['close']:,.0f}원

=== 기술적 지표 ===
RSI(14): {indicators['rsi']}
  - 70 초과: 과매수 (매도 고려)
  - 30 미만: 과매도 (매수 고려)

MACD: {indicators['macd']}
MACD 시그널: {indicators['macd_signal']}
MACD 히스토그램: {indicators['macd_hist']}
  - 히스토그램 양수이면 상승 모멘텀
  - 히스토그램이 0을 상향 돌파하면 매수 신호

볼린저밴드:
  상단: {indicators['bb_upper']:,.0f}원
  중단(기준선): {indicators['bb_mid']:,.0f}원
  하단: {indicators['bb_lower']:,.0f}원
  밴드 내 위치: {indicators['bb_pct']*100:.1f}% (0%=하단, 100%=상단)

거래량:
  오늘 거래량: {indicators['volume']:,}주
  20일 평균: {int(indicators['vol_ma20']):,}주
  거래량 비율: {indicators['vol_ratio']}배 (1.5 이상이면 거래량 폭발)
  거래량 추세: {indicators['vol_trend']} (5일평균/20일평균, 1 초과면 증가 추세)

=== 판단 기준 ===
- 네 지표(RSI, MACD, 볼린저밴드, 거래량)가 같은 방향을 가리킬 때 확신도를 높게 설정하세요.
- 거래량 비율 1.5 이상 + 가격 상승 = 강한 매수 신호
- 거래량 비율 1.5 이상 + 가격 하락 = 강한 매도/관망 신호
- 거래량 없는 상승은 신뢰도가 낮습니다 (vol_ratio < 0.8이면 확신도 낮게)
- 지표들이 엇갈릴 때는 HOLD를 선호하세요.
- 목표가는 현재가 기준 현실적인 단기 목표(3~7%)로 설정하세요.
- 손절가는 현재가 기준 -2~3% 수준으로 설정하세요.
"""

    response = claude.messages.parse(
        model="claude-opus-4-6",
        max_tokens=1024,
        thinking={"type": "adaptive"},   # 복잡한 판단이므로 adaptive thinking 활성화
        messages=[{"role": "user", "content": prompt}],
        output_format=TradeDecision,
    )

    return response.parsed_output

# ─────────────────────────────────────────
# 리스크 관리 설정
# ─────────────────────────────────────────
MIN_CONFIDENCE = 70      # 확신도 70% 미만이면 주문 안 냄
MAX_POSITION_RATIO = 0.3 # 보유 현금의 최대 30%까지만 매수

# 9. 해시키 발급 (KIS POST 요청 보안용)
def get_hashkey(token, body: dict) -> str:
    url = f"{BASE_URL}/uapi/hashkey"
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "Content-Type": "application/json"
    }
    res = requests.post(url, headers=headers, json=body)
    return res.json()["HASH"]

# 10. 주문 가능 현금 잔고 조회
def get_balance(token) -> int:
    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
    # 모의투자와 실전투자 tr_id 자동 분기
    is_paper = "vts" in BASE_URL
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "VTTC8908R" if is_paper else "TTTC8908R"
    }
    params = {
        "CANO": ACCOUNT[:8],
        "ACNT_PRDT_CD": ACCOUNT[9:] if len(ACCOUNT) > 8 else "01",
        "PDNO": "005930",
        "ORD_UNPR": "0",
        "ORD_DVSN": "01",
        "CMA_EVLU_AMT_ICLD_YN": "Y",
        "OVRS_ICLD_YN": "N"
    }
    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    if data.get("rt_cd") != "0":
        raise RuntimeError(f"잔고 조회 실패: {data.get('msg1')}")
    cash = int(data["output"]["ord_psbl_cash"])
    print(f"💰 주문 가능 현금: {cash:,}원")
    return cash

# 11. 매수/매도 주문 실행
def place_order(token, symbol: str, action: str, quantity: int) -> dict:
    is_paper = "vts" in BASE_URL
    if action == "BUY":
        tr_id = "VTTC0802U" if is_paper else "TTTC0802U"
    else:
        tr_id = "VTTC0801U" if is_paper else "TTTC0801U"

    body = {
        "CANO": ACCOUNT[:8],
        "ACNT_PRDT_CD": ACCOUNT[9:] if len(ACCOUNT) > 8 else "01",
        "PDNO": symbol,
        "ORD_DVSN": "01",    # 01=시장가
        "ORD_QTY": str(quantity),
        "ORD_UNPR": "0",     # 시장가는 0
    }
    hashkey = get_hashkey(token, body)
    url = f"{BASE_URL}/uapi/domestic-stock/v1/trading/order-cash"
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": tr_id,
        "hashkey": hashkey,
        "Content-Type": "application/json"
    }
    res = requests.post(url, headers=headers, json=body)
    return res.json()

# 12. 리스크 필터 + 주문 실행 통합
def execute_trade(token, decision: TradeDecision, symbol: str, current_price: int, order_enabled: bool = True):
    action_emoji = {"BUY": "🟢 매수", "SELL": "🔴 매도", "HOLD": "⚪ 관망"}
    print(f"\n{'='*40}")
    print(f"판단:     {action_emoji[decision.action]}")
    print(f"확신도:   {decision.confidence}%")
    print(f"목표가:   {decision.target_price:,}원")
    print(f"손절가:   {decision.stop_loss:,}원")
    print(f"근거:     {decision.reason}")
    print(f"{'='*40}")

    # HOLD면 주문 없음
    if decision.action == "HOLD":
        print("→ 관망. 주문 없음.")
        return

    # 확신도 필터: 70% 미만이면 주문 안 냄
    if decision.confidence < MIN_CONFIDENCE:
        print(f"→ 확신도 {decision.confidence}% < {MIN_CONFIDENCE}% 기준 미달. 주문 취소.")
        return

    # 장 외 시간이면 분석만 하고 주문은 건너뜀
    if not order_enabled:
        print(f"→ {action_emoji[decision.action]} 신호 확인. 장 시작 후 주문 예정.")
        return

    # 매수: 주문 가능 현금의 30%로 수량 계산
    if decision.action == "BUY":
        cash = get_balance(token)
        budget = int(cash * MAX_POSITION_RATIO)
        quantity = budget // current_price
        if quantity < 1:
            print(f"→ 예산 부족 (가용 예산: {budget:,}원 / 현재가: {current_price:,}원). 주문 취소.")
            return

    # 매도: 1주 (실제 보유 수량 조회는 추후 추가)
    else:
        quantity = 1

    mode = "모의투자" if "vts" in BASE_URL else "실전투자"
    print(f"\n📨 [{mode}] {action_emoji[decision.action]} 주문 전송 중...")
    print(f"   종목: {symbol} / 수량: {quantity}주 / 방식: 시장가")

    result = place_order(token, symbol, decision.action, quantity)

    if result.get("rt_cd") == "0":
        print(f"✅ 주문 성공! 주문번호: {result['output']['ODNO']}")
    else:
        print(f"❌ 주문 실패: {result.get('msg1')}")

# ─────────────────────────────────────────
# 13. 거래량 상위 종목 조회
# ─────────────────────────────────────────
def get_top_volume_stocks(token, top_n=20) -> list[dict]:
    url = f"{BASE_URL}/uapi/domestic-stock/v1/quotations/volume-rank"
    headers = {
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHPST01710000"
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_COND_SCR_DIV_CODE": "20171",
        "FID_INPUT_ISCD": "0000",
        "FID_DIV_CLS_CODE": "0",
        "FID_BLNG_CLS_CODE": "0",
        "FID_TRGT_CLS_CODE": "111111111",
        "FID_TRGT_EXLS_CLS_CODE": "000000",
        "FID_INPUT_PRICE_1": "5000",     # 최소 주가 5,000원 (너무 저가주 제외)
        "FID_INPUT_PRICE_2": "500000",   # 최대 주가
        "FID_VOL_CNT": "100000",         # 최소 거래량
        "FID_INPUT_DATE_1": ""
    }
    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    if data.get("rt_cd") != "0":
        raise RuntimeError(f"거래량 조회 실패: {data.get('msg1')}")

    stocks = []
    for item in data["output"][:top_n]:
        stocks.append({
            "symbol": item["mksc_shrn_iscd"],
            "name": item["hts_kor_isnm"],
            "price": int(item["stck_prpr"]),
            "volume": int(item["acml_vol"]),
        })
    return stocks

# ─────────────────────────────────────────
# 14. 1차 필터: 지표로 후보 종목 추리기
# ─────────────────────────────────────────
def screen_stocks(token, stocks: list[dict]) -> list[dict]:
    candidates = []
    print(f"\n🔍 {len(stocks)}개 종목 1차 스크리닝 중...")

    for stock in stocks:
        try:
            time.sleep(0.5)  # KIS API 초당 요청 제한 대응
            indicators, _ = get_indicators(token, stock["symbol"])

            # 1차 필터 조건 (네 가지 모두 통과해야 함)
            rsi_ok  = 30 <= indicators["rsi"] <= 65         # 과매수 아닌 것
            macd_ok = indicators["macd_hist"] > 0            # 상승 모멘텀
            bb_ok   = indicators["bb_pct"] < 0.75            # 밴드 상단 여유 있는 것
            vol_ok  = indicators["vol_ratio"] >= 0.8         # 거래량 너무 적지 않은 것 (평균의 80% 이상)

            passed = rsi_ok and macd_ok and bb_ok and vol_ok
            status = "✅" if passed else "❌"
            print(f"  {status} {stock['name']:10s} RSI={indicators['rsi']:.1f} "
                  f"MACD히스토={indicators['macd_hist']:.1f} BB={indicators['bb_pct']*100:.0f}% "
                  f"거래량={indicators['vol_ratio']:.1f}x")

            if passed:
                stock["indicators"] = indicators
                candidates.append(stock)

        except Exception as e:
            print(f"  ⚠️  {stock['name']} 스킵: {e}")

    print(f"\n→ 1차 통과: {len(candidates)}개")
    return candidates

# ─────────────────────────────────────────
# 15. 2차 필터: Claude 최종 판단
# ─────────────────────────────────────────
def rank_by_claude(token, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    print(f"\n🤖 Claude AI 2차 분석 중... ({len(candidates)}개)")
    results = []

    for stock in candidates:
        print(f"  분석 중: {stock['name']} ({stock['symbol']})")
        decision = ask_claude(stock["indicators"])
        stock["decision"] = decision

        action_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
        print(f"  → {action_emoji[decision.action]} {decision.action} / 확신도 {decision.confidence}%")

        if decision.action == "BUY" and decision.confidence >= MIN_CONFIDENCE:
            results.append(stock)

    # 확신도 높은 순으로 정렬
    results.sort(key=lambda x: x["decision"].confidence, reverse=True)
    return results

# ─────────────────────────────────────────
# 16. 단일 종목 분석 출력
# ─────────────────────────────────────────
def analyze_and_print(token, symbol: str, order_enabled: bool = True):
    price = get_price(token, symbol)

    print("\n📊 기술적 지표 계산 중...")
    indicators, _ = get_indicators(token, symbol)

    print(f"\n=== {indicators['symbol']} ({indicators['date']}) ===")
    print(f"현재가:    {indicators['close']:,.0f}원")
    print(f"RSI(14):   {indicators['rsi']} {'🔴 과매수' if indicators['rsi'] > 70 else '🔵 과매도' if indicators['rsi'] < 30 else '⚪ 중립'}")
    print(f"MACD 히스: {indicators['macd_hist']} {'↑ 상승' if indicators['macd_hist'] > 0 else '↓ 하락'}")
    print(f"BB 위치:   {indicators['bb_pct']*100:.1f}% {'⚠️ 상단 근접' if indicators['bb_pct'] > 0.8 else '💡 하단 근접' if indicators['bb_pct'] < 0.2 else ''}")
    vol_label = "🔥 거래량 폭발" if indicators['vol_ratio'] >= 1.5 else ("📉 거래량 감소" if indicators['vol_ratio'] < 0.8 else "")
    print(f"거래량:    평균 대비 {indicators['vol_ratio']}배 {vol_label}")

    print("\n🤖 Claude AI 판단 중...")
    decision = ask_claude(indicators)
    execute_trade(token, decision, symbol, price, order_enabled=order_enabled)

# ─────────────────────────────────────────
# 17. 장 운영 시간 체크
# ─────────────────────────────────────────
LOOP_INTERVAL = 30  # 분

def is_market_open() -> bool:
    now = datetime.now()
    # 주말 제외
    if now.weekday() >= 5:
        return False
    # 09:00 ~ 15:30
    market_open  = now.replace(hour=9,  minute=0,  second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

# ─────────────────────────────────────────
# 18. 1회 스크리닝 사이클
# ─────────────────────────────────────────
def run_once(token, user_symbols: list[str], order_enabled: bool = True):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"  실행 시각: {now_str}")
    print(f"{'='*50}")

    if not order_enabled:
        print("  [분석 전용 — 장 외 시간이라 주문 없음]")

    if user_symbols:
        # ── 직접 입력 모드 ──
        for symbol in user_symbols:
            print(f"\n{'─'*40}")
            analyze_and_print(token, symbol, order_enabled=order_enabled)
    else:
        # ── 자동 스크리닝 모드 ──
        print("\n🔎 자동 스크리닝 모드")
        stocks = get_top_volume_stocks(token, top_n=20)
        candidates = screen_stocks(token, stocks)

        if not candidates:
            print("\n😔 1차 필터 통과 종목 없음. 관망.")
            return

        top_picks = rank_by_claude(token, candidates)

        if not top_picks:
            print("\n😔 BUY 신호 없음. 관망.")
            return

        print(f"\n🏆 최종 추천 종목 (확신도 순)")
        print(f"{'─'*50}")
        for i, stock in enumerate(top_picks, 1):
            d = stock["decision"]
            print(f"{i}. {stock['name']} ({stock['symbol']})")
            print(f"   현재가: {stock['price']:,}원 | 확신도: {d.confidence}%")
            print(f"   목표가: {d.target_price:,}원 | 손절가: {d.stop_loss:,}원")
            print(f"   근거: {d.reason[:60]}...")

        best = top_picks[0]
        if order_enabled:
            print(f"\n→ 1위 종목 {best['name']} 주문 실행")
            execute_trade(token, best["decision"], best["symbol"], best["price"])
        else:
            print(f"\n→ 1위 종목 {best['name']} — 장 시작 후 주문 예정")

# ─────────────────────────────────────────
# 실행 진입점
# ─────────────────────────────────────────
if __name__ == "__main__":
    user_symbols = sys.argv[1:]
    mode = f"직접 지정 {user_symbols}" if user_symbols else "자동 스크리닝"
    print(f"\n🤖 트레이딩 봇 시작 | 모드: {mode} | 인터벌: {LOOP_INTERVAL}분")
    print("종료하려면 Ctrl+C")

    token = get_token()

    try:
        while True:
            market_open = is_market_open()
            if not market_open:
                now = datetime.now()
                next_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
                if now >= next_open:
                    days_ahead = 1
                    if now.weekday() == 4:
                        days_ahead = 3
                    elif now.weekday() == 5:
                        days_ahead = 2
                    next_open += timedelta(days=days_ahead)
                wait_sec = (next_open - now).total_seconds()
                print(f"\n💤 장 외 시간 — 분석은 실행하되 주문은 건너뜀")
                print(f"   다음 장 시작: {next_open.strftime('%m/%d %H:%M')} "
                      f"({int(wait_sec//3600)}시간 {int((wait_sec%3600)//60)}분 후)")

            try:
                run_once(token, user_symbols, order_enabled=market_open)
            except Exception as e:
                print(f"\n❌ 오류 발생: {e} → 다음 사이클에 재시도")

            print(f"\n⏳ {LOOP_INTERVAL}분 후 재실행...")
            time.sleep(LOOP_INTERVAL * 60)

    except KeyboardInterrupt:
        print("\n\n👋 봇 종료")
