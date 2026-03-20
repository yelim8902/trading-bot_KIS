"""
백테스트 — 과거 데이터로 봇 전략 성능 검증
사용: python backtest.py [종목코드] [시작일] [종료일]
예시: python backtest.py 005930 20240101 20260319
"""

import sys
import pandas as pd
from pykrx import stock as krx
from datetime import datetime

# ─────────────────────────────────────────
# 전략 파라미터
# ─────────────────────────────────────────
STOP_LOSS_PCT    = 0.05   # 손절: 매수가 대비 -5%
TAKE_PROFIT_PCT  = 0.08   # 익절: 매수가 대비 +8%
MIN_HOLD_DAYS    = 3      # 최소 보유 기간 (손절 제외)
MACD_SELL_DAYS   = 2      # MACD 히스토그램 N일 연속 감소 시 청산
SENTIMENT_MIN    = 30     # 시장 심리 최소 (극단적 공포 구간 제외)
SENTIMENT_MAX    = 75     # 시장 심리 최대 (극단적 탐욕 구간 제외)

# ── 평균회귀 전략 전용 파라미터 ──
MR_CRASH_PCT     = -0.07  # 5일 수익률 -7% 이하 = 단기 급락 판정
MR_RSI_MAX       = 38     # 과매도 RSI 상한
MR_VOL_MIN       = 1.3    # 패닉셀 확인용 최소 거래량 비율
MR_STOP_LOSS     = 0.07   # 평균회귀 손절 -7% (추세추종보다 넓게)
MR_TAKE_PROFIT   = 0.10   # 평균회귀 익절 +10%

# ─────────────────────────────────────────
# 거래 비용 (현실 반영)
# ─────────────────────────────────────────
BUY_COST_PCT  = 0.00015   # 매수 수수료 0.015%
SELL_COST_PCT = 0.00015 + 0.0018   # 매도 수수료 0.015% + 증권거래세 0.18%
# 왕복 합계 ≈ 0.21% (HTS 수수료 기준, 증권사별 상이)

# ─────────────────────────────────────────
# 1. 과거 데이터 수집 (pykrx)
# ─────────────────────────────────────────
def get_historical_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = krx.get_market_ohlcv(start, end, symbol)
    df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].astype(str)
    return df

# ─────────────────────────────────────────
# 2. 시장 심리 지수 계산 (KODEX200 기반)
# 0 = 극단적 공포 / 50 = 중립 / 100 = 극단적 탐욕
# ─────────────────────────────────────────
def calc_market_sentiment(start: str, end: str) -> pd.DataFrame:
    print("  📡 KOSPI 시장 심리 지수 계산 중...")
    raw = krx.get_market_ohlcv(start, end, "069500")
    kospi = raw.rename(columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"})
    kospi.index.name = "date"
    kospi = kospi.reset_index()
    kospi["date"] = kospi["date"].astype(str)

    kospi["ma20"] = kospi["close"].rolling(20).mean()
    kospi["ma50"] = kospi["close"].rolling(50).mean()

    delta = kospi["close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    kospi["rsi"] = 100 - (100 / (1 + gain / loss))

    mid = kospi["close"].rolling(20).mean()
    std = kospi["close"].rolling(20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    kospi["bb_pct"] = (kospi["close"] - lower) / (upper - lower)

    kospi["momentum"] = kospi["close"].pct_change(20) * 100

    ma_score  = (kospi["close"] > kospi["ma20"]).astype(float) * 12.5 + \
                (kospi["close"] > kospi["ma50"]).astype(float) * 12.5
    rsi_score = kospi["rsi"].clip(0, 100) / 100 * 25
    bb_score  = kospi["bb_pct"].clip(0, 1) * 25
    mom_score = ((kospi["momentum"].clip(-10, 10) + 10) / 20) * 25

    kospi["sentiment"] = (ma_score + rsi_score + bb_score + mom_score).round(1)
    return kospi[["date", "sentiment", "close"]].rename(columns={"close": "kospi_close"})

def sentiment_label(score: float) -> str:
    if score < 20: return "😱 극단적 공포"
    if score < 40: return "😨 공포"
    if score < 60: return "😐 중립"
    if score < 80: return "😏 탐욕"
    return              "🤑 극단적 탐욕"

# ─────────────────────────────────────────
# 3. 종목 지표 계산
# ─────────────────────────────────────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI(14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # MACD(12,26,9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # MACD 히스토그램 연속 감소 여부 (N일 연속)
    df["macd_hist_prev"] = df["macd_hist"].shift(1)
    df["macd_hist_prev2"] = df["macd_hist"].shift(2)

    # 볼린저밴드(20일)
    df["bb_mid"] = df["close"].rolling(20).mean()
    std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std
    df["bb_lower"] = df["bb_mid"] - 2 * std
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # 볼린저밴드 하단 터치 후 반등 여부
    # 전일 bb_pct < 0.2 이고 오늘 bb_pct > 전일 bb_pct (반등)
    df["bb_pct_prev"] = df["bb_pct"].shift(1)
    df["bb_bounce"] = (df["bb_pct_prev"] < 0.2) & (df["bb_pct"] > df["bb_pct_prev"])

    # ── 평균회귀 전용 지표 ──
    # 5일/10일/20일 수익률 (단기 급락 & 장기 추세 감지)
    df["ret_5d"]  = df["close"].pct_change(5) * 100
    df["ret_10d"] = df["close"].pct_change(10) * 100
    df["ret_20d"] = df["close"].pct_change(20) * 100

    # BB 하단 이탈 정도 (음수일수록 많이 이탈)
    # bb_pct < 0 = 하단선 아래
    # BB 중간선 복귀 여부 (전일 중간선 아래 → 오늘 중간선 위)
    df["bb_mid_recovery"] = (df["bb_pct_prev"] < 0.5) & (df["bb_pct"] >= 0.5)

    # 거래량 지표
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]

    # MA50 추세 필터
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma50_rising"] = df["ma50"] > df["ma50"].shift(5)

    # MACD 골든크로스 (히스토그램 0 상향 돌파)
    df["macd_cross_up"] = (df["macd_hist"] > 0) & (df["macd_hist_prev"] <= 0)

    return df

# ─────────────────────────────────────────
# 4. 매수 신호 생성 (개선된 눌림목 전략)
# ─────────────────────────────────────────
def generate_signals(df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(sentiment_df, on="date", how="left")
    df["sentiment"] = df["sentiment"].ffill()

    # ── 공통 필터 ──
    base = (
        (df["ma50_rising"] == True) &                    # MA50 우상향 (추세)
        (df["sentiment"] >= SENTIMENT_MIN) &             # 극단적 공포 아님
        (df["sentiment"] <= SENTIMENT_MAX) &             # 극단적 탐욕 아님
        (df["vol_ratio"] >= 0.8)                         # 거래량 충분
    )

    # ── 전략 A: 눌림목 매수 (추세추종) ──
    # RSI 40~55 (과매수 아닌 조정 구간) + MACD 양수 + BB 하단~중단
    pullback = (
        (df["rsi"] >= 40) & (df["rsi"] <= 55) &
        (df["macd_hist"] > 0) &
        (df["bb_pct"] >= 0.2) & (df["bb_pct"] <= 0.55)
    )

    # ── 전략 B: 볼린저 하단 반등 (추세추종) ──
    # 전일 하단 터치 후 반등 + MACD 깊은 음수 아닌 것 (낙하 중 칼날 방지)
    bb_bounce = (
        (df["bb_bounce"] == True) &
        (df["rsi"] >= 30) & (df["rsi"] <= 52) &
        (df["macd_hist"] > df["macd_hist"].rolling(20).quantile(0.25))  # 하위 25% 이상만 (극단적 하락추세 제외)
    )

    # ── 전략 C: MACD 골든크로스 (추세추종) ──
    # MACD 히스토그램이 0 상향 돌파 + RSI 45~65
    macd_cross = (
        (df["macd_cross_up"] == True) &
        (df["rsi"] >= 45) & (df["rsi"] <= 65)
    )

    # ── 전략 D: 평균회귀 (갑작스러운 급락 후 반등) ──
    # 핵심: "갑자기" 빠진 것만 잡음 — 장기 하락 추세 중인 것 제외
    # 조건: 5일 급락 + RSI 과매도 + BB 하단 이탈 + 패닉셀(거래량 폭발)
    #        + 20일 수익률 -15% 이내 (장기 추세 하락 중이면 제외)
    mr_signal = (
        (df["ret_5d"] <= MR_CRASH_PCT * 100) &      # 5일 -7% 이하 급락
        (df["ret_20d"] >= -15) &                     # 20일 -15% 이내 (장기 추세 하락 아님)
        (df["rsi"] <= MR_RSI_MAX) &                  # RSI 과매도 (38 이하)
        (df["bb_pct"] <= 0.1) &                      # BB 하단 근접/이탈
        (df["vol_ratio"] >= MR_VOL_MIN) &            # 패닉셀 거래량 동반
        (df["sentiment"] >= 20)                      # 극단적 공포 아래는 제외
    )

    # 추세추종: 공통 필터(MA50 우상향 + 시장심리) 적용
    df["buy_signal"] = (base & (pullback | bb_bounce | macd_cross)) | mr_signal
    df["signal_type"] = ""
    df.loc[base & pullback,   "signal_type"] = "눌림목"
    df.loc[base & bb_bounce,  "signal_type"] = "BB반등"
    df.loc[base & macd_cross, "signal_type"] = "MACD골든"
    df.loc[mr_signal,         "signal_type"] = "평균회귀"  # 마지막에 덮어씀 (우선순위)

    return df

# ─────────────────────────────────────────
# 5. 백테스트 시뮬레이션
# ─────────────────────────────────────────
def run_backtest(df: pd.DataFrame, initial_cash: int = 10_000_000) -> dict:
    cash = initial_cash
    shares = 0
    entry_price = 0
    entry_day = 0
    entry_signal = ""         # 어떤 전략으로 매수했는지 기억
    macd_decline_streak = 0   # MACD 히스토그램 연속 감소 카운터
    trades = []

    start_idx = 55  # MA50 + 여유

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        price = row["close"]
        hold_days = i - entry_day

        # ── 포지션 있을 때 청산 조건 ──
        if shares > 0:
            is_mr = (entry_signal == "평균회귀")

            # 전략별 손절/익절 기준 분리
            sl_pct = MR_STOP_LOSS   if is_mr else STOP_LOSS_PCT
            tp_pct = MR_TAKE_PROFIT if is_mr else TAKE_PROFIT_PCT

            loss_hit   = price <= entry_price * (1 - sl_pct)
            profit_hit = price >= entry_price * (1 + tp_pct)

            # MACD 히스토그램 연속 감소 추적
            if row["macd_hist"] < row["macd_hist_prev"]:
                macd_decline_streak += 1
            else:
                macd_decline_streak = 0

            if is_mr:
                # 평균회귀: BB 중간선 회복 or RSI 55 이상 = 목표 달성 청산
                signal_sell = hold_days >= MIN_HOLD_DAYS and (
                    row["bb_mid_recovery"] or
                    row["rsi"] >= 55 or
                    macd_decline_streak >= MACD_SELL_DAYS
                )
            else:
                # 추세추종: RSI 과매수 or BB 상단 or MACD N일 연속 감소
                signal_sell = hold_days >= MIN_HOLD_DAYS and (
                    row["rsi"] > 70 or
                    row["bb_pct"] > 0.85 or
                    macd_decline_streak >= MACD_SELL_DAYS
                )

            if loss_hit or profit_hit or signal_sell:
                # 거래 비용 반영 (매도)
                sell_cost = shares * price * SELL_COST_PCT
                proceeds = shares * price - sell_cost
                gross_profit = shares * (price - entry_price)
                net_profit = gross_profit - sell_cost - (shares * entry_price * BUY_COST_PCT)
                profit_pct = (price / entry_price - 1) * 100
                net_profit_pct = profit_pct - (BUY_COST_PCT + SELL_COST_PCT) * 100

                reason = "손절" if loss_hit else "익절" if profit_hit else "신호"
                cash += proceeds
                trades.append({
                    "type": "SELL",
                    "strategy": entry_signal,
                    "reason": reason,
                    "date": row["date"],
                    "price": price,
                    "shares": shares,
                    "profit": round(net_profit),
                    "profit_pct": round(profit_pct, 2),
                    "net_profit_pct": round(net_profit_pct, 2),
                    "hold_days": hold_days,
                    "sentiment": round(row["sentiment"], 1),
                })
                shares = 0
                entry_price = 0
                entry_signal = ""
                macd_decline_streak = 0
                continue

        # ── 포지션 없을 때 매수 신호 ──
        if shares == 0 and row["buy_signal"]:
            budget = cash * 0.3
            qty = int(budget * (1 - BUY_COST_PCT) // price)
            if qty >= 1:
                shares = qty
                entry_price = price
                entry_day = i
                entry_signal = row["signal_type"]
                macd_decline_streak = 0
                cash -= shares * price * (1 + BUY_COST_PCT)
                trades.append({
                    "type": "BUY",
                    "signal": entry_signal,
                    "date": row["date"],
                    "price": price,
                    "shares": shares,
                    "buy_cost": round(shares * price * BUY_COST_PCT),
                    "rsi": round(row["rsi"], 1),
                    "ret_5d": round(row["ret_5d"], 1),
                    "macd_hist": round(row["macd_hist"], 1),
                    "bb_pct": round(row["bb_pct"] * 100, 1),
                    "vol_ratio": round(row["vol_ratio"], 2),
                    "ma50_rising": bool(row["ma50_rising"]),
                    "sentiment": round(row["sentiment"], 1),
                    "sentiment_label": sentiment_label(row["sentiment"]),
                })

    # 마지막 날 강제 청산
    if shares > 0:
        last = df.iloc[-1]
        sell_cost = shares * last["close"] * SELL_COST_PCT
        proceeds = shares * last["close"] - sell_cost
        gross_profit = shares * (last["close"] - entry_price)
        net_profit = gross_profit - sell_cost - (shares * entry_price * BUY_COST_PCT)
        profit_pct = (last["close"] / entry_price - 1) * 100
        net_profit_pct = profit_pct - (BUY_COST_PCT + SELL_COST_PCT) * 100
        cash += proceeds
        trades.append({
            "type": "SELL", "reason": "청산",
            "date": last["date"], "price": last["close"],
            "shares": shares,
            "profit": round(net_profit),
            "profit_pct": round(profit_pct, 2),
            "net_profit_pct": round(net_profit_pct, 2),
            "hold_days": len(df) - 1 - entry_day,
            "sentiment": round(last["sentiment"], 1),
        })

    sell_trades = [t for t in trades if t["type"] == "SELL"]
    wins = [t for t in sell_trades if t.get("profit", 0) > 0]
    win_rate = len(wins) / len(sell_trades) * 100 if sell_trades else 0

    by_reason = {}
    for t in sell_trades:
        r = t.get("reason", "기타")
        by_reason.setdefault(r, {"count": 0, "wins": 0})
        by_reason[r]["count"] += 1
        if t.get("profit", 0) > 0:
            by_reason[r]["wins"] += 1

    # 신호 유형별 통계
    buy_trades = [t for t in trades if t["type"] == "BUY"]
    by_signal = {}
    for t in buy_trades:
        s = t.get("signal", "기타")
        by_signal[s] = by_signal.get(s, 0) + 1

    # 총 거래 비용
    total_buy_cost  = sum(t.get("buy_cost", 0) for t in buy_trades)
    total_sell_cost = sum(round(t["shares"] * t["price"] * SELL_COST_PCT)
                          for t in sell_trades if "shares" in t)

    # 전략별 수익 통계
    by_strategy = {}
    for buy_t in buy_trades:
        sig = buy_t.get("signal", "기타")
        by_strategy.setdefault(sig, {"count": 0, "wins": 0, "total_pct": 0.0})
        by_strategy[sig]["count"] += 1
    for sell_t in sell_trades:
        sig = sell_t.get("strategy", "기타")
        if sig in by_strategy:
            by_strategy[sig]["total_pct"] += sell_t.get("net_profit_pct", 0)
            if sell_t.get("profit", 0) > 0:
                by_strategy[sig]["wins"] += 1

    # MDD 계산 (평균회귀 손절/익절도 반영)
    pv_list = []
    _cash, _shares, _entry, _entry_day, _streak, _sig = initial_cash, 0, 0, 0, 0, ""
    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        price = row["close"]
        hold = i - _entry_day
        if _shares > 0:
            _is_mr = (_sig == "평균회귀")
            _sl = MR_STOP_LOSS if _is_mr else STOP_LOSS_PCT
            _tp = MR_TAKE_PROFIT if _is_mr else TAKE_PROFIT_PCT
            if row["macd_hist"] < row["macd_hist_prev"]:
                _streak += 1
            else:
                _streak = 0
            loss_hit   = price <= _entry * (1 - _sl)
            profit_hit = price >= _entry * (1 + _tp)
            if _is_mr:
                signal_sell = hold >= MIN_HOLD_DAYS and (
                    row["bb_mid_recovery"] or row["rsi"] >= 55 or _streak >= MACD_SELL_DAYS)
            else:
                signal_sell = hold >= MIN_HOLD_DAYS and (
                    row["rsi"] > 70 or row["bb_pct"] > 0.85 or _streak >= MACD_SELL_DAYS)
            if loss_hit or profit_hit or signal_sell:
                _cash += _shares * price * (1 - SELL_COST_PCT)
                _shares = 0
                _streak = 0
                _sig = ""
        if _shares == 0 and row["buy_signal"]:
            qty = int((_cash * 0.3 * (1 - BUY_COST_PCT)) // price)
            if qty >= 1:
                _shares = qty
                _entry = price
                _entry_day = i
                _sig = row["signal_type"]
                _cash -= _shares * price * (1 + BUY_COST_PCT)
        pv_list.append(_cash + _shares * price)

    pv = pd.Series(pv_list)
    mdd = ((pv - pv.cummax()) / pv.cummax() * 100).min()

    return {
        "initial_cash": initial_cash,
        "final_value": int(cash),
        "total_return": round((cash / initial_cash - 1) * 100, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": len(sell_trades),
        "win_trades": len(wins),
        "max_drawdown": round(mdd, 2),
        "total_cost": total_buy_cost + total_sell_cost,
        "by_reason": by_reason,
        "by_signal": by_signal,
        "by_strategy": by_strategy,
        "trades": trades,
    }

# ─────────────────────────────────────────
# 6. 결과 출력
# ─────────────────────────────────────────
def print_results(symbol: str, result: dict, start: str, end: str):
    r = result
    profit = r["final_value"] - r["initial_cash"]
    sign = "+" if profit >= 0 else ""

    print(f"\n{'='*60}")
    print(f"  백테스트 결과: {symbol}  ({start} ~ {end})")
    print(f"{'='*60}")
    print(f"  초기 자금:       {r['initial_cash']:>15,}원")
    print(f"  최종 자산:       {r['final_value']:>15,}원")
    print(f"  손익:            {sign}{profit:>14,}원")
    print(f"  총 수익률:       {sign}{r['total_return']:>13}%")
    print(f"  총 거래 비용:    {r['total_cost']:>15,}원  ← 수수료+세금 포함")
    print(f"{'─'*60}")
    print(f"  총 거래 횟수:    {r['total_trades']:>15}회")
    print(f"  승리 거래:       {r['win_trades']:>15}회")
    print(f"  승률:            {r['win_rate']:>14}%")
    print(f"  최대 낙폭:       {r['max_drawdown']:>14}%")
    print(f"{'─'*60}")
    print(f"  청산 유형별:")
    for reason, stat in r["by_reason"].items():
        wr = stat["wins"] / stat["count"] * 100 if stat["count"] else 0
        print(f"    {reason:6s}: {stat['count']:3}회  승률 {wr:.0f}%")
    print(f"{'─'*60}")
    print(f"  전략별 성과:")
    for sig, stat in r["by_strategy"].items():
        if stat["count"] == 0:
            continue
        wr = stat["wins"] / stat["count"] * 100
        avg_pct = stat["total_pct"] / stat["count"]
        s = "+" if avg_pct >= 0 else ""
        print(f"    {sig:8s}: {stat['count']}회  승률 {wr:.0f}%  평균 {s}{avg_pct:.2f}%")
    print(f"{'='*60}")

    print(f"\n{'─'*60}")
    print(f"  거래 내역")
    print(f"{'─'*60}")
    for t in r["trades"]:
        if t["type"] == "BUY":
            ret5 = t.get("ret_5d", 0)
            ret5_str = f"  5일수익률={ret5:+.1f}%" if t.get("signal") == "평균회귀" else ""
            print(f"  🟢 [{t.get('signal','?')}] 매수 {t['date']}  {t['price']:,}원 × {t['shares']}주  (수수료 {t['buy_cost']:,}원)")
            print(f"       RSI={t['rsi']} MACD={t['macd_hist']} BB={t['bb_pct']}% 거래량={t['vol_ratio']}x{ret5_str}")
            print(f"       시장심리: {t['sentiment']} — {t['sentiment_label']}")
        else:
            emoji = "✅" if t.get("profit", 0) >= 0 else "❌"
            s1 = "+" if t.get("profit_pct", 0) >= 0 else ""
            s2 = "+" if t.get("net_profit_pct", 0) >= 0 else ""
            strat = f"[{t.get('strategy','')}] " if t.get("strategy") else ""
            print(f"  🔴 {strat}매도 {t['date']}  {t['price']:,}원  "
                  f"{emoji} 총{s1}{t['profit_pct']}% / 비용후{s2}{t['net_profit_pct']}%  "
                  f"[{t['reason']} / {t['hold_days']}일]")
            print()

# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "005930"
    start  = sys.argv[2] if len(sys.argv) > 2 else "20240101"
    end    = sys.argv[3] if len(sys.argv) > 3 else datetime.now().strftime("%Y%m%d")

    print(f"\n📊 {symbol} 백테스트 시작 ({start} ~ {end})")
    print(f"   [추세추종] 눌림목/BB반등/MACD골든 | 손절{STOP_LOSS_PCT*100:.0f}% | 익절{TAKE_PROFIT_PCT*100:.0f}%")
    print(f"   [평균회귀] 급락+과매도+패닉셀    | 손절{MR_STOP_LOSS*100:.0f}% | 익절{MR_TAKE_PROFIT*100:.0f}%")
    print(f"   청산: MACD {MACD_SELL_DAYS}일 연속 감소 / BB중간선회복(평균회귀) | 거래비용 왕복 {(BUY_COST_PCT+SELL_COST_PCT)*100:.3f}%")

    df = get_historical_data(symbol, start, end)
    print(f"  종목 데이터: {len(df)}일치")

    sentiment_df = calc_market_sentiment(start, end)

    df = calc_indicators(df)
    df = generate_signals(df, sentiment_df)

    result = run_backtest(df)
    print_results(symbol, result, start, end)
