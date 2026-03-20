# 협업 가이드라인

## 처음 세팅하는 법

```bash
git clone https://github.com/yelim8902/trading-bot_KIS.git
cd trading-bot_KIS
pip install requests python-dotenv anthropic pykrx pandas

# .env.example 복사해서 본인 API 키 넣기
cp .env.example .env
# .env 열어서 본인 KIS키 / Anthropic키 입력
```

`.env` 파일은 **절대 커밋하지 말 것** — API 키 유출 위험

---

## 브랜치 전략

```
main          ← 검증된 안정 버전만 (대회 실전용)
dev-yelim     ← yelim 작업 브랜치
dev-{상대방}  ← 상대방 작업 브랜치
```

### 작업 흐름

```bash
# 1. 작업 시작 전 항상 main 최신화
git checkout main
git pull origin main

# 2. 내 브랜치 만들어서 작업
git checkout -b dev-yelim

# 3. 작업 후 커밋
git add bot.py
git commit -m "feat: 손절 조건 -3% → -5%로 변경 (백테스트 MDD 개선)"

# 4. 푸시
git push origin dev-yelim

# 5. GitHub에서 Pull Request 생성 → 상대방 리뷰 후 main 머지
```

---

## 커밋 메시지 규칙

| 태그 | 언제 |
|------|------|
| `feat:` | 새 기능 추가 |
| `fix:` | 버그 수정 |
| `strategy:` | 매매 전략 변경 |
| `backtest:` | 백테스트 결과 반영 |
| `refactor:` | 코드 정리 (기능 변화 없음) |
| `docs:` | 문서 수정 |

**예시**
```
strategy: 1차 스크리닝 RSI 상한 65 → 60으로 강화
backtest: 삼성전자 MA50 필터 추가, MDD -9.7% → -2.7% 개선
fix: 장 외 시간 주문 오류 수정
```

전략 바꿀 때는 **백테스트 결과를 커밋 메시지나 PR에 꼭 기록** — 왜 바꿨는지 근거 남기기

---

## 역할 분담 (제안)

| 역할 | 내용 |
|------|------|
| 전략 개발 | 지표 조건 튜닝, 새 전략 아이디어 |
| 백테스트 검증 | `backtest.py`로 성과 확인 후 반영 |
| 봇 운영 | 실행/모니터링/결과 기록 |

---

## 절대 규칙

1. `.env` 커밋 금지
2. `main`에 직접 푸시 금지 — 항상 브랜치 → PR → 머지
3. 전략 변경은 백테스트 결과 확인 후 반영
4. 모의투자 충분히 테스트 후 실전 전환
