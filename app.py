import os
import json
import logging
import threading
import re
import anthropic

from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"].strip()
ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

BASE_DIR = os.path.dirname(__file__)


def load_json(fname):
    try:
        with open(os.path.join(BASE_DIR, fname), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"{fname} 로드 실패: {e}")
        return None


# 지표 DB (30개 지표 구조화 정보)
INDICATOR_DB = load_json("indicator_db.json") or []

# 종사자 면담 예시 (4.8K 토큰 - 작은 파일)
PDF_CONTENT = load_json("pdf_content_clean.json") or {}
QUESTIONNAIRE_TEXT = PDF_CONTENT.get("evaluation_questionnaire.pdf", "")

logger.info(f"지표 DB: {len(INDICATOR_DB)}개 / 면담예시: {len(QUESTIONNAIRE_TEXT)}자")


def find_indicator(question: str):
    """질문에서 지표 번호 또는 이름으로 관련 지표 찾기"""
    # 번호 패턴: "1번", "지표1", "지표 1" 등
    num_match = re.search(r'지표\s*(\d+)번?|(\d+)\s*번\s*지표|지표\s*번호\s*(\d+)', question)
    if not num_match:
        num_match = re.search(r'\b(\d{1,2})\s*번\b', question)

    if num_match:
        no = int(next(g for g in num_match.groups() if g))
        for ind in INDICATOR_DB:
            if ind["no"] == no:
                return ind

    # 이름 매칭
    for ind in INDICATOR_DB:
        if ind["name"] in question:
            return ind

    return None


def build_context(question: str) -> str:
    """질문에 맞는 최소 컨텍스트 생성"""
    ind = find_indicator(question)

    if ind:
        # 지표 정보 (200자 미만)
        ind_text = (
            f"[지표 {ind['no']}번: {ind['name']}]\n"
            f"평가기준: {', '.join(ind['criteria'])}\n"
            f"적용 급여: {ind['note']}\n"
        )
    else:
        ind_text = ""

    # 관련 면담 예시 섹션 추출 (최대 3000자)
    if ind and QUESTIONNAIRE_TEXT:
        # 지표 이름으로 관련 섹션 찾기
        keyword = ind["name"]
        idx = QUESTIONNAIRE_TEXT.find(keyword)
        if idx >= 0:
            start = max(0, idx - 100)
            end = min(len(QUESTIONNAIRE_TEXT), idx + 3000)
            qa_text = QUESTIONNAIRE_TEXT[start:end]
        else:
            qa_text = QUESTIONNAIRE_TEXT[:3000]
    else:
        qa_text = QUESTIONNAIRE_TEXT[:3000]

    return f"{ind_text}\n[종사자 면담 예시]\n{qa_text}"


SYSTEM_BASE = (
    "당신은 노인장기요양보험 평가 전문가입니다. "
    "제공된 2026년 장기요양 평가 자료를 바탕으로 "
    "정확하고 친절하게 한국어로 답변해주세요. "
    "자료에 명시된 내용은 구체적으로 인용하고, "
    "자료에 없는 내용은 솔직하게 모른다고 말씀해주세요."
)


def ask_claude(question: str) -> str:
    context = build_context(question)
    system = f"{SYSTEM_BASE}\n\n[평가 자료]\n{context}"
    response = ai.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def send_callback(callback_url, answer):
    import httpx
    if len(answer) > 4000:
        answer = answer[:3990] + "\n\n...(이하 생략)"
    payload = {
        "version": "2.0",
        "template": {"outputs": [{"simpleText": {"text": answer}}]}
    }
    try:
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(callback_url, json=payload)
            logger.info(f"콜백 전송 완료: {resp.status_code}")
    except Exception as e:
        logger.error(f"콜백 전송 실패: {e}")


def process_in_background(question, callback_url):
    try:
        answer = ask_claude(question)
        send_callback(callback_url, answer)
    except Exception as e:
        logger.error(f"처리 오류: {e}")
        send_callback(callback_url, "⚠️ 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


@app.route("/skill", methods=["POST"])
def skill():
    data = request.get_json()
    question = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl", "")
    logger.info(f"질문: {question[:50]}")

    if not question:
        return jsonify({
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": "질문을 입력해주세요."}}]}
        })

    if callback_url:
        threading.Thread(
            target=process_in_background,
            args=(question, callback_url),
            daemon=True
        ).start()
        return jsonify({
            "version": "2.0",
            "useCallback": True,
            "data": {"text": "답변을 생성하고 있습니다... ⏳\n잠시만 기다려 주세요."}
        })
    else:
        try:
            answer = ask_claude(question)
            if len(answer) > 4000:
                answer = answer[:3990] + "\n\n...(이하 생략)"
            return jsonify({
                "version": "2.0",
                "template": {"outputs": [{"simpleText": {"text": answer}}]}
            })
        except Exception as e:
            logger.error(f"오류: {e}")
            return jsonify({
                "version": "2.0",
                "template": {"outputs": [{"simpleText": {"text": f"⚠️ 오류: {str(e)[:300]}"}}]}
            })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "indicators": len(INDICATOR_DB),
        "questionnaire_chars": len(QUESTIONNAIRE_TEXT),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
