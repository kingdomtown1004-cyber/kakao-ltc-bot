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

# 미리 추출한 PDF 텍스트 로드 (약 25K 토큰)
def load_pdf_text():
    path = os.path.join(os.path.dirname(__file__), "pdf_content_clean.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 큰 파일(21K토큰) 제외, 작은 2개만 사용 → 총 ~11.5K 토큰 (5초 이내)
        use_files = [
            "2026_home_benefit_indicators_freq_2.pdf",
            "evaluation_questionnaire.pdf",
        ]
        combined = ""
        for fname in use_files:
            if fname in data:
                label = fname.replace(".pdf", "").replace("_", " ")
                combined += f"\n\n=== {label} ===\n{data[fname]}"
        logger.info(f"PDF 텍스트 로드 완료: {len(combined)}자")
        return combined
    except Exception as e:
        logger.error(f"PDF 텍스트 로드 실패: {e}")
        return ""

PDF_TEXT = load_pdf_text()

SYSTEM_PROMPT = (
    "당신은 노인장기요양보험 평가 전문가입니다. "
    "아래에 제공된 2026년 장기요양 평가 매뉴얼 Q&A 자료를 바탕으로 "
    "정확하고 친절하게 한국어로 답변해주세요. "
    "자료에 명시된 내용은 구체적으로 인용하고, "
    "자료에 없는 내용은 솔직하게 모른다고 말씀해주세요.\n\n"
    f"[평가 매뉴얼 자료]\n{PDF_TEXT}"
)


def ask_claude(question):
    response = ai.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def send_callback(callback_url, answer):
    import httpx
    if len(answer) > 4000:
        answer = answer[:3990] + "\n\n...(이하 생략)"
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": answer}}]
        }
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
        thread = threading.Thread(
            target=process_in_background,
            args=(question, callback_url),
            daemon=True
        )
        thread.start()
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
    return jsonify({"status": "ok", "pdf_chars": len(PDF_TEXT)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
