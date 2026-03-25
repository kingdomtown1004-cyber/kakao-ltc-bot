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

# 전체 PDF 텍스트 (6개 파일, 128K자) - 검색용
PDF_CONTENT = load_json("pdf_content_clean.json") or {}
# 전체 텍스트를 하나로 합침 (키워드 검색용)
ALL_TEXT = "\n\n".join(PDF_CONTENT.values())
QUESTIONNAIRE_TEXT = PDF_CONTENT.get("evaluation_questionnaire.pdf", "")

total_chars = sum(len(v) for v in PDF_CONTENT.values())
logger.info(f"지표 DB: {len(INDICATOR_DB)}개 / PDF 전체: {total_chars}자 ({len(PDF_CONTENT)}개 파일)")


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


def search_text(keywords: list, max_chars: int = 4000) -> str:
    """전체 PDF 텍스트에서 키워드 관련 단락 수집"""
    collected = []
    seen = set()
    for keyword in keywords:
        start = 0
        while True:
            idx = ALL_TEXT.find(keyword, start)
            if idx < 0:
                break
            # 앞뒤 문단 포함
            chunk_start = max(0, idx - 200)
            chunk_end = min(len(ALL_TEXT), idx + 800)
            chunk = ALL_TEXT[chunk_start:chunk_end].strip()
            key = chunk[:80]
            if key not in seen:
                seen.add(key)
                collected.append(chunk)
            start = idx + 1
            if sum(len(c) for c in collected) > max_chars:
                break
        if sum(len(c) for c in collected) > max_chars:
            break
    return "\n\n---\n\n".join(collected)[:max_chars]


def build_context(question: str) -> str:
    """질문에 맞는 관련 내용 수집"""
    ind = find_indicator(question)

    # 검색 키워드 구성
    keywords = []
    if ind:
        keywords.append(ind["name"])
        keywords += [c.lstrip("①②③④⑤").strip() for c in ind["criteria"]]
    # 질문에서 핵심 단어 추가 (2글자 이상 한국어 단어)
    question_words = re.findall(r'[가-힣]{2,}', question)
    keywords += [w for w in question_words if len(w) >= 3]

    if ind:
        ind_text = (
            f"[지표 {ind['no']}번: {ind['name']}]\n"
            f"평가기준: {', '.join(ind['criteria'])}\n"
            f"적용 급여: {ind['note']}\n"
        )
    else:
        ind_text = ""

    # 전체 텍스트에서 관련 내용 검색 (최대 4000자)
    relevant = search_text(keywords, max_chars=4000) if keywords else ""

    parts = []
    if ind_text:
        parts.append(ind_text)
    if relevant:
        parts.append(f"[관련 매뉴얼 내용]\n{relevant}")
    return "\n\n".join(parts) if parts else ""


SYSTEM_BASE = (
    "당신은 노인장기요양보험 평가 전문가입니다. "
    "제공된 2026년 장기요양 평가 자료를 바탕으로 "
    "정확하고 친절하게 한국어로 답변해주세요. "
    "자료에 명시된 내용은 구체적으로 인용하고, "
    "자료에 없는 내용은 솔직하게 모른다고 말씀해주세요."
)


def format_indicator_answer(ind: dict, question: str) -> str:
    """지표 DB에서 즉시 포맷된 답변 생성 (Claude 없이)"""
    lines = [
        f"📋 지표 {ind['no']}번: {ind['name']}",
        "",
        "✅ 평가기준",
    ]
    for c in ind["criteria"]:
        lines.append(f"  {c}")
    lines.append("")
    lines.append(f"📌 적용 급여: {ind['note']}")

    # 면담 예시 추가 (전체 텍스트에서 검색)
    qa_section = search_text([ind["name"]] + [c.lstrip("①②③④⑤").strip() for c in ind["criteria"]], max_chars=1500)
    if qa_section:
        lines.append("")
        lines.append("💬 관련 Q&A")
        lines.append(qa_section[:1200])

    return "\n".join(lines)


def ask_claude(question: str) -> str:
    context = build_context(question)
    system = f"{SYSTEM_BASE}\n\n[평가 자료]\n{context}"
    response = ai.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def get_answer(question: str) -> str:
    """지표 직접 조회 → 즉시 응답, 아니면 Claude 호출"""
    ind = find_indicator(question)
    if ind and ind["criteria"]:
        # 지표 번호/이름이 명확히 매칭되면 DB에서 즉시 응답
        return format_indicator_answer(ind, question)
    # 일반 질문은 Claude 호출
    return ask_claude(question)


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
        answer = get_answer(question)
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
            answer = get_answer(question)
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
