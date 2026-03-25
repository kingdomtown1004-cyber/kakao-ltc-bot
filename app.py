import os
import logging
import threading
import httpx
import anthropic

from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 환경변수
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"].strip()
SUPABASE_URL = os.environ["SUPABASE_URL"].strip()
SUPABASE_KEY = os.environ["SUPABASE_KEY"].strip()

ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

PDF_FILES = [
    "evaluation_questionnaire.pdf",           # 0.34MB ~15K 토큰
    "2026_home_benefit_indicators_freq_2.pdf", # 1.44MB ~64K 토큰
    "2026_home_care_indicators_freq_1.pdf",    # 2.05MB ~91K 토큰
    # 총 ~170K 토큰 (200K 한도 이내)
]

# 전역 파일 ID 캐시 (모듈 로드 시 초기화 → gunicorn에서도 실행됨)
file_ids = {}


def init_files():
    global file_ids
    logger.info("PDF 파일 준비 중...")
    file_ids = upload_pdfs()
    logger.info(f"✅ {len(file_ids)}개 PDF 준비 완료!")


def get_existing_files():
    existing = {}
    try:
        page = ai.beta.files.list()
        for f in page.data:
            existing[f.filename] = f.id
        logger.info(f"기존 파일 {len(existing)}개 발견")
    except Exception as e:
        logger.warning(f"기존 파일 조회 실패: {e}")
    return existing


def download_from_supabase(pdf_name):
    url = f"{SUPABASE_URL}/storage/v1/object/evaluation-pdf/{pdf_name}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
    }
    with httpx.Client(timeout=120.0) as http:
        resp = http.get(url, headers=headers)
        resp.raise_for_status()
        return resp.content


def upload_pdfs():
    existing = get_existing_files()
    ids = {}
    for pdf_name in PDF_FILES:
        if pdf_name in existing:
            logger.info(f"재사용: {pdf_name}")
            ids[pdf_name] = existing[pdf_name]
            continue
        try:
            pdf_bytes = download_from_supabase(pdf_name)
            uploaded = ai.beta.files.upload(
                file=(pdf_name, pdf_bytes, "application/pdf"),
            )
            ids[pdf_name] = uploaded.id
            logger.info(f"업로드 완료: {pdf_name}")
        except Exception as e:
            logger.error(f"업로드 실패: {pdf_name} - {e}")
    return ids


def ask_claude(question):
    content = []
    for pdf_name, file_id in file_ids.items():
        content.append({
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
            "title": pdf_name.replace(".pdf", "").replace("_", " "),
        })
    content.append({"type": "text", "text": question})

    response = ai.beta.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=(
            "당신은 노인장기요양보험 평가 전문가입니다. "
            "제공된 2026년 장기요양 평가 매뉴얼 문서를 바탕으로 "
            "정확하고 친절하게 한국어로 답변해주세요. "
            "문서에 없는 내용은 솔직하게 모른다고 말씀해주세요. "
            "답변 시 관련 지표명이나 매뉴얼 항목을 구체적으로 언급해주세요."
        ),
        messages=[{"role": "user", "content": content}],
        betas=["files-api-2025-04-14"],
    )
    return response.content[0].text


def send_callback(callback_url, answer):
    """카카오 콜백 URL로 답변 전송"""
    # 카카오 메시지 4000자 제한 처리
    if len(answer) > 4000:
        answer = answer[:3990] + "\n\n...(이하 생략)"

    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": answer
                    }
                }
            ]
        }
    }
    try:
        with httpx.Client(timeout=30.0) as http:
            resp = http.post(callback_url, json=payload)
            logger.info(f"콜백 전송 완료: {resp.status_code}")
    except Exception as e:
        logger.error(f"콜백 전송 실패: {e}")


def process_in_background(question, callback_url):
    """백그라운드에서 Claude 호출 후 콜백 전송"""
    try:
        logger.info(f"질문 처리 시작: {question[:50]}")
        answer = ask_claude(question)
        send_callback(callback_url, answer)
    except Exception as e:
        logger.error(f"처리 오류: {e}")
        send_callback(callback_url, "⚠️ 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")


@app.route("/skill", methods=["POST"])
def skill():
    data = request.get_json()

    # 사용자 발화 추출
    question = data.get("userRequest", {}).get("utterance", "")
    callback_url = data.get("userRequest", {}).get("callbackUrl", "")

    logger.info(f"질문: {question[:50]}")

    if not question:
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [{"simpleText": {"text": "질문을 입력해주세요."}}]
            }
        })

    if callback_url:
        # 비동기 처리: 즉시 응답 후 백그라운드에서 처리
        thread = threading.Thread(
            target=process_in_background,
            args=(question, callback_url),
            daemon=True
        )
        thread.start()

        return jsonify({
            "version": "2.0",
            "useCallback": True,
            "data": {
                "text": "답변을 생성하고 있습니다... ⏳\n잠시만 기다려 주세요."
            }
        })
    else:
        # 동기 처리 (callbackUrl 없는 경우)
        try:
            answer = ask_claude(question)
            if len(answer) > 4000:
                answer = answer[:3990] + "\n\n...(이하 생략)"
            return jsonify({
                "version": "2.0",
                "template": {
                    "outputs": [{"simpleText": {"text": answer}}]
                }
            })
        except Exception as e:
            logger.error(f"오류: {e}")
            return jsonify({
                "version": "2.0",
                "template": {
                    "outputs": [{"simpleText": {"text": "⚠️ 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}}]
                }
            })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "files_loaded": len(file_ids), "file_names": list(file_ids.keys())})


@app.route("/debug", methods=["GET"])
def debug():
    result = {}
    # 1. Anthropic Files API 테스트
    try:
        page = ai.beta.files.list()
        result["anthropic_files"] = [f.filename for f in page.data]
    except Exception as e:
        result["anthropic_error"] = str(e)

    # 2. Supabase 다운로드 테스트 (첫 번째 파일만)
    test_pdf = PDF_FILES[0]
    try:
        data = download_from_supabase(test_pdf)
        result["supabase_ok"] = True
        result["supabase_size"] = len(data)
    except Exception as e:
        result["supabase_error"] = str(e)

    result["file_ids_count"] = len(file_ids)

    # 3. Claude 호출 테스트
    if file_ids:
        try:
            answer = ask_claude("방문요양 평가지표 1번은 무엇인가요?")
            result["claude_ok"] = True
            result["claude_answer"] = answer[:200]
        except Exception as e:
            result["claude_error"] = str(e)

    return jsonify(result)


# gunicorn 포함 모든 실행 방식에서 PDF 초기화
init_files()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
