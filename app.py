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

# 급여 유형별 지표번호 → 일반 지표 매핑 테이블 (note 필드에서 자동 생성)
CARE_TYPE_MAP = {
    "요": {},  # 방문요양
    "목": {},  # 방문목욕
    "주": {},  # 주야간보호
    "간": {},  # 간호
    "단": {},  # 단기보호
    "복": {},  # 복지용구
}
for _ind in INDICATOR_DB:
    for _prefix in CARE_TYPE_MAP:
        for _m in re.finditer(rf'{_prefix}\s*\((\d+(?:,\s*\d+)*)\)', _ind.get("note", "")):
            for _n in _m.group(1).split(","):
                CARE_TYPE_MAP[_prefix][int(_n.strip())] = _ind

# 질문에서 급여 유형 접두어 감지
def detect_care_prefix(question: str) -> str:
    if any(k in question for k in ["주야간", "주간보호", "야간보호"]):
        return "주"
    if any(k in question for k in ["방문목욕", "목욕"]):
        return "목"
    if any(k in question for k in ["방문요양", "재가요양"]):
        return "요"
    if any(k in question for k in ["방문간호", "간호"]):
        return "간"
    if any(k in question for k in ["단기보호", "단기"]):
        return "단"
    if any(k in question for k in ["복지용구"]):
        return "복"
    return ""


def find_indicator(question: str):
    """질문에서 지표 번호 또는 이름으로 관련 지표 찾기 (급여유형별 번호 지원)"""
    num_match = re.search(r'지표\s*(\d+)번?|(\d+)\s*번\s*지표|지표\s*번호\s*(\d+)', question)
    if not num_match:
        num_match = re.search(r'\b(\d{1,2})\s*번\b', question)

    if num_match:
        no = int(next(g for g in num_match.groups() if g))
        # 급여 유형이 명시된 경우 유형별 매핑 우선 적용
        prefix = detect_care_prefix(question)
        if prefix and no in CARE_TYPE_MAP.get(prefix, {}):
            return CARE_TYPE_MAP[prefix][no]
        # 일반 지표 번호 매칭
        for ind in INDICATOR_DB:
            if ind["no"] == no:
                return ind

    # 이름 매칭
    for ind in INDICATOR_DB:
        if ind["name"] in question:
            return ind

    return None


def is_toc_line(text: str) -> bool:
    """목차 라인 여부 판단 (점선이 많으면 목차)"""
    dot_count = text.count('·') + text.count('…') + text.count('.')
    return dot_count > len(text) * 0.2


def clean_chunk(text: str) -> str:
    """목차 줄 제거 및 공백 정리"""
    lines = text.splitlines()
    cleaned = [l for l in lines if not is_toc_line(l) and len(l.strip()) > 1]
    return '\n'.join(cleaned).strip()


# 급여 유형별 관련 파일 매핑
FILE_GROUPS = {
    "방문요양": ["2026_home_care_evaluation_manual_1.pdf", "split/home_care_5.pdf", "split/home_care_6.pdf", "split/home_care_7.pdf", "attach_1_2026_home_care_manual_qa_case_v2.pdf"],
    "방문목욕": ["2026_home_bath_evaluation_manual_2.pdf"],
    "주야간보호": ["split/day_care_1.pdf","split/day_care_2.pdf","split/day_care_3.pdf","split/day_care_4.pdf","split/day_care_5.pdf","split/day_care_6.pdf","split/day_care_7.pdf","split/day_care_8.pdf","split/day_care_9.pdf","split/day_care_10.pdf"],
    "재가": ["2026_home_care_indicators_freq_1.pdf", "2026_home_care_benefit_manual_freq_2.pdf"],
}

def detect_care_type(question: str) -> list:
    """질문에서 급여 유형 감지 → 관련 파일 우선 검색"""
    if any(k in question for k in ["방문목욕", "목욕"]):
        return FILE_GROUPS["방문목욕"]
    if any(k in question for k in ["주야간", "주간보호", "야간보호", "데이케어"]):
        return FILE_GROUPS["주야간보호"]
    if any(k in question for k in ["방문요양", "요양보호사", "방문"]):
        return FILE_GROUPS["방문요양"]
    return []  # 전체 검색


def search_text(keywords: list, max_chars: int = 4000, priority_files: list = None) -> str:
    """전체 PDF 텍스트에서 키워드 관련 단락 수집 (목차 제외)"""
    # 우선 검색 텍스트 구성: priority_files 먼저, 나머지 뒤에
    if priority_files:
        priority_text = "\n\n".join(PDF_CONTENT.get(f, "") for f in priority_files if f in PDF_CONTENT)
        other_text = "\n\n".join(v for k, v in PDF_CONTENT.items() if k not in priority_files)
        search_source = priority_text + "\n\n" + other_text
    else:
        search_source = ALL_TEXT

    collected = []
    seen = set()
    for keyword in keywords:
        start = 0
        while True:
            idx = search_source.find(keyword, start)
            if idx < 0:
                break
            chunk_start = max(0, idx - 300)
            chunk_end = min(len(search_source), idx + 1500)  # 더 넓은 창
            raw = search_source[chunk_start:chunk_end]
            chunk = clean_chunk(raw)
            if len(chunk) < 50:
                start = idx + 1
                continue
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
    priority_files = detect_care_type(question)

    # 검색 키워드 구성
    keywords = []
    if ind:
        keywords.append(ind["name"])
        # criteria에서 핵심어 추출
        for c in ind["criteria"]:
            stripped = re.sub(r'^[①②③④⑤⑥⑦⑧⑨]', '', c).strip()
            # 괄호 안 설명 제거
            stripped = re.sub(r'\s*\(.*?\)', '', stripped).strip()
            if len(stripped) >= 2:
                keywords.append(stripped)
    # 질문에서 핵심 단어 추가
    question_words = re.findall(r'[가-힣]{2,}', question)
    keywords += [w for w in question_words if len(w) >= 3]
    # 중복 제거
    seen_kw = set()
    keywords = [k for k in keywords if not (k in seen_kw or seen_kw.add(k))]

    if ind:
        ind_parts = [
            f"[지표 {ind['no']}번: {ind['name']}]",
            f"평가기준: {', '.join(ind['criteria'])}",
        ]
        if ind.get("detail"):
            ind_parts.append(ind["detail"])
        ind_parts.append(f"적용 급여: {ind['note']}")
        ind_text = "\n".join(ind_parts)
    else:
        ind_text = ""

    # 전체 텍스트에서 관련 내용 검색 (최대 8000자, 우선 파일 먼저)
    relevant = search_text(keywords, max_chars=8000, priority_files=priority_files) if keywords else ""

    parts = []
    if ind_text:
        parts.append(ind_text)
    if relevant:
        parts.append(f"[관련 매뉴얼 내용]\n{relevant}")
    return "\n\n".join(parts) if parts else ""


SYSTEM_DETAIL = (
    "당신은 2026년 노인장기요양보험 평가 전문가입니다. "
    "반드시 제공된 [평가 자료]에 있는 내용만 사용하여 답변하세요. 자료에 없는 내용은 절대 추측하거나 만들어 내지 마세요. "
    "답변 형식: 📋 지표명과 번호로 시작 → ✅ 평가기준(각 항목별 번호/내용) → 🔍 확인방법 → ⚠️ 주의사항 순서로 구조화하세요. "
    "각 항목은 줄바꿈으로 구분하고, 실무에 바로 활용할 수 있도록 구체적으로 작성하세요. "
    "자료에서 확인되지 않는 내용은 '📌 자료에서 확인되지 않습니다'라고 명시하세요."
)

SYSTEM_FAST = (
    "당신은 2026년 노인장기요양보험 평가 전문가입니다. "
    "반드시 제공된 [평가 자료]에 있는 내용만 사용하세요. 자료에 없는 내용은 추측하지 마세요. "
    "핵심 평가기준과 확인방법 위주로 간결하게 답변하고, 자료에 없으면 '자료에서 확인되지 않습니다'라고 하세요."
)


def format_db_answer(ind: dict) -> str:
    """Claude 없이 DB에서 즉시 포맷된 답변 생성 (빠름, search 없음)"""
    lines = [f"📋 지표 {ind['no']}번: {ind['name']}", "", "✅ 평가기준"]
    for c in ind["criteria"]:
        lines.append(f"  {c}")
    if ind.get("detail"):
        lines.append("")
        lines.append(ind["detail"])
    lines.append("")
    lines.append(f"📌 적용 급여: {ind['note']}")
    return "\n".join(lines)


def is_simple_lookup(question: str) -> bool:
    """단순 지표 조회 질문 여부 (Claude 불필요)"""
    simple_patterns = [r'지표\s*\d+', r'\d+\s*번\s*지표', r'지표\s*번호\s*\d+']
    has_number = any(re.search(p, question) for p in simple_patterns)
    detail_words = ['조심', '주의', '어떻게', '방법', '이유', '왜', '설명', '자세히', '예시', '차이', '비교', '확인']
    has_detail = any(w in question for w in detail_words)
    return has_number and not has_detail


def ask_claude(question: str, detailed: bool = False) -> str:
    if detailed:
        context = build_context(question)
        system = f"{SYSTEM_DETAIL}\n\n[평가 자료]\n{context}"
        max_tok = 1500
        timeout = 60.0
    else:
        ind = find_indicator(question)
        # 단순 지표 조회는 Claude 없이 즉시 답변 (< 0.1초)
        if ind and is_simple_lookup(question):
            return format_db_answer(ind)
        # 상세 질문은 Claude 사용 (타임아웃 3.5초)
        keywords = []
        if ind:
            keywords.append(ind["name"])
            keywords += [c.lstrip("①②③④⑤").strip() for c in ind["criteria"]]
        keywords += [w for w in re.findall(r'[가-힣]{2,}', question) if len(w) >= 3]
        if ind:
            _parts = [
                f"[지표 {ind['no']}번: {ind['name']}]",
                f"평가기준: {', '.join(ind['criteria'])}",
            ]
            if ind.get("detail"):
                _parts.append(ind["detail"])
            _parts.append(f"적용 급여: {ind['note']}")
            ind_text = "\n".join(_parts)
        else:
            ind_text = ""
        priority_files = detect_care_type(question)
        relevant = search_text(keywords, max_chars=1500, priority_files=priority_files) if keywords else ""
        parts = []
        if ind_text:
            parts.append(ind_text)
        if relevant:
            parts.append(f"[관련 자료]\n{relevant}")
        context = "\n\n".join(parts)
        system = f"{SYSTEM_FAST}\n\n[평가 자료]\n{context}"
        max_tok = 350
        timeout = 3.5

    result_holder = []

    def _call():
        try:
            resp = ai.messages.create(
                model="claude-haiku-4-5",
                max_tokens=max_tok,
                system=system,
                messages=[{"role": "user", "content": question}],
            )
            result_holder.append(resp.content[0].text)
        except Exception as e:
            logger.error(f"Claude 오류: {e}")

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if result_holder:
        return result_holder[0]

    # 타임아웃 → DB 답변으로 대체
    logger.warning(f"Claude 타임아웃({timeout}s), DB 대체 답변 사용")
    ind2 = find_indicator(question)
    if ind2:
        return format_db_answer(ind2)
    return "⚠️ 잠시 응답이 지연되고 있습니다. 다시 질문해 주세요."


def get_answer(question: str, detailed: bool = False) -> str:
    return ask_claude(question, detailed=detailed)


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
        answer = get_answer(question, detailed=True)
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
            answer = get_answer(question, detailed=False)
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
