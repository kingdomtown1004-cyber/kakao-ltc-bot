"""
지표별 사전 답변 생성 스크립트 (v2 - 전체 자료 연계)
로컬에서 한 번 실행 → indicator_answers.json 저장
"""
import os, json, time, re, sys, anthropic

BASE_DIR = os.path.dirname(__file__)

def load_json(fname):
    with open(os.path.join(BASE_DIR, fname), "r", encoding="utf-8") as f:
        return json.load(f)

INDICATOR_DB = load_json("indicator_db.json")
PDF_CONTENT = load_json("pdf_content_clean.json")
ALL_TEXT = "\n\n".join(PDF_CONTENT.values())

# 파일 그룹 정의
MANUAL_FILES = {
    "방문요양": [
        "2026_home_care_evaluation_manual_1.pdf",
        "split/home_care_3.pdf",
        "split/home_care_4.pdf",
        "split/home_care_5.pdf",
        "split/home_care_6.pdf",
        "split/home_care_7.pdf",
    ],
    "방문목욕": ["2026_home_bath_evaluation_manual_2.pdf"],
    "주야간보호": [
        "split/day_care_1.pdf","split/day_care_2.pdf","split/day_care_3.pdf",
        "split/day_care_4.pdf","split/day_care_5.pdf","split/day_care_6.pdf",
        "split/day_care_7.pdf","split/day_care_8.pdf","split/day_care_9.pdf",
        "split/day_care_10.pdf",
    ],
}
QA_FILES = [
    "attach_1_2026_home_care_manual_qa_case_v2.pdf",
    "attach_2_2026_home_care_manual_qa_compare.pdf",
]
FREQ_FILES = [
    "2026_home_care_indicators_freq_1.pdf",
    "2026_home_benefit_indicators_freq_2.pdf",
    "2026_home_care_benefit_manual_freq_2.pdf",
]

def is_toc_line(text):
    dot_count = text.count('·') + text.count('…') + text.count('.')
    return dot_count > len(text) * 0.2

def clean_chunk(text):
    lines = text.splitlines()
    return '\n'.join(l for l in lines if not is_toc_line(l) and len(l.strip()) > 1).strip()

def search_in(source_text, keywords, max_chars):
    """source_text 내에서 키워드 관련 단락 수집"""
    collected = []
    seen = set()
    for keyword in keywords:
        start = 0
        while True:
            idx = source_text.find(keyword, start)
            if idx < 0:
                break
            chunk = clean_chunk(source_text[max(0, idx-300):min(len(source_text), idx+1800)])
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


def build_context(ind):
    """
    전체 자료를 연계한 풍부한 컨텍스트 구성:
    1. 매뉴얼 본문 (적용 급여유형별)
    2. Q&A 사례집
    3. 빈도/중요도 자료
    4. 비교 자료
    """
    name = ind["name"]
    note = ind.get("note", "")
    keywords = [name]
    for c in ind["criteria"]:
        stripped = re.sub(r'^[①②③④⑤⑥⑦⑧⑨]', '', c).strip()
        stripped = re.sub(r'\s*\(.*?\)', '', stripped).strip()
        if len(stripped) >= 2:
            keywords.append(stripped)

    # 1) 매뉴얼 본문 (급여유형별 우선)
    manual_sources = []
    if any(p in note for p in ["요", "목"]):
        manual_sources += MANUAL_FILES["방문요양"] + MANUAL_FILES["방문목욕"]
    if "주" in note or "간" in note or "단" in note:
        manual_sources += MANUAL_FILES["주야간보호"]
    if not manual_sources:
        manual_sources = MANUAL_FILES["방문요양"] + MANUAL_FILES["방문목욕"] + MANUAL_FILES["주야간보호"]

    manual_text = "\n\n".join(PDF_CONTENT.get(f, "") for f in manual_sources if f in PDF_CONTENT)
    manual_ctx = search_in(manual_text, keywords, max_chars=7000)

    # 2) Q&A 사례집
    qa_text = "\n\n".join(PDF_CONTENT.get(f, "") for f in QA_FILES if f in PDF_CONTENT)
    qa_ctx = search_in(qa_text, keywords, max_chars=4000)

    # 3) 빈도/중요도 자료
    freq_text = "\n\n".join(PDF_CONTENT.get(f, "") for f in FREQ_FILES if f in PDF_CONTENT)
    freq_ctx = search_in(freq_text, keywords, max_chars=3000)

    # 지표 DB 정보
    ind_text = (
        f"[지표 {ind['no']}번: {ind['name']}]\n"
        f"평가기준: {', '.join(ind['criteria'])}\n"
        f"적용 급여유형: {note}"
    )
    if ind.get("detail"):
        ind_text += f"\n상세내용: {ind['detail']}"

    parts = [ind_text]
    if manual_ctx:
        parts.append(f"[매뉴얼 본문]\n{manual_ctx}")
    if qa_ctx:
        parts.append(f"[Q&A 사례집]\n{qa_ctx}")
    if freq_ctx:
        parts.append(f"[빈도·중요도 자료]\n{freq_ctx}")

    return "\n\n" + ("="*40) + "\n\n".join(parts)


SYSTEM = """당신은 2026년 노인장기요양보험 평가 전문가입니다.

반드시 제공된 자료(매뉴얼 본문, Q&A 사례집, 빈도·중요도 자료)를 모두 종합하여 답변하세요.
자료에 없는 내용은 절대 추측하지 마세요. 마크다운 표(|)는 사용하지 마세요.

답변 형식 (카카오톡 메시지용, 이모지와 줄바꿈 사용):

📋 지표번호 지표명 (적용 급여유형, 점수)

✅ 평가기준
각 기준 항목별로 구체적으로 설명
(필수기록사항, 인정/불인정 범위 포함)

🔍 확인방법
평가 시 실제 확인 절차와 방법

⚠️ 주의사항
자주 틀리는 점, 불인정 사례, Q&A에서 나온 실수 유형

💡 실무 팁
현장에서 바로 활용할 수 있는 준비 방법과 체크포인트
(빈도·중요도 자료에서 확인된 핵심 포인트 포함)"""


def generate_answer(ai, ind):
    context = build_context(ind)
    question = (
        f"지표 {ind['no']}번 {ind['name']}에 대해 "
        f"매뉴얼 본문, Q&A 사례집, 빈도·중요도 자료를 모두 종합하여 "
        f"평가기준, 확인방법, 주의사항, 실무 팁을 자세히 알려주세요."
    )
    response = ai.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=f"{SYSTEM}\n\n[평가 자료]\n{context}",
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def main():
    force = "--force" in sys.argv
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        print("ANTHROPIC_API_KEY 환경변수를 설정해주세요")
        return

    ai = anthropic.Anthropic(api_key=api_key)
    answers = {}
    out_path = os.path.join(BASE_DIR, "indicator_answers.json")

    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            answers = json.load(f)
        print(f"기존 파일 로드: {len(answers)}개")

    total = len(INDICATOR_DB)
    for i, ind in enumerate(INDICATOR_DB):
        key = str(ind["no"])
        if key in answers and not force:
            print(f"[{i+1}/{total}] 지표 {ind['no']}번 {ind['name']} - 스킵")
            continue

        print(f"[{i+1}/{total}] 지표 {ind['no']}번 {ind['name']} 생성 중...")
        try:
            answer = generate_answer(ai, ind)
            answers[key] = answer
            print(f"  → {len(answer)}자")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(answers, f, ensure_ascii=False, indent=2)

            time.sleep(1)
        except Exception as e:
            print(f"  → 오류: {e}")
            time.sleep(3)

    print(f"\n완료! 총 {len(answers)}개 저장: {out_path}")


if __name__ == "__main__":
    main()
