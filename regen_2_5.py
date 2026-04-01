"""지표 2(운영규정), 5(직원교육) 캐시 재생성"""
import os, json, sys, anthropic

sys.stdout.reconfigure(encoding='utf-8')
BASE = os.path.dirname(__file__)

def load(f):
    with open(os.path.join(BASE, f), encoding='utf-8') as fp:
        return json.load(fp)

db   = load("indicator_db.json")
pdf  = load("pdf_content_clean.json")
cache = load("indicator_answers.json")

api_key = os.environ.get("ANTHROPIC_API_KEY","").strip()
if not api_key:
    print("ANTHROPIC_API_KEY 환경변수를 설정해주세요")
    sys.exit(1)

ai = anthropic.Anthropic(api_key=api_key)

MANUAL_FILES = [
    "2026_home_care_evaluation_manual_1.pdf",
    "split/home_care_3.pdf","split/home_care_4.pdf",
    "split/home_care_5.pdf","split/home_care_6.pdf","split/home_care_7.pdf",
    "2026_home_bath_evaluation_manual_2.pdf",
    "split/day_care_1.pdf","split/day_care_2.pdf","split/day_care_3.pdf",
    "split/day_care_4.pdf","split/day_care_5.pdf","split/day_care_6.pdf",
    "split/day_care_7.pdf","split/day_care_8.pdf",
]
QA_FILES  = ["attach_1_2026_home_care_manual_qa_case_v2.pdf"]
FREQ_FILES = ["2026_home_care_indicators_freq_1.pdf","2026_home_care_benefit_manual_freq_2.pdf"]

def get_text(files, max_chars):
    parts = [pdf.get(f,"") for f in files if f in pdf]
    return "\n\n".join(parts)[:max_chars]

def search_keyword(keyword, source, window=1000):
    results = []
    idx = 0
    while True:
        i = source.find(keyword, idx)
        if i < 0: break
        results.append(source[max(0,i-200):i+window])
        idx = i + 1
        if len(results) >= 5: break
    return "\n---\n".join(results)

def build_ctx(ind):
    name = ind['name']
    manual_src = get_text(MANUAL_FILES, 200000)
    qa_src     = get_text(QA_FILES, 100000)
    freq_src   = get_text(FREQ_FILES, 100000)

    manual_hits = search_keyword(name, manual_src, 2000)
    qa_hits     = search_keyword(name, qa_src, 1500)
    freq_hits   = search_keyword(name, freq_src, 1000)

    parts = [
        f"[지표 {ind['no']}번: {name}]",
        f"평가기준: {', '.join(ind['criteria'])}",
    ]
    if ind.get('detail'):
        parts.append(f"[항목 상세]\n{ind['detail']}")
    parts.append(f"적용급여: {ind['note']}")
    if manual_hits:
        parts.append(f"[매뉴얼 내용]\n{manual_hits[:7000]}")
    if qa_hits:
        parts.append(f"[Q&A 사례]\n{qa_hits[:4000]}")
    if freq_hits:
        parts.append(f"[빈도·통계 자료]\n{freq_hits[:3000]}")
    return "\n\n".join(parts)

SYSTEM = (
    "당신은 2026년 노인장기요양보험 평가 전문가입니다. "
    "아래 [평가 자료]를 바탕으로 해당 지표에 대한 실무 중심 상세 답변을 작성하세요. "
    "반드시 포함할 내용: "
    "① 지표번호·이름 ② 평가기준 각 항목 상세 ③ 포함해야 할 모든 세부 항목(번호·목록) "
    "④ 확인방법 ⑤ 주의사항 ⑥ 실무 팁. "
    "특히 '몇 개', '목록', '항목' 관련 질문이 올 때 즉시 답변할 수 있도록 "
    "항목 목록을 빠짐없이 작성하세요. 자료에 없는 내용은 추측하지 마세요."
)

for no in [2, 5]:
    ind = next((x for x in db if x['no']==no), None)
    if not ind:
        print(f"지표 {no} 없음"); continue

    print(f"\n지표 {no}번 [{ind['name']}] 재생성 중...")
    ctx = build_ctx(ind)
    resp = ai.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=f"{SYSTEM}\n\n[평가 자료]\n{ctx}",
        messages=[{"role":"user","content":f"지표 {no}번 {ind['name']}에 대해 실무자가 바로 활용할 수 있도록 상세히 설명해주세요. 특히 포함해야 할 항목 목록을 빠짐없이 나열해주세요."}],
    )
    answer = resp.content[0].text
    cache[str(no)] = answer
    print(f"완료 ({len(answer)}자)")

with open(os.path.join(BASE,"indicator_answers.json"), 'w', encoding='utf-8') as f:
    json.dump(cache, f, ensure_ascii=False, indent=2)
print("\n저장 완료: indicator_answers.json")
