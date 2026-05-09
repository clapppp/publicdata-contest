"""
rag.py — 채용공고 RAG (ChromaDB)
- 임베딩: BAAI/bge-m3 (CPU)
- 저장: PersistentClient (./chroma_db)
- 데이터: 한국노인인력개발원 100세누리 구인정보 OpenAPI
         → 60세 이상 시니어 대상 민간 일자리 (일 1회 갱신)
"""
import time
from xml.etree import ElementTree as ET

import requests
import chromadb
from chromadb.utils import embedding_functions

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "job_postings"
DB_PATH = "./chroma_db"

# 한국노인인력개발원 100세누리 구인정보 채용목록 API
# 가이드: OpenAPI활용가이드(한국노인인력개발원_100세누리구인정보)_v1.1.docx
SENURI_BASE = "https://apis.data.go.kr/B552474/SenuriService/getJobList"
SENURI_API_KEY = "b989a98fd4bd839b1a1e8f8f3fe5dd428629d05a4b3d1ba3d13d0317e5d5368f"

# 고용형태 코드 (응답엔 emplymShpNm으로 한글이 같이 옴 — 참고용)
EMPLOYMENT_TYPES = {
    "CM0101": "정규직",
    "CM0102": "계약직",
    "CM0103": "시간제일자리",
    "CM0104": "일당직",
    "CM0105": "기타",
}

# 워크넷 API 키 없을 때 폴백용 샘플 (Colab 검증 시 사용한 8건)
SAMPLE_JOBS = [
    {"id": "JOB001", "title": "아파트 경비원", "company": "행복아파트 관리사무소",
     "location": "서울 강남구", "work_type": "상근",
     "description": "아파트 출입 관리, 주차 안내, 택배 수령 업무. 야간근무 없음. 60세 이상 우대.",
     "age_friendly": True, "physical_intensity": "low"},
    {"id": "JOB002", "title": "학교 급식 보조원", "company": "서울 강남초등학교",
     "location": "서울 강남구", "work_type": "시간제",
     "description": "급식 조리 보조, 식기 세척, 배식 지원. 오전 10시~오후 3시 근무. 주 5일.",
     "age_friendly": True, "physical_intensity": "medium"},
    {"id": "JOB003", "title": "편의점 야간 계산원", "company": "GS25 강남점",
     "location": "서울 강남구", "work_type": "시간제",
     "description": "심야 계산 및 진열 업무. 오전 12시~오전 6시 근무.",
     "age_friendly": False, "physical_intensity": "low"},
    {"id": "JOB004", "title": "실버 돌봄 도우미", "company": "강남구 노인복지관",
     "location": "서울 강남구", "work_type": "상근",
     "description": "어르신 생활 지원, 말벗, 외출 동행. 요양보호사 자격증 우대. 60세 이상 가능.",
     "age_friendly": True, "physical_intensity": "low"},
    {"id": "JOB005", "title": "주차 관리원", "company": "코엑스 주차장",
     "location": "서울 강남구", "work_type": "교대",
     "description": "차량 입출차 관리, 요금 정산. 주간/야간 교대 근무.",
     "age_friendly": True, "physical_intensity": "low"},
    {"id": "JOB006", "title": "농산물 포장 작업원", "company": "강동농협",
     "location": "서울 강동구", "work_type": "일용직",
     "description": "농산물 선별 및 포장 작업. 서있는 작업 많음. 체력 필요.",
     "age_friendly": False, "physical_intensity": "high"},
    {"id": "JOB007", "title": "도서관 사서 보조", "company": "강남구립도서관",
     "location": "서울 강남구", "work_type": "시간제",
     "description": "도서 정리, 반납 처리, 이용자 안내. 조용한 실내 환경. 주 4일 근무.",
     "age_friendly": True, "physical_intensity": "low"},
    {"id": "JOB008", "title": "전통시장 안내 도우미", "company": "강남구청",
     "location": "서울 강남구", "work_type": "시간제",
     "description": "전통시장 방문객 안내, 지역 홍보. 오전 9시~오후 1시. 지역 어르신 우선 채용.",
     "age_friendly": True, "physical_intensity": "low"},
]

_client = None
_collection = None


SKIP_FETCH_THRESHOLD = 100  # 컬렉션에 이 수 이상이면 API 수집 스킵


def setup():
    """
    ChromaDB 초기화.
    - 컬렉션이 SKIP_FETCH_THRESHOLD(100)건 이상이면 기존 데이터 그대로 사용.
    - 미만이거나 없으면 컬렉션 재생성 후 100세누리 API 수집.
    """
    global _client, _collection

    print(f"📦 임베딩 모델 로드: {EMBED_MODEL} (CPU)")
    print(f"   ↳ 첫 실행 시 ~570MB 다운로드 (이후엔 캐시)")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu",
    )
    print(f"   ↳ 모델 준비 완료")

    _client = chromadb.PersistentClient(path=DB_PATH)

    # 기존 컬렉션 확인
    try:
        _collection = _client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )
        count = _collection.count()
        if count >= SKIP_FETCH_THRESHOLD:
            print(f"✅ 기존 컬렉션 재사용 ({count}건 ≥ {SKIP_FETCH_THRESHOLD}) — API 수집 스킵")
            return
        print(f"  기존 컬렉션 {count}건 ({SKIP_FETCH_THRESHOLD}건 미만) — 재수집")
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        print("  컬렉션 없음 — 새로 생성")

    _collection = _client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    print("📝 100세누리 API에서 데이터 수집")
    refresh()


def refresh(batch_size: int = 100):
    """
    데이터 새로 수집 후 컬렉션 업데이트 (스케줄러에서 매일 06시 호출 예정).
    임베딩은 batch_size 단위로 나눠서 진행 상황 출력.
    """
    try:
        jobs = fetch_from_senuri(SENURI_API_KEY)
        if not jobs:
            print("⚠️ 100세누리 응답 0건 → 샘플 데이터로 폴백")
            jobs = SAMPLE_JOBS
    except Exception as e:
        print(f"⚠️ 100세누리 수집 실패 ({e}) → 샘플 데이터로 폴백")
        jobs = SAMPLE_JOBS

    total = len(jobs)
    print(f"📐 임베딩 시작 ({total}건, bge-m3 CPU, batch={batch_size})")

    for i in range(0, total, batch_size):
        chunk = jobs[i:i + batch_size]
        _collection.upsert(
            ids=[j["id"] for j in chunk],
            documents=[_job_to_text(j) for j in chunk],
            metadatas=[{
                "title": j["title"],
                "company": j["company"],
                "location": j["location"],
                "work_type": j["work_type"],
                "age_friendly": str(j["age_friendly"]),
                "physical_intensity": j["physical_intensity"],
            } for j in chunk],
        )
        done = min(i + batch_size, total)
        pct = round(done / total * 100)
        print(f"   ↳ [{done:>4}/{total}] {pct}% 적재")

    print(f"✅ 적재 완료: {total}건")


def _fetch_page_with_retry(api_key: str, page: int, page_size: int, max_retries: int = 3):
    """
    단일 페이지 요청 + 재시도 로직.

    Returns:
        ("ok", response)        — 성공
        ("stop", reason_str)    — rate limit (401/429/503) → 전체 fetch 중단
        ("skip", reason_str)    — 일시 실패 (timeout/conn err 등) → 이 페이지만 스킵
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(
                SENURI_BASE,
                params={"serviceKey": api_key, "pageNo": page, "numOfRows": page_size},
                timeout=60,
            )
            res.raise_for_status()
            return ("ok", res)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (401, 429, 503):
                return ("stop", f"HTTP {status}")
            last_err = f"HTTP {status}"
        except requests.RequestException as e:
            last_err = type(e).__name__

        if attempt < max_retries:
            wait = 2 * attempt
            print(f"  ⚠️ page {page} attempt {attempt} 실패 ({last_err}), {wait}s 후 재시도")
            time.sleep(wait)

    return ("skip", last_err)


def fetch_from_senuri(
    api_key: str,
    page_size: int = 100,
    target_count: int = 1000,
    max_pages: int = 50,
    exclude_closed: bool = True,
    sleep_between: float = 0.5,
) -> list[dict]:
    """
    한국노인인력개발원 100세누리 구인정보 채용목록 API.

    엔드포인트: GET https://apis.data.go.kr/B552474/SenuriService/getJobList
    필수 파라미터: serviceKey, pageNo, numOfRows
    응답: <response><header><resultCode/><resultMsg/></header>
              <body><items><item>...</item></items>
                    <numOfRows/><pageNo/><totalCount/></body></response>

    페이지를 순회하며 마감/중복 제외한 **활성** 공고가 target_count에 도달하면 중단.
    100세누리는 워크넷/일모아 통합 집계라 totalCount가 수십만 단위인데 활성 비율은
    낮음(~30%) — 따라서 raw 페이지 cap이 아니라 활성 건수 목표 기반으로 fetch.

    Args:
        api_key: 공공데이터포털 발급 인증키
        page_size: 페이지당 결과 수 (numOfRows). 기본 100
        target_count: 활성 공고 목표 건수. 도달 시 중단 (기본 1000)
        max_pages: 안전 상한. target 못 채워도 여기서 강제 중단 (기본 50 = raw 5000건)
        exclude_closed: 마감 공고 제외 (기본 True)
        sleep_between: 페이지 간 sleep 초. 401 burst rate limit 방지용 (기본 0.5)

    Returns:
        정규화된 활성 채용공고 dict 리스트 (목표 도달 시 정확히 target_count, 못 채우면 그 이하)
    """
    all_jobs = []
    seen_ids = set()
    page = 1
    total_count = None
    skipped_closed = 0
    skipped_dup = 0

    # 페이지를 받아도 활성 공고가 거의 안 늘면 중단
    STAGNANT_LIMIT = 3   # 연속 N페이지 동안 미달 시 중단
    MIN_GAIN = 5         # 페이지당 최소 활성 증가 기대치
    stagnant_pages = 0
    prev_count = 0

    while page <= max_pages and len(all_jobs) < target_count:
        if page > 1:
            time.sleep(sleep_between)

        outcome, payload = _fetch_page_with_retry(api_key, page, page_size)
        if outcome == "stop":
            print(f"  ⚠️ page {page}: {payload} (rate limit/quota) — "
                  f"여기까지 {len(all_jobs)}건만 사용")
            break
        if outcome == "skip":
            print(f"  ⚠️ page {page}: {payload} 재시도 모두 실패, 다음 페이지로")
            page += 1
            continue

        res = payload
        root = ET.fromstring(res.text)

        # 결과코드 확인 (정상: 0 또는 00)
        result_code = (root.findtext(".//resultCode") or "").strip()
        if result_code not in ("0", "00"):
            result_msg = (root.findtext(".//resultMsg") or "").strip()
            raise RuntimeError(f"100세누리 API 에러 [{result_code}]: {result_msg}")

        if total_count is None:
            total_count = int((root.findtext(".//totalCount") or "0").strip() or "0")
            print(f"  totalCount: {total_count}건 (활성 {target_count}건까지 수집)")
            if total_count == 0:
                return []

        items = root.findall(".//item")
        if not items:
            break

        for item in items:
            job_id = (item.findtext("jobId") or "").strip()
            if not job_id:
                continue

            # 100세누리는 같은 jobId가 페이지 간/시스템 간 중복 등장하는 경우 있음
            if job_id in seen_ids:
                skipped_dup += 1
                continue
            seen_ids.add(job_id)

            deadline = (item.findtext("deadline") or "").strip()
            if exclude_closed and deadline == "마감":
                skipped_closed += 1
                continue

            jobcls_nm = (item.findtext("jobclsNm") or "").strip()
            acpt = (item.findtext("acptMthd") or "").strip()
            fr_dd = (item.findtext("frDd") or "").strip()
            to_dd = (item.findtext("toDd") or "").strip()

            description_parts = []
            if jobcls_nm:
                description_parts.append(f"직종: {jobcls_nm}")
            if acpt:
                description_parts.append(f"접수방법: {acpt}")
            if fr_dd or to_dd:
                description_parts.append(f"접수기간: {fr_dd}~{to_dd}")
            if deadline:
                description_parts.append(f"상태: {deadline}")

            # emplymShpNm이 한글 대신 코드(CM0105 등)로 오는 API 버그 → 클라에서 매핑
            emp_code = (item.findtext("emplymShp") or "").strip()
            emp_nm = (item.findtext("emplymShpNm") or "").strip()
            work_type = EMPLOYMENT_TYPES.get(emp_code, emp_nm or emp_code or "기타")

            all_jobs.append({
                "id": job_id,
                "title": (item.findtext("recrtTitle") or "").strip(),
                "company": (item.findtext("oranNm") or "").strip() or "(기업명 미공개)",
                "location": (item.findtext("workPlcNm") or "").strip(),
                "work_type": work_type,
                "description": " | ".join(description_parts),
                "age_friendly": True,  # 100세누리 전체가 60세 이상 시니어 대상
                "physical_intensity": "unknown",
            })

            # 활성 target 도달 시 즉시 중단
            if len(all_jobs) >= target_count:
                break

        # 이번 페이지에서 활성 공고 얼마나 늘었나
        gained = len(all_jobs) - prev_count
        prev_count = len(all_jobs)
        if gained < MIN_GAIN:
            stagnant_pages += 1
        else:
            stagnant_pages = 0

        print(
            f"   ↳ page {page:>2}: 누적 {len(all_jobs):>4}/{target_count}건 "
            f"(+{gained}, 마감 {skipped_closed}, 중복 {skipped_dup})"
        )

        # 종료 조건들
        processed = (page - 1) * page_size + len(items)
        if len(all_jobs) >= target_count:
            break
        if processed >= total_count or len(items) < page_size:
            break
        if stagnant_pages >= STAGNANT_LIMIT:
            print(f"   ↳ {STAGNANT_LIMIT}페이지 연속 활성 증가 {MIN_GAIN}건 미만 — 수집 중단")
            break
        page += 1

    msg = f"  수집 완료: {len(all_jobs)}건"
    extras = []
    if skipped_closed:
        extras.append(f"마감 {skipped_closed}건")
    if skipped_dup:
        extras.append(f"중복 {skipped_dup}건")
    if extras:
        msg += f" ({', '.join(extras)} 제외)"
    if len(all_jobs) < target_count:
        msg += f" — 목표 {target_count}건 미달, max_pages={max_pages} 도달"
    print(msg)
    return all_jobs


def _job_to_text(job: dict) -> str:
    """채용공고 → 임베딩용 텍스트"""
    return (
        f"직종: {job['title']} | 회사: {job['company']} | "
        f"지역: {job['location']} | 근무형태: {job['work_type']} | "
        f"내용: {job['description']}"
    )


def _resume_to_query(resume: dict) -> str:
    """이력서 → 검색 쿼리 텍스트"""
    parts = []
    if resume.get("career"):
        parts.append(f"경력: {resume['career']}")
    if resume.get("location"):
        parts.append(f"거주지역: {resume['location']}")
    if resume.get("preferred_work_type"):
        parts.append(f"희망근무: {resume['preferred_work_type']}")
    if resume.get("age"):
        parts.append(f"나이: {resume['age']}세")
    return " | ".join(parts) if parts else "시니어 구직자"


def search(resume: dict, top_k: int = 5) -> list[dict]:
    """이력서 기반 유사 채용공고 검색 (고령친화 하드필터 적용)"""
    if _collection is None:
        raise RuntimeError("setup()을 먼저 호출하세요.")

    query_text = _resume_to_query(resume)
    print(f"🔍 검색 쿼리: {query_text}")

    results = _collection.query(
        query_texts=[query_text],
        n_results=min(top_k, _collection.count()),
        where={"age_friendly": "True"},
    )

    return [
        {
            "rank": i + 1,
            "job_id": doc_id,
            "title": meta["title"],
            "company": meta["company"],
            "location": meta["location"],
            "work_type": meta["work_type"],
            "similarity_score": round((1 - dist) * 100, 1),
        }
        for i, (doc_id, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ))
    ]


def build_recommend_prompt(resume: dict, jobs: list[dict]) -> str:
    """LLM에 넘길 추천 프롬프트"""
    resume_text = "\n".join(f"- {k}: {v}" for k, v in resume.items())
    jobs_text = "\n".join(
        f"{j['rank']}. [{j['job_id']}] {j['title']} / {j['company']} / "
        f"{j['location']} / {j['work_type']} (유사도: {j['similarity_score']}점)"
        for j in jobs
    )
    return f"""[이력서 정보]
{resume_text}

[RAG로 선별된 채용공고 {len(jobs)}건]
{jobs_text}

위 정보를 바탕으로 JSON 형식으로 추천 결과를 반환해."""


if __name__ == "__main__":
    setup()
    test_resume = {
        "name": "김철수", "age": 65, "location": "서울 강남구",
        "career": "아파트 관리, 경비, 시설 관리 20년",
        "preferred_work_type": "상근 또는 시간제",
        "physical_condition": "가벼운 실내업무 선호",
    }
    print("\n--- RAG 검색 테스트 ---")
    for r in search(test_resume, top_k=5):
        print(f"  {r['rank']}위: {r['title']} ({r['company']}) | {r['similarity_score']}점")
