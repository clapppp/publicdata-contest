"""
rag.py — 채용공고 RAG (ChromaDB)
- 임베딩: BAAI/bge-m3 (CPU)
- 저장: PersistentClient (./chroma_db)
- 데이터: 워크넷 채용정보 OpenAPI
"""
from xml.etree import ElementTree as ET

import requests
import chromadb
from chromadb.utils import embedding_functions

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "job_postings"
DB_PATH = "./chroma_db"

# 워크넷 채용정보 채용목록 API
# 참조: https://www.data.go.kr/data/3038225/openapi.do
WORKNET_BASE = "https://www.work24.go.kr/cm/openApi/call/wk/callOpenApiSvcInfo210L01.do"
WORKNET_API_KEY = "5e05eff1-fb3c-4037-8d3f-687294fa2e34"

# 통계청 시도 코드 (워크넷 region 파라미터에 사용)
# TODO: 실제 워크넷 API 가이드 PDF에서 region 코드 형식 확인 후 조정
REGIONS = {
    "서울": "11000", "부산": "26000", "대구": "27000", "인천": "28000",
    "광주": "29000", "대전": "30000", "울산": "31000", "세종": "36000",
    "경기": "41000", "강원": "42000", "충북": "43000", "충남": "44000",
    "전북": "45000", "전남": "46000", "경북": "47000", "경남": "48000",
    "제주": "50000",
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


def setup():
    """ChromaDB 초기화. DB 비어있으면 데이터 적재."""
    global _client, _collection

    print(f"📦 임베딩 모델 로드: {EMBED_MODEL} (CPU)")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL,
        device="cpu",
    )

    _client = chromadb.PersistentClient(path=DB_PATH)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    if _collection.count() == 0:
        print("📝 빈 DB → 초기 데이터 적재")
        refresh()
    else:
        print(f"✅ RAG 준비 완료 | 채용공고 {_collection.count()}건")


def refresh():
    """데이터 새로 수집 후 컬렉션 업데이트 (스케줄러에서 매일 06시 호출 예정)"""
    try:
        jobs = fetch_from_worknet(WORKNET_API_KEY)
        if not jobs:
            print("⚠️ 워크넷 응답 0건 → 샘플 데이터로 폴백")
            jobs = SAMPLE_JOBS
    except Exception as e:
        print(f"⚠️ 워크넷 수집 실패 ({e}) → 샘플 데이터로 폴백")
        jobs = SAMPLE_JOBS

    _collection.upsert(
        ids=[j["id"] for j in jobs],
        documents=[_job_to_text(j) for j in jobs],
        metadatas=[{
            "title": j["title"],
            "company": j["company"],
            "location": j["location"],
            "work_type": j["work_type"],
            "age_friendly": str(j["age_friendly"]),
            "physical_intensity": j["physical_intensity"],
        } for j in jobs],
    )
    print(f"✅ 적재 완료: {len(jobs)}건")


def fetch_from_worknet(api_key: str) -> list[dict]:
    """
    워크넷 채용정보 채용목록 API 호출 (17개 시도 순회).

    NOTE — 아래 파라미터/응답 필드명은 일반적인 워크넷 OpenAPI 패턴 기반 추정.
    공식 가이드 PDF (work24.go.kr 로그인 후 다운로드) 받으면 다음 항목 검증/조정 필요:
      - region 코드 형식 (현재: 통계청 시도코드)
      - 우대조건 파라미터명/코드 (preferentialCd? prefCd? 고령자 코드는?)
      - 응답 XML 루트/항목 태그명 (현재 추정: <wantedInfo>)
      - 필드명 (wantedAuthNo / title / company 등)
    """
    all_jobs = []
    for region_name, region_code in REGIONS.items():
        try:
            res = requests.get(
                WORKNET_BASE,
                params={
                    "authKey": api_key,
                    "callTp": "L",
                    "returnType": "XML",
                    "startPage": 1,
                    "display": 100,
                    "region": region_code,
                    # "preferentialCd": "...",  # 고령자 우대 코드 확인 필요
                },
                timeout=30,
            )
            res.raise_for_status()
            root = ET.fromstring(res.text)

            count = 0
            for item in root.iter("wanted"):  # 응답 항목 태그명 추정
                job_id = item.findtext("wantedAuthNo", "").strip()
                if not job_id:
                    continue
                all_jobs.append({
                    "id": job_id,
                    "title": item.findtext("title", "").strip(),
                    "company": item.findtext("company", "").strip(),
                    "location": item.findtext("workPlcNm", region_name).strip(),
                    "work_type": item.findtext("empTpNm", "").strip(),
                    "description": (item.findtext("jobsCd", "") or item.findtext("etcItm", "")).strip(),
                    "age_friendly": True,  # 우대조건 필터 적용했다고 가정
                    "physical_intensity": "unknown",
                })
                count += 1
            print(f"  {region_name}: {count}건")
        except Exception as e:
            print(f"  ⚠️ {region_name} 실패: {e}")
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
