"""
ingest.py
---------
data/index_docstore_export.jsonl 파일을 읽어
ChromaDB에 임베딩을 저장합니다.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 환경변수 로드 
load_dotenv()

# 경로 설정 
BASE_DIR   = Path(__file__).parent
DATA_FILE_ORIGINAL = BASE_DIR / "data" / "index_docstore_export.jsonl"
DATA_FILE_AUGMENTED = BASE_DIR / "data" / "index_docstore_augmented.jsonl"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "airline_regulations"


def get_data_file() -> Path:
    """존재하는 데이터 파일을 가져온다. augmented 파일이 존재하면 우선 적용한다."""
    if DATA_FILE_AUGMENTED.exists():
        print(f"✨ 증강된(Augmented) 데이터 파일을 발견하여 우선 적용합니다: {DATA_FILE_AUGMENTED.name}")
        return DATA_FILE_AUGMENTED
    return DATA_FILE_ORIGINAL

def load_jsonl(filepath: Path) -> list[dict]:
    """JSONL 파일을 읽어 dict 리스트로 반환."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_documents(records: list[dict]) -> list[Document]:
    """
    JSONL 레코드 → LangChain Document 변환.
    recommended_metadata 값을 메타데이터로 매핑.
    ChromaDB는 메타데이터 값으로 str/int/float/bool만 허용하므로
    중첩 dict는 문자열로 변환.
    """
    docs = []
    for rec in records:
        meta_raw = rec.get("recommended_metadata", {})
        # ChromaDB 호환: 모든 값을 str 처리
        metadata = {k: str(v) for k, v in meta_raw.items()}
        # doc_id 및 country 추가
        metadata["doc_id"]  = rec.get("doc_id", "")
        metadata["country"] = rec.get("country", "")

        docs.append(Document(
            page_content=rec["page_content"],
            metadata=metadata,
        ))
    return docs


def main():
    print("=" * 50)
    print("🛫 기내뭐돼 — 데이터 임베딩 시작")
    print("=" * 50)

    # 이미 DB가 존재하면 스킵
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f" ChromaDB가 이미 존재합니다: {CHROMA_DIR}")
        print("   재생성하려면 chroma_db/ 폴더를 삭제 후 다시 실행하세요.")
        return

    # 1) 데이터 로드 찾기
    DATA_FILE = get_data_file()
    print(f"\n 데이터 파일 로드 중: {DATA_FILE}")
    records = load_jsonl(DATA_FILE)
    print(f"   → {len(records)}개 레코드 발견")

    # 2) Document 변환
    docs = build_documents(records)
    print(f"   → {len(docs)}개 문서 변환 완료")

    # 3) 임베딩 모델 초기화
    print("\n🔑 OpenAI 임베딩 모델 초기화 중 (text-embedding-3-small)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4) ChromaDB 저장
    print(f"\n ChromaDB 저장 중: {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    print(f"\n 임베딩 완료. 총 {len(docs)}개 문서가 chroma_db/에 저장되었습니다.")
    print("=" * 50)


if __name__ == "__main__":
    main()
