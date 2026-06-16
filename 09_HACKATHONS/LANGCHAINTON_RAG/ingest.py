"""

ingest.py : data/index_docstore_export.jsonl 파일을 읽어 ChromaDB에 임베딩을 저장하는 모듈

"""

# 필요한 라이브러리 임포트
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 환경변수 로드 
load_dotenv()

# 설정값 모음 
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

DATA_FILE_ORIGINAL = DATA_DIR / "index_docstore_export.jsonl"
DATA_FILE_AUGMENTED = DATA_DIR / "index_docstore_augmented.jsonl"

CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "airline_regulations"

# 임베딩 모델
EMBEDDING_MODEL = "gemini-embedding-001"


# ChromaDB에 이미 임베딩 저장되어 있는지 체크
def chroma_db_exists() -> bool:
   
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())

# 증강 데이터가 있으면 우선 사용하고, 없으면 원본 데이터를 사용
def select_data_file() -> Path:

    if DATA_FILE_AUGMENTED.exists():
        print(f"증강 데이터 사용: {DATA_FILE_AUGMENTED.name}")
        return DATA_FILE_AUGMENTED

    print(f"원본 데이터 사용: {DATA_FILE_ORIGINAL.name}")
    return DATA_FILE_ORIGINAL

# JSONL 파일을 읽어서 ChromaDB에 임베딩 저장
def load_jsonl(filepath: Path) -> list[dict]:
    
    records = []

    with open(filepath, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                records.append(json.loads(stripped_line))
            except json.JSONDecodeError as error:
                raise ValueError(f"JSON 파싱 실패: {filepath}, line={line_number}") from error

    return records

# ChromaDB에 저장할 metadata 생성
def build_metadata(record: dict) -> dict:
    
    metadata_raw = record.get("recommended_metadata", {})

    # ChromaDB metadata 값은 str/int/float/bool 같은 단순 타입이어야 한다.
    metadata = {
        key: str(value)
        for key, value in metadata_raw.items()
    }

    metadata["doc_id"] = str(record.get("doc_id", ""))
    metadata["country"] = str(record.get("country", ""))

    return metadata


# JSONL 레코드 1개를 ChromaDB에 저장할 LangChain Document 객체로 변환
def build_document(record: dict) -> Document:
    
    return Document(
        page_content=record.get("page_content", ""),
        metadata=build_metadata(record),
    )

# 전체 레코드를 읽어서 Document 리스트로 변환
def build_documents(records: list[dict]) -> list[Document]:
    
    return [
        build_document(record)
        for record in records
    ]

# Google Gemini임베딩 모델 생성 
def create_embeddings() -> GoogleGenerativeAIEmbeddings:
    
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

# Document 리스트를 임베딩해 ChromaDB에 저장
def save_documents_to_chroma(documents: list[Document]) -> None:
   
    embeddings = create_embeddings()

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

# 메인 함수 (전체 임베딩 프로세스 실행)
def main() -> None:
    print("=" * 50)
    print("데이터 임베딩 시작")
    print("=" * 50)

    if chroma_db_exists():
        print(f"ChromaDB가 이미 존재합니다: {CHROMA_DIR}")
        print("새 데이터로 다시 만들려면 chroma_db/ 폴더를 직접 삭제한 뒤 실행하세요.")
        return

    data_file = select_data_file()

    records = load_jsonl(data_file)
    print(f"레코드 {len(records)}개 로드")

    documents = build_documents(records)
    print(f"Document {len(documents)}개 생성")

    print(f"임베딩 모델 사용: {EMBEDDING_MODEL}")
    print("ChromaDB 저장 중...")

    save_documents_to_chroma(documents)

    print(f"저장 완료: {CHROMA_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    main()