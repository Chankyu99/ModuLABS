```mermaid
graph TD
    %% 0. Data Augmentation
    subgraph "0. Data Augmentation (Offline)"
        A[Original Regulations JSONL] -->|data_augmenter.py| B((GPT-4o-mini))
        B -->|Synonyms & Keywords 10~15ea| C[Augmented Data JSONL]
    end

    %% 1. Data Ingestion
    subgraph "1. Data Ingestion & Indexing"
        C -->|ingest.py| D((text-embedding-3-small))
        D -->|Vectorization| E[(Chroma DB)]
    end

    %% 2. User Input & Router
    subgraph "2. Router & Slot Filling"
        User((User)) -->|Streamlit Chat| F{bot_logic.py: Router}
        F -->|Extract 'Departure, Arrival, Item'| G((GPT-4o-mini))
        G --> H{Are Slots Filled?}
        H -->|No| I[Ask User for Missing Info]
        I --> User
        H -->|Yes| J[Slot Confirmed]
    end

    %% 3. Retriever
    subgraph "3. Rewriter & Retriever"
        J --> K[Query Expansion / Normalization]
        K -->|Vector Search + Metadata Filter| E
        E -->|Return Top-K Docs| L{Docs Found?}
    end

    %% 4. Judge & Generator
    subgraph "4. Judge & Generator (Dual LLM)"
        L -->|Yes (Mapped)| M((GPT-4o-mini))
        M -->|RAG Answer| Result(Answer: 🟢 / 🟡 / 🔴)
        L -->|No (Not in DB)| N((GPT-5.2 Fallback))
        N -->|IATA/TSA General Knowledge Inference| Fallback_Result(Answer: 🟢 / 🟡 / 🔴 + ⚠️)
        Fallback_Result --> Result
    end

    %% Final Output
    Result --> User

    %% Styling
    classDef llm fill:#f9f,stroke:#333,stroke-width:2px;
    classDef db fill:#bbf,stroke:#333,stroke-width:2px;
    class B,G,M,N llm;
    class E db;
```
