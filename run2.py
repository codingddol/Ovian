import streamlit as st
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from io import BytesIO
import os
from transformers import ViTFeatureExtractor, ViTForImageClassification
from sentence_transformers import SentenceTransformer
import torch
import base64
import openai

# LangChain & LLM
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from Bio import Entrez

# 세션 상태 초기화
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# 이미지 유사 논문 찾기
def find_similar_papers_from_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return [
        Document(page_content="Ovarian cancer tissue with high mitotic index...", metadata={"pmid": "12345678", "source": "pubmed"}),
        Document(page_content="Study on serous subtype of ovarian carcinoma...", metadata={"pmid": "87654321", "source": "pubmed"})
    ]

# PubMed 검색 쿼리 자동 생성 (LLM 활용)
def generate_pubmed_query_from_question(question: str) -> str:
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        아래 설명은 AI가 조직 병리 이미지를 분석한 결과입니다.
        이 설명을 기반으로 PubMed 검색을 위한 Boolean 쿼리를 생성하세요.

        - 핵심 질병명, 조직명, 염색법(H&E 등), 병리 용어 중심 키워드를 포함하세요.
        - Boolean 연산자 (AND, OR)와 괄호를 사용하세요.
        - 키워드는 3~6개 사이로 구성하세요.
        - 예시: (histology OR tissue) AND (H&E OR staining) AND (diagnosis OR cancer)

        설명:
        {question}

        출력 형식: PubMed Boolean Query 
        """
    )
    prompt_str = query_prompt.format(question=question)
    return llm.invoke(prompt_str).content.strip()

# PubMed 논문 검색
def search_pubmed(question: str, max_results: int = 3):
    Entrez.email = "yejijin98@gmail.com"
    query = generate_pubmed_query_from_question(question)
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]

    summaries = []
    seen_pmids = set()
    for pmid in ids:
        if pmid in seen_pmids:
            continue
        seen_pmids.add(pmid)

        summary = Entrez.esummary(db="pubmed", id=pmid, retmode="xml")
        summary_record = Entrez.read(summary)
        if not summary_record or not isinstance(summary_record[0], dict):
            continue
        title = summary_record[0].get("Title", "[제목 없음]")
        pubdate = summary_record[0].get("PubDate", "")                      # PubMed는 'PubDate' 키를 사용
        year = pubdate.split()[0] if pubdate else ""                        # 안전하게 연도만 추출
        authors = summary_record[0].get("AuthorList", [])                   # authors → AuthorList
        author_str = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")


        fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
        abstract = fetch.read()
        if not abstract:
            continue

        summaries.append(Document(
            page_content=abstract,
            metadata={"pmid": pmid, "title": title, "year": year, "authors": author_str, "source": "pubmed"}
        ))
    return summaries

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    
당신은 난소암 조직 병리 이미지와 관련된 AI 전문가입니다.

다음은 논문에서 발췌한 내용입니다:
---------------------
{context}
---------------------

이 내용을 바탕으로 아래 질문에 답변하세요.
질문: {question}
"""
)

# Streamlit UI
st.set_page_config(page_title="Ovarian Cancer RAG", layout="wide")

st.sidebar.title("🔐 OpenAI API KEY 입력")
user_api_key = st.sidebar.text_input("API Key를 입력하세요", type="password")
if user_api_key:
    st.session_state["user_api_key"] = user_api_key
openai.api_key = st.session_state.get("user_api_key", "")

st.title("🔬 난소암 분석 AI 어시스턴트")

question = st.text_input("텍스트 질문을 입력하세요", placeholder="예시) What are the subtypes of ovarian cancer?")

uploaded_file = st.file_uploader("조직 이미지를 업로드하세요 (선택)", type=["png", "jpg", "jpeg"])

if st.button("분석 실행"):
    # LLM 설정
    llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=st.session_state.get("user_api_key", ""))
    embeddings = HuggingFaceEmbeddings(
    model_name="dmis-lab/biobert-base-cased-v1.1",
    model_kwargs={}
    )
        
    if uploaded_file is not None:
        # ✅ 이미지 분석 블록
        image_bytes = uploaded_file.read()
        image = Image.open(BytesIO(image_bytes))
        st.image(image, caption="업로드된 조직 이미지", use_container_width=True)

        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 이미지에 대해 설명해주세요."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}}
                ],
            }
        ]

        try:
            with st.spinner("GPT-4o 분석 중..."):
                response = openai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=500
                )
            if not response.choices:
                raise ValueError("GPT 응답이 비어 있습니다.")

            ai_answer = response.choices[0].message.content
            st.subheader("AI 답변")
            st.write(ai_answer)

            search_query = generate_pubmed_query_from_question(ai_answer)
            related_papers = search_pubmed(search_query, max_results=3)

            st.subheader("📄 관련 논문")
            if not related_papers:
                st.markdown("🔎 관련 논문을 찾을 수 없습니다.")
            else:
                for doc in related_papers:
                    title = doc.metadata.get("title", "[제목 없음]")
                    year = doc.metadata.get("year", "")
                    authors = doc.metadata.get("authors", "")
                    pmid = doc.metadata.get("pmid", "")
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                    st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")

            with st.expander("답변 관련 논문"):
                    docs = find_similar_papers_from_image(image_bytes)
                    splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
                    


        except Exception as e:
            st.error(f"❌ GPT 호출 오류: {e}")

    else:
        # ✅ 텍스트 기반 Q&A 블록        
        docs = search_pubmed(question=question)
        splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        result = rag_chain.invoke({"query": question})

        st.session_state.qa_history.insert(0, {
            "question": question,
            "answer": result["result"],
            "sources": result["source_documents"]
        })

        st.subheader("🖍️ 요약 답변")
        st.write(result["result"])
        st.subheader("📄 논문 출처")

        seen_pmids = set()
        for doc in result["source_documents"]:
            pmid = doc.metadata.get("pmid")
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            title = doc.metadata.get("title", "[제목 없음]")
            year = doc.metadata.get("year", "")
            authors = doc.metadata.get("authors", "")
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

            st.markdown(f"- [{title}]({url}) ({year}, {authors})" if url else f"- {title} ({year}, {authors})")


# ✅ 하단에 이전 Q&A 표시 (가장 최근 것은 제외)
if len(st.session_state.qa_history) > 1:
    st.markdown("## 📚 이전 Q&A 기록")

    for idx, entry in enumerate(st.session_state.qa_history[1:]):
        with st.expander(f"Q{len(st.session_state.qa_history) - idx - 1}: {entry['question']}"):
            st.markdown("**🖍️ 요약 답변**")
            st.write(entry["answer"])
            st.markdown("**📄 출처**")
            for doc in entry["sources"]:
                source = doc.metadata.get("source", "pubmed")
                title = doc.metadata.get("title", "[제목 없음]")
                year = doc.metadata.get("year", "")
                authors = doc.metadata.get("authors", "")
                doi = doc.metadata.get("doi")
                pmid = doc.metadata.get("pmid")
                pmcid = doc.metadata.get("pmcid")

                if source == "pubmed" and pmid:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                elif source == "crossref" and doi:
                    url = f"https://doi.org/{doi}"
                elif source == "europepmc":
                    if pmcid:
                        url = f"https://europepmc.org/article/PMC/{pmcid}"
                    elif pmid:
                        url = f"https://europepmc.org/article/MED/{pmid}"
                    elif doi:
                        url = f"https://europepmc.org/article/DOI/{doi}"
                    else:
                        url = ""
                else:
                    url = ""

                if url:
                    st.markdown(f"- [{title}]({url}) ({year}, {authors})")
                else:
                    st.markdown(f"- {title} ({year}, {authors})")

# 🧼 기록 초기화
if st.button("기록 초기화"):
    st.session_state.qa_history = []
    st.rerun()


