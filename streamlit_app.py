import streamlit as st
import openai
import requests
import pandas as pd
import fitz  # PyMuPDF
import io
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# --- API KEYS (SET YOURS HERE) ---
OPENAI_API_KEY = "YOUR-OPENAI-API-KEY"
openai.api_key = OPENAI_API_KEY

# --- Helper Functions ---

def search_semantic_scholar(topic: str, top_n: int = 5) -> List[dict]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={topic}&limit={top_n}&fields=title,authors,year,url,openAccessPdf"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('data', [])
    else:
        return []


def download_pdf(pdf_url: str) -> bytes:
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return response.content
    else:
        return None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def summarize_text_with_openai(text: str) -> str:
    prompt = f"""
    Please create a summary table for the following research paper. The table should include:
    Title, Authors, Publication year, Research question or objective, Methodology, Key findings, Limitations, Conclusions.
    Provide brief but informative summaries for each, using bullet points where appropriate.

    Text:
    {text[:6000]}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


def ask_question(text: str, question: str) -> str:
    prompt = f"""
    Based on the following research paper content, answer the question:

    Paper Content:
    {text[:6000]}

    Question:
    {question}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


def auto_cluster_texts(texts: List[str], n_clusters: int = 3) -> List[int]:
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels


def specialized_analysis(texts: List[str], task: str) -> str:
    joined_text = "\n".join(texts)
    prompt = f"""
    Analyze the following group of research papers and {task}. Provide detailed but concise points.

    Texts:
    {joined_text[:6000]}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# --- Streamlit App ---

def streamlit_app():
    st.title("📚 AI Research Assistant")

    st.sidebar.header("Configuration")
    mode = st.sidebar.selectbox("Select Mode", ["Search Papers by Topic", "Upload Your Own Papers"])
    num_papers = st.sidebar.slider("Number of papers (if searching)", min_value=1, max_value=20, value=5)
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=3)

    summaries = []
    full_texts = []
    titles = []
    qa_memory = []

    if mode == "Search Papers by Topic":
        topic = st.text_input("Enter Research Topic:")
        if st.button("Search and Summarize"):
            with st.spinner("Searching papers..."):
                papers = search_semantic_scholar(topic, top_n=num_papers)
                for paper in papers:
                    title = paper.get("title", "Unknown Title")
                    pdf_link = paper.get("openAccessPdf", {}).get("url")
                    if pdf_link:
                        file_bytes = download_pdf(pdf_link)
                        if file_bytes:
                            text = extract_text_from_pdf(file_bytes)
                            summary = summarize_text_with_openai(text)
                            summaries.append(summary)
                            full_texts.append(text)
                            titles.append(title)

    elif mode == "Upload Your Own Papers":
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            if st.button("Summarize Uploaded Papers"):
                with st.spinner("Summarizing uploaded papers..."):
                    for uploaded_file in uploaded_files:
                        file_bytes = uploaded_file.read()
                        text = extract_text_from_pdf(file_bytes)
                        summary = summarize_text_with_openai(text)
                        summaries.append(summary)
                        full_texts.append(text)
                        titles.append(uploaded_file.name)

    if summaries:
        st.header("📒 Paper Summaries")
        df = pd.DataFrame({"Title": titles, "Summary": summaries})
        st.dataframe(df)

        clusters = auto_cluster_texts(full_texts, n_clusters=n_clusters)
        cluster_map = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(clusters):
            cluster_map[label].append(titles[idx])

        st.subheader("🔍 Auto-Generated Clusters")
        for cluster_id, papers in cluster_map.items():
            st.markdown(f"**Cluster {cluster_id+1}:** {', '.join(papers)}")

        selected_titles = st.multiselect("Select Papers for Question Answering or Group Analysis", titles)

        if selected_titles:
            selected_texts = [full_texts[titles.index(t)] for t in selected_titles]
            cluster_text = "\n".join(selected_texts)

            user_question = st.text_input("Ask a custom question about the selected paper(s):")
            if st.button("Get Custom Answer") and user_question:
                answer = ask_question(cluster_text, user_question)
                st.success(answer)
                qa_memory.append({"Question": user_question, "Answer": answer, "Papers": selected_titles})

            if st.button("Find Research Gaps"):
                answer = specialized_analysis(selected_texts, "identify the research gaps")
                st.success(answer)
                qa_memory.append({"Question": "Research Gaps", "Answer": answer, "Papers": selected_titles})

            if st.button("Suggest Future Work"):
                answer = specialized_analysis(selected_texts, "suggest future research directions")
                st.success(answer)
                qa_memory.append({"Question": "Future Work", "Answer": answer, "Papers": selected_titles})

            if st.button("Find Contradictions"):
                answer = specialized_analysis(selected_texts, "detect contradictions among the papers")
                st.success(answer)
                qa_memory.append({"Question": "Contradictions", "Answer": answer, "Papers": selected_titles})

        if qa_memory:
            st.subheader("💬 Q&A Session History")
            for entry in qa_memory:
                st.markdown(f"**Question:** {entry['Question']}")
                st.markdown(f"**Answer:** {entry['Answer']}")
                st.markdown(f"**Papers:** {', '.join(entry['Papers'])}")
                st.markdown("---")

            qa_df = pd.DataFrame(qa_memory)
            qa_csv = qa_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Q&A History", data=qa_csv, file_name="qa_history.csv", mime='text/csv')

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Summaries as CSV", data=csv, file_name="summaries.csv", mime='text/csv')

if __name__ == "__main__":
    streamlit_app()
