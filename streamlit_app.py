# app.py

import os
# !pip install streamlit # Removed to avoid re-install
import requests
# !pip install PyMuPDF  # Install PyMuPDF for fitz # Removed to avoid re-install
!pip install pymupdf
import fitz  # PyMuPDF
!pip install google-search-results # Install the google-search-results package to provide the serpapi module. # Added to fix error
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
import openai
import streamlit as st # Import streamlit and alias it as 'st'

# ======= CONFIGURATION ========
# Set your API Keys as environment variables OR directly here
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY") or "your-serpapi-key"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-openai-key"
openai.api_key = OPENAI_API_KEY

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# ======= FUNCTIONS ========

def search_google_scholar(query, num_results=5):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    papers = []
    for res in results.get('organic_results', []):
        papers.append({
            'title': res.get('title'),
            'link': res.get('link'),
            'authors': res.get('publication_info', {}).get('summary', 'Unknown'),
            'year': res.get('publication_info', {}).get('year', 'Unknown')
        })
    return papers

def search_semantic_scholar(query, num_results=5):
    params = {
        "query": query,
        "limit": num_results,
        "fields": "title,authors,year,url"
    }
    response = requests.get(SEMANTIC_SCHOLAR_API, params=params)
    data = response.json()
    papers = []
    for paper in data.get('data', []):
        papers.append({
            'title': paper['title'],
            'link': paper.get('url', 'Unavailable'),
            'authors': ', '.join([author['name'] for author in paper.get('authors', [])]),
            'year': paper.get('year', 'Unknown')
        })
    return papers

def download_pdf(url, save_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.headers.get('content-type') == 'application/pdf':
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading directly: {e}")
    return False

def download_from_scihub(doi_or_url, save_path):
    sci_hub_base = "https://sci-hub.se/"
    try:
        session = requests.Session()
        response = session.get(sci_hub_base + doi_or_url, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        iframe = soup.find('iframe')
        if iframe:
            pdf_url = iframe['src']
            if pdf_url.startswith('//'):
                pdf_url = 'https:' + pdf_url
            pdf_response = session.get(pdf_url, stream=True)
            with open(save_path, 'wb') as f:
                f.write(pdf_response.content)
            return True
    except Exception as e:
        print(f"Error using Sci-Hub: {e}")
    return False

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def summarize_paper(text, paper_metadata):
    prompt = f"""
Please summarize the following paper into a structured table with these fields:
- Title
- Authors
- Publication Year
- Research Question or Objective
- Methodology
- Key Findings
- Limitations
- Conclusions

Paper Title: {paper_metadata.get('title')}
Authors: {paper_metadata.get('authors')}
Year: {paper_metadata.get('year')}

Paper Text:
{text[:5000]}  # Limiting input size for token safety.

Please give clear and concise bullet points for each section.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error during summarization: {e}"

# ======= STREAMLIT APP ========


st.title("📚 Research Summarizer (Google + Semantic + Sci-Hub Backup)")

query = st.text_input("What are the behavioural and psychological factors that explain why urban middle-class consumers in India increasingly use credit cards?")

if query:  # <--- This triggers automatically once the user types
    with st.spinner("Searching papers..."):
        google_papers = search_google_scholar(query)
        semantic_papers = search_semantic_scholar(query)
        papers = google_papers + semantic_papers

    for idx, paper in enumerate(papers):
        st.subheader(f"Paper {idx+1}: {paper['title']}")
        filename = f"temp_{idx}.pdf"
        got_pdf = False

        if paper['link'] and download_pdf(paper['link'], filename):
            got_pdf = True
        else:
            doi = paper['link'].split('doi.org/')[-1] if 'doi.org' in paper['link'] else paper['link']
            if download_from_scihub(doi, filename):
                got_pdf = True

        if got_pdf:
            text = extract_text_from_pdf(filename)
            if text:
                with st.spinner("Summarizing..."):
                    summary = summarize_paper(text, paper)
                st.markdown(summary)
            else:
                st.warning("Failed to extract text from PDF.")
            os.remove(filename)  # Clean up temp file
        else:
            st.warning("Could not fetch full text. Only metadata available.")
