import glob
import os
import re
import xml.etree.ElementTree as ET

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

def _extract_article_text(article: ET.Element) -> str:
    """Collect text from an Article element."""
    caption = article.findtext('ArticleCaption') or ''
    title = article.findtext('ArticleTitle') or ''
    sentences = [s.text.strip() for s in article.findall('.//Sentence') if s.text]
    body = ' '.join(sentences)
    return ' '.join([caption, title, body]).strip()


@st.cache_data
def load_tax_law_xml(path_pattern: str) -> pd.DataFrame:
    records = []
    for filepath in glob.glob(path_pattern):
        tree = ET.parse(filepath)
        root = tree.getroot()
        date = root.findtext('.//EffectiveDate')
        if not date:
            m = re.search(r'_(\d{8})_', os.path.basename(filepath))
            date = m.group(1) if m else '不明'
        for art in root.findall('.//Article'):
            num = art.attrib.get('Num') or art.findtext('ArticleTitle') or '不明'
            title = art.findtext('ArticleTitle') or ''
            caption = art.findtext('ArticleCaption') or ''
            txt = _extract_article_text(art)
            records.append({
                'id': os.path.basename(filepath) + '#' + str(num),
                'effective_date': date,
                'article': str(num),
                'title': title,
                'caption': caption,
                'article_label': f"{num} {title or caption}",
                'text': txt
            })
    return pd.DataFrame(records)

@st.cache_resource
def init_vector_index(df: pd.DataFrame):
    # multilingual model works better for Japanese queries
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    texts = df['text'].tolist()
    embs = model.encode(texts, convert_to_tensor=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs.cpu().numpy())
    return model, index, embs

# ─── Streamlit UI ───
st.set_page_config(page_title="消費税法ベクトル検索", layout="wide")
st.title('💡 消費税法ベクトル検索アプリ')

# データロード
df = load_tax_law_xml('data/*.xml')
model, index, embs = init_vector_index(df)

# サイドバー：絞り込みフィルター
st.sidebar.header('🔍 絞り込みフィルター')
dates = st.sidebar.multiselect('施行日', sorted(df['effective_date'].unique()))
arts  = st.sidebar.multiselect('条文', sorted(df['article_label'].unique()))


# メイン：検索クエリ
query = st.text_input('🔎 ご質問を入力してください')
synonyms = {'不課税': '非課税'}
search_query = query
for k, v in synonyms.items():
    search_query = search_query.replace(k, v)
top_k = st.slider('表示件数', 1, 10, 5)

if query:
    q_emb = model.encode(search_query, convert_to_tensor=True).cpu().numpy()
    dists, idxs = index.search(q_emb.reshape(1, -1), len(df))
    results = df.iloc[idxs[0]]
    if dates:
        results = results[results['effective_date'].isin(dates)]
    if arts:
        results = results[results['article_label'].isin(arts)]
    results = results.head(top_k)
    st.markdown("### 検索結果")
    for _, row in results.iterrows():
        with st.expander(f"条文 {row['article_label']} （施行日: {row['effective_date']}）"):
            st.write(row['text'])

# フッター
st.sidebar.markdown("---")
st.sidebar.markdown("**使い方**\n1. dataフォルダにXMLを配置\n2. `streamlit run app.py` で起動\n")
