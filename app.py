import streamlit as st
import glob, os
import xml.etree.ElementTree as ET
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

@st.cache_data
def load_tax_law_xml(path_pattern: str) -> pd.DataFrame:
    records = []
    for filepath in glob.glob(path_pattern):
        tree = ET.parse(filepath)
        root = tree.getroot()
        date = root.findtext('.//EffectiveDate') or '不明'
        for art in root.findall('.//Article'):
            num = art.findtext('Number') or '不明'
            txt = art.findtext('Content') or ''
            records.append({
                'id': os.path.basename(filepath) + '#' + num,
                'effective_date': date,
                'article': num,
                'text': txt.strip()
            })
    return pd.DataFrame(records)

@st.cache_resource
def init_vector_index(df: pd.DataFrame):
    model = SentenceTransformer('all-MiniLM-L6-v2')
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
arts  = st.sidebar.multiselect('条文番号', sorted(df['article'].unique()))

fdf = df
if dates:
    fdf = fdf[fdf['effective_date'].isin(dates)]
if arts:
    fdf = fdf[fdf['article'].isin(arts)]

# メイン：検索クエリ
query = st.text_input('🔎 ご質問を入力してください')
top_k = st.slider('表示件数', 1, 10, 5)

if query:
    q_emb = model.encode(query, convert_to_tensor=True).cpu().numpy()
    dists, idxs = index.search(q_emb.reshape(1,-1), top_k)
    results = fdf.iloc[idxs[0]]
    st.markdown("### 検索結果")
    for _, row in results.iterrows():
        st.subheader(f"条文 {row['article']} （施行日: {row['effective_date']}）")
        st.write(row['text'])

# フッター
st.sidebar.markdown("---")
st.sidebar.markdown("**使い方**\n1. dataフォルダにXMLを配置\n2. `streamlit run app.py` で起動\n")
