# 消費税法ベクトル検索アプリ

Streamlit と Sentence-Transformers + FAISS を使った、
消費税法関連 XML の全文ベクトル検索＆絞り込みアプリ。

## 構成

- `app.py` : Streamlit アプリ本体
- `data/`   : XML ファイル格納フォルダ
- `requirements.txt` : 必要パッケージ
- `.streamlit/config.toml` : Streamlit 設定
- `.github/workflows/ci.yml` : CI 用

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
