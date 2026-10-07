# メイン画面：app.py
import streamlit as st
from blink import run_blink_detection

st.title("ドライアイ防止アプリ")

st.info("6秒以上瞬きをしていないと「WARNING BLINK!」と表示されます")
run_blink_detection()