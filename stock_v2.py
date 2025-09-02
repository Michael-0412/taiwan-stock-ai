import streamlit as st
import pandas as pd
import requests
from io import StringIO
import datetime
import yfinance as yf

st.set_page_config(page_title="台股熱門股選股器", layout="wide")
st.title("📈 台股熱門股選股器（自動抓取 TWSE 股票清單）")

# -------------------------------
# 抓上市股票清單
# -------------------------------
@st.cache_data(ttl=3600)
def get_twse_listed_stocks():
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    r = requests.get(url)
    r.encoding = 'big5'
    df_list = pd.read_html(r.text)[0]
    df_list.columns = df_list.iloc[0]
    df_list = df_list[1:]
    df_list = df_list[['有價證券代號及名稱']].dropna()
    df_list['代號'] = df_list['有價證券代號及名稱'].str[:4] + ".TW"
    return df_list['代號'].tolist()

st.info("抓取 TWSE 股票清單中...")
tickers = get_twse_listed_stocks()
st.success(f"共抓到 {len(tickers)} 檔股票")

# -------------------------------
# 抓近 10 日收盤價
# -------------------------------
st.info("抓取近 10 日股價資料中...（視股票數量可能需 1~2 分鐘）")

start_date = (datetime.datetime.today() - datetime.timedelta(days=20)).strftime("%Y-%m-%d")
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

all_data = []
progress_text = st.empty()
for i, t in enumerate(tickers):
    try:
        df = yf.download(t, start=start_date, end=end_date, progress=False)
        if df.empty:
            continue
        df['Ticker'] = t
        df['PctChange'] = df['Close'].pct_change(10) * 100
        df['VolScore'] = df['Volume'] / df['Volume'].mean()
        df['Score'] = 0.7*df['VolScore'] + 0.3*df['PctChange'].fillna(0)
        all_data.append(df)
    except:
        continue
    if i % 50 == 0:
        progress_text.text(f"已處理 {i}/{len(tickers)} 檔股票")

progress_text.text("資料抓取完成！")

# -------------------------------
# 組成 DataFrame 並挑選前 10
# -------------------------------
if not all_data:
    st.warning("沒有抓到任何股價資料")
else:
    df_all = pd.concat(all_data)
    df_last = df_all.groupby('Ticker').tail(1)
    df_top10 = df_last.sort_values('Score', ascending=False).head(10)

    st.subheader("🔥 今日熱門股 Top 10")
    st.dataframe(df_top10[['Ticker','Close','Volume','PctChange','Score']])

    st.subheader("📊 前 10 股分數長條圖")
    st.bar_chart(df_top10.set_index("Ticker")["Score"])
