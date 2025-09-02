import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="台股選股器（熱門股選股）", layout="wide")
st.title("📈 台股 AI 選股器（熱門股選股）")

# -------------------------------
# 抓 TWSE 當日收盤 CSV
# -------------------------------
def get_twse_daily_csv(date_str):
    try:
        url = f"https://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date={date_str}&type=ALLBUT0999"
        r = requests.get(url, timeout=10)
        r.encoding = 'big5'
        if r.status_code != 200:
            return None
        content = r.text
        lines = [line for line in content.split('\n') if len(line.split('","')) > 10]
        if len(lines) == 0:
            return None
        csv_data = "\n".join(lines)
        df = pd.read_csv(StringIO(csv_data))
        return df
    except:
        return None

# -------------------------------
# 取得最近交易日
# -------------------------------
def get_recent_trade_date(max_days=14):
    for i in range(max_days):
        dt = datetime.now() - pd.Timedelta(days=i)
        yield dt.strftime("%Y%m%d")

# -------------------------------
# 整理收盤資料
# -------------------------------
def parse_twse_df(df):
    df = df.rename(columns=lambda x: x.strip())
    df = df[['證券代號','證券名稱','成交股數','收盤價']].dropna()
    df['成交股數'] = df['成交股數'].astype(str).str.replace(',','').astype(float)
    df['收盤價'] = pd.to_numeric(df['收盤價'], errors='coerce')
    df = df.dropna(subset=['收盤價'])
    df['代號'] = df['證券代號'].astype(str).str.replace('"','').str.strip()
    return df[['代號','證券名稱','成交股數','收盤價']]

# -------------------------------
# 計算 RSI
# -------------------------------
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta).clip(lower=0).rolling(period).mean()
    rsi = 100 * gain / (gain + loss + 1e-6)
    return rsi

# -------------------------------
# 計算分數（熱門股）
# -------------------------------
def calc_score(df):
    df = df.copy()
    # 成交量 score
    df['vol_score'] = (df['成交股數'] - df['成交股數'].min()) / (df['成交股數'].max() - df['成交股數'].min() + 1e-6)
    
    # 漲幅 score (近5日漲幅)
    df['pct_chg_5'] = df['收盤價'].pct_change(periods=5).fillna(0)
    df['pct_score'] = (df['pct_chg_5'] - df['pct_chg_5'].min()) / (df['pct_chg_5'].max() - df['pct_chg_5'].min() + 1e-6)
    
    # RSI score
    df['rsi'] = calc_rsi(df['收盤價']).fillna(50)
    df['rsi_score'] = df['rsi'] / 100
    
    # 總分（加權）
    df['score'] = 0.4*df['vol_score'] + 0.4*df['pct_score'] + 0.2*df['rsi_score']
    return df

# -------------------------------
# 主程式
# -------------------------------
st.info("抓取 TWSE 當日資料中...")

twse_df = None
trade_date = None
for date_str in get_recent_trade_date(max_days=14):
    twse_df = get_twse_daily_csv(date_str)
    if twse_df is not None:
        trade_date = date_str
        break

if twse_df is None:
    st.warning("無法抓取 TWSE 當日收盤資料，請稍後再試。")
    st.stop()

df_price = parse_twse_df(twse_df)
st.success(f"資料日期（近似）: {trade_date}，共 {len(df_price)} 檔股票")

# 計算分數
df_score = calc_score(df_price)
df_top20 = df_score.sort_values("score", ascending=False).head(20)

# -------------------------------
# 顯示結果（帶 RSI 顏色）
# -------------------------------
def highlight_rsi(val):
    if val > 70:
        color = 'red'      # 過熱
    elif val < 30:
        color = 'green'    # 過冷
    else:
        color = 'black'    # 中性
    return f'color: {color}'

st.subheader("📊 今日前20名熱門股（含 RSI）")
st.dataframe(
    df_top20[['代號','證券名稱','收盤價','成交股數','rsi','score']].style.applymap(highlight_rsi, subset=['rsi']),
    use_container_width=True
)
st.bar_chart(df_top20.set_index("代號")["score"])

# 顯示結果
st.subheader("📊 今日前20名熱門股（含 RSI）")
st.dataframe(df_top20[['代號','證券名稱','收盤價','成交股數','rsi','score']], use_container_width=True)
st.bar_chart(df_top20.set_index("代號")["score"])
