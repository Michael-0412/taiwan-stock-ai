import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta

st.set_page_config(page_title="台股選股器（含三大法人）", layout="wide")
st.title("📈 台股 AI 選股器（含三大法人買超加分）")

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
# 抓三大法人買超
# -------------------------------
def get_twse_investor_csv(date_str):
    try:
        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={date_str}&selectType=ALL"
        r = requests.get(url, timeout=10)
        r.encoding = 'utf-8'
        if r.status_code != 200:
            return None
        content = r.text
        lines = [line for line in content.split('\n') if len(line.split(',')) > 10]
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
# 整理法人資料（容錯版）
# -------------------------------
def parse_investor_df(df):
    if df is None or df.empty:
        return pd.DataFrame({'代號':[], 'inst_score':[]})  # 空 DataFrame
    
    df = df.rename(columns=lambda x: x.strip())
    col_map = {}
    for c in df.columns:
        if '證券代號' in c:
            col_map['證券代號'] = c
        elif '外資' in c and '買賣超' in c:
            col_map['外資買賣超股數'] = c
        elif '自營商' in c and '買賣超' in c:
            col_map['自營商買賣超股數'] = c
        elif '投信' in c and '買賣超' in c:
            col_map['投信買賣超股數'] = c

    if len(col_map) < 4:
        return pd.DataFrame({'代號':[], 'inst_score':[]})  # 欄位不完整
    
    df = df[list(col_map.values())]
    df.columns = ['證券代號','外資買賣超股數','自營商買賣超股數','投信買賣超股數']

    for col in ['外資買賣超股數','自營商買賣超股數','投信買賣超股數']:
        df[col] = df[col].astype(str).str.replace(',','').astype(float)
    
    df['代號'] = df['證券代號'].astype(str).str.replace('"','').str.strip()
    df['inst_score'] = df['外資買賣超股數'] + df['自營商買賣超股數'] + df['投信買賣超股數']
    max_inst = df['inst_score'].max()
    df['inst_score'] = df['inst_score'] / max_inst if max_inst>0 else 0.5
    return df[['代號','inst_score']]

# -------------------------------
# 計算分數
# -------------------------------
def calc_score(df_price, df_inst=None):
    df = df_price.copy()

    # 成交量正規化
    df['vol_score'] = (df['成交股數'] - df['成交股數'].min()) / (df['成交股數'].max() - df['成交股數'].min() + 1e-6)

    # 漲跌幅正規化 (用最近10日的漲跌)
    df['pct_chg'] = df['收盤價'].pct_change(periods=10).fillna(0)
    df['pct_score'] = (df['pct_chg'] - df['pct_chg'].min()) / (df['pct_chg'].max() - df['pct_chg'].min() + 1e-6)

    # 法人分數
    if df_inst is not None and not df_inst.empty:
        df = df.merge(df_inst, on='代號', how='left')
        df['inst_score'] = df['inst_score'].fillna(0.5)
    else:
        df['inst_score'] = 0.5

    # 總分
    df['score'] = 0.4*df['vol_score'] + 0.4*df['pct_score'] + 0.2*df['inst_score']
    return df

# -------------------------------
# 主程式
# -------------------------------
st.info("抓取 TWSE 當日資料中...")

twse_df, inst_df = None, None
trade_date = None
for date_str in get_recent_trade_date(max_days=14):
    twse_df = get_twse_daily_csv(date_str)
    inst_df = get_twse_investor_csv(date_str)
    if twse_df is not None:
        trade_date = date_str
        break

if twse_df is None:
    st.warning("無法抓取 TWSE 當日收盤資料，請稍後再試。")
    st.stop()

df_price = parse_twse_df(twse_df)
df_inst = parse_investor_df(inst_df)
st.success(f"資料日期（近似）: {trade_date}，共 {len(df_price)} 檔股票")

# 計算分數
df_score = calc_score(df_price, df_inst)
df_top10 = df_score.sort_values("score", ascending=False).head(10)

# 顯示結果
st.subheader("📊 今日前10名潛力股（含法人加分）")
st.dataframe(df_top10[['代號','證券名稱','收盤價','成交股數','score','inst_score']], use_container_width=True)
st.bar_chart(df_top10.set_index("代號")["score"])
