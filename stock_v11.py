import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta

st.set_page_config(page_title="台股選股器（含 RSI & 三大法人）", layout="wide")
st.title("📈 台股 AI 選股器（收盤資料 + 三大法人 + 官網連結）")

# -------------------------------
# 最近交易日
# -------------------------------
def get_recent_trade_date(max_days=14):
    for i in range(max_days):
        dt = datetime.now() - timedelta(days=i)
        yield dt.strftime("%Y%m%d")

# -------------------------------
# 抓取 TWSE 收盤 CSV
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
# 抓三大法人資料
# -------------------------------
def get_institutional_investors(date_str):
    url = f"https://www.twse.com.tw/fund/T86?response=json&date={date_str}&selectType=ALLBUT0999"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    if "data" not in data or len(data["data"]) == 0:
        return None
    df = pd.DataFrame(data["data"], columns=[x.strip() for x in data["fields"]])
    return df

# -------------------------------
# 計算 RSI
# -------------------------------
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    rsi = 100 - 100 / (1 + rs)
    return rsi

# -------------------------------
# 計算漲跌幅 %
# -------------------------------
def calc_pct_change(df):
    df['漲跌符號'] = df['漲跌(+/-)'].map({'+':1, '-':-1}).fillna(0)
    df['漲跌價差數值'] = df['漲跌價差'].astype(str).str.replace('+','').str.replace('-','').str.replace(',','')
    df['漲跌價差數值'] = pd.to_numeric(df['漲跌價差數值'], errors='coerce').fillna(0)
    df['pct_change'] = df['漲跌符號'] * df['漲跌價差數值'] / (df['收盤價'] - df['漲跌符號'] * df['漲跌價差數值'] + 1e-6) * 100
    return df

# -------------------------------
# 計算熱門股分數
# -------------------------------
def calc_score(df):
    df = df.copy()
    df['vol_score'] = (df['成交金額'] - df['成交金額'].min()) / \
                      (df['成交金額'].max() - df['成交金額'].min() + 1e-6)
    df['pct_score'] = (df['pct_change'] - df['pct_change'].min()) / \
                      (df['pct_change'].max() - df['pct_change'].min() + 1e-6)
    df['inst_score'] = (df['外陸資買賣超股數(不含外資自營商)']*4 +
                        df['投信買賣超股數']*4 +
                        df['自營商買賣超股數']*2) / 10
    df['三大法人加權合計'] = df['外陸資買賣超股數(不含外資自營商)'] + \
                             df['投信買賣超股數'] + df['自營商買賣超股數']
    df['score'] = 0.5*df['vol_score'] + 0.2*df['pct_score'] + 0.3*(df['inst_score']/df['inst_score'].max())
    return df

# -------------------------------
# 主程式
# -------------------------------
st.info("抓取 TWSE 收盤資料與三大法人資料...")

twse_df = None
df_inst = None
trade_date = None

for date_str in get_recent_trade_date(14):
    twse_df = get_twse_daily_csv(date_str)
    df_inst = get_institutional_investors(date_str)
    if twse_df is not None and df_inst is not None:
        trade_date = date_str
        break

if twse_df is None or df_inst is None:
    st.warning("⚠️ 無法抓取 TWSE 收盤資料或三大法人資料")
    st.stop()

st.success(f"資料日期：{trade_date}")

# -------------------------------
# 整理收盤資料
# -------------------------------
twse_df = twse_df.rename(columns=lambda x: x.strip())
for col in ['成交股數','成交金額','收盤價','漲跌價差']:
    twse_df[col] = pd.to_numeric(twse_df[col].astype(str).str.replace(',',''), errors='coerce').fillna(0)

df_price = twse_df[['證券代號','證券名稱','成交股數','成交金額','收盤價','漲跌價差','漲跌(+/-)']].copy()
df_price = calc_pct_change(df_price)
df_price['rsi'] = compute_rsi(df_price['收盤價'])

# -------------------------------
# 整理三大法人資料
# -------------------------------
cols_inst = ["外陸資買賣超股數(不含外資自營商)","投信買賣超股數","自營商買賣超股數"]
df_inst = df_inst.rename(columns=lambda x: x.strip())
for col in cols_inst:
    if col in df_inst.columns:
        df_inst[col] = pd.to_numeric(df_inst[col].astype(str).str.replace(',',''), errors='coerce').fillna(0)
    else:
        df_inst[col] = 0

# -------------------------------
# 合併資料
# -------------------------------
df_merged = pd.merge(df_price, df_inst[['證券代號']+cols_inst], on='證券代號', how='left')
df_merged[cols_inst] = df_merged[cols_inst].fillna(0)

df_merged = calc_score(df_merged)
df_top50 = df_merged.sort_values("score", ascending=False).head(50)
df_top50['名次'] = range(1, len(df_top50)+1)

# -------------------------------
# 載入上市公司官網資料
# -------------------------------
company_info = pd.read_csv("data/t187ap03_L.csv", dtype={"公司代號": str})
company_info = company_info[['公司代號', '公司名稱', '網址']]

# 合併官網
df_top50 = df_top50.merge(company_info, left_on="證券代號", right_on="公司代號", how="left")

# 自動補 https:// 前綴
def make_link(url, name):
    if pd.isna(url) or url.strip() == "":
        return name
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return f'<a href="{url}" target="_blank">{name}</a>'

df_top50["公司連結"] = df_top50.apply(lambda r: make_link(r["網址"], r["證券名稱"]), axis=1)

# -------------------------------
# 顯示表格
# -------------------------------
show_cols = ['名次','證券代號','公司連結','收盤價','pct_change','成交金額','rsi'] + \
            cols_inst + ['三大法人加權合計','inst_score','score']

# Styler
styler = df_top50[show_cols].style.format({
    "pct_change": "{:.2f}%",
    "rsi": "{:.1f}",
    "收盤價": "{:.2f}",
    "成交金額": "{:,}",
    "三大法人加權合計": "{:,}",
    "inst_score": "{:.2f}",
    "score": "{:.2f}"
})

styler = styler.background_gradient(subset=['score'], cmap='Reds') \
               .background_gradient(subset=['成交金額'], cmap='Blues') \
               .map(lambda v: 'color:red' if isinstance(v,(int,float)) and v>0 else 'color:green',
                    subset=['pct_change','rsi']+cols_inst+['三大法人加權合計'])

st.subheader("📊 前50名熱門股（收盤資料 + 三大法人 + 官網連結）")
st.markdown(styler.to_html(escape=False), unsafe_allow_html=True)

# -------------------------------
# 顯示分數圖表
# -------------------------------
st.bar_chart(df_top50.set_index("證券代號")["score"])
