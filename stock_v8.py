import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta

st.set_page_config(page_title="台股選股器（含 RSI & 當日漲跌幅 & 三大法人）", layout="wide")
st.title("📈 台股 AI 選股器（含 RSI、漲跌幅 & 三大法人）")

# -------------------------------
# 抓取 TWSE 當日收盤 CSV
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
    df = df[['證券代號','證券名稱','成交股數','成交金額','收盤價','漲跌(+/-)','漲跌價差']].dropna()

    # 數字轉換
    df['成交股數'] = df['成交股數'].astype(str).str.replace(',','').astype(float)
    df['成交金額'] = df['成交金額'].astype(str).str.replace(',','').astype(float)
    df['收盤價'] = pd.to_numeric(df['收盤價'], errors='coerce')
    df['漲跌價差'] = pd.to_numeric(df['漲跌價差'].astype(str).str.replace(',',''), errors='coerce')

    # ✅ 加上正負號
    df['漲跌價差'] = df.apply(
        lambda x: x['漲跌價差'] if str(x['漲跌(+/-)']).strip() == '+' 
                  else (-x['漲跌價差']), axis=1
    )

    df['代號'] = df['證券代號'].astype(str).str.replace('"','').str.strip()
    return df[['代號','證券名稱','成交股數','成交金額','收盤價','漲跌價差']]

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
# 計算分數（熱門股簡單分數）
# -------------------------------
def calc_score(df_price):
    df = df_price.copy()
    df['vol_score'] = (df['成交金額'] - df['成交金額'].min()) / (df['成交金額'].max() - df['成交金額'].min() + 1e-6)
    df['pct_score'] = (df['pct_change'] - df['pct_change'].min()) / (df['pct_change'].max() - df['pct_change'].min() + 1e-6)
    df['score'] = 0.7*df['vol_score'] + 0.3*df['pct_score']
    return df

# -------------------------------
# 抓取三大法人買賣超
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
# 主程式
# -------------------------------
st.info("抓取 TWSE 當日資料中...")

# 抓收盤資料
twse_df = None
trade_date = None
for date_str in get_recent_trade_date(max_days=14):
    twse_df = get_twse_daily_csv(date_str)
    if twse_df is not None:
        trade_date = date_str
        break

if twse_df is None:
    st.warning("無法抓取 TWSE 當日收盤資料")
    st.stop()

df_price = parse_twse_df(twse_df)
df_price['pct_change'] = (df_price['漲跌價差'] / (df_price['收盤價'] - df_price['漲跌價差'])) * 100
df_price['rsi'] = compute_rsi(df_price['收盤價'])
df_score = calc_score(df_price)
df_top50 = df_score.sort_values("score", ascending=False).head(50)
df_top50['名次'] = range(1, len(df_top50)+1)

# 抓法人資料
df_inst = get_institutional_investors(trade_date)
if df_inst is not None:
    cols = ["外陸資買賣超股數(不含外資自營商)", "投信買賣超股數", "自營商買賣超股數"]
    for col in cols:
        if col in df_inst.columns:
            df_inst[col] = pd.to_numeric(df_inst[col].astype(str).str.replace(',',''), errors='coerce').fillna(0)
        else:
            df_inst[col] = 0

    df_top50 = df_top50.merge(df_inst[['證券名稱'] + cols], on='證券名稱', how='left')
    df_top50[cols] = df_top50[cols].fillna(0)

# -------------------------------
# 顯示表格
# -------------------------------
def highlight_rsi(val):
    if val > 70: return 'color:red'
    elif 50 <= val <= 70: return 'color:green'
    else: return 'color:black'

def highlight_pct(val):
    if val > 0: return 'color:red'
    elif val < 0: return 'color:green'
    else: return 'color:black'

def highlight_buy_sell(val):
    if val > 0: return 'color:red'
    elif val < 0: return 'color:green'
    else: return 'color:black'

st.subheader("📊 今日前50名熱門股（含 RSI & 漲跌幅 & 三大法人）")
st.dataframe(
    df_top50[['名次','代號','證券名稱','收盤價','pct_change','成交金額','rsi',
              '外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數','score']].style
        .applymap(highlight_rsi, subset=['rsi'])
        .applymap(highlight_pct, subset=['pct_change'])
        .applymap(highlight_buy_sell, subset=['外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數']),
    use_container_width=True
)

st.bar_chart(df_top50.set_index("代號")["score"])
