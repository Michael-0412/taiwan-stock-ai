import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ===============================
# 抓取 TWSE 上市股票清單
# ===============================
def get_all_twse_symbols():
    url = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
    try:
        r = requests.get(url, verify=False)
        tables = pd.read_html(r.text)
        df = tables[0]
        df = df.iloc[:,0:2]
        df.columns = ["代號名稱","分類"]
        df = df.dropna()
        stocks = []
        for row in df["代號名稱"]:
            if "　" in row:
                code, name = row.split("　",1)
                if code.isdigit():
                    stocks.append(f"{code}.TW")
        return stocks
    except Exception as e:
        print("抓取上市股票失敗:", e)
        return ["2330.TW","2317.TW","2454.TW","2303.TW","2882.TW","2881.TW"]

# ===============================
# 抓取 TWSE 三大法人資料
# ===============================
def find_recent_twse_date(max_lookback_days=7):
    cur = datetime.now()
    for d in range(max_lookback_days):
        yield (cur - timedelta(days=d)).strftime("%Y%m%d")

def get_twse_three_major_robust(max_lookback_days=7, max_retries=2, sleep_sec=0.5):
    base_url = "https://www.twse.com.tw/fund/T86"
    headers = {"User-Agent": "Mozilla/5.0"}
    for date_str in find_recent_twse_date(max_lookback_days):
        params = {"response":"json", "date":date_str, "selectType":"ALLBUT0999"}
        for attempt in range(max_retries):
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=8)
                if r.status_code != 200:
                    time.sleep(sleep_sec)
                    continue
                j = r.json()
                if j.get("data"):
                    cols = j.get("fields") or j.get("columns") or []
                    df = pd.DataFrame(j["data"], columns=cols)
                    df["_twse_date"] = date_str
                    return df
                else:
                    break
            except Exception:
                time.sleep(sleep_sec)
                continue
    return None

def parse_twse_three_major_to_dict(df):
    res = {}
    if df is None:
        return res
    cols = list(df.columns)
    code_col = next((c for c in ['證券代號','證券代碼','代碼','symbol','stock_id'] if c in cols), None)
    if code_col is None:
        return res
    def find_col(names):
        for n in names:
            if n in cols:
                return n
        return None
    ext_buy_col = find_col(['外資買賣超股數','外資買賣超','外資買賣超(股)'])
    inv_buy_col = find_col(['投信買賣超股數','投信買賣超','投信買賣超(股)'])
    dealer_buy_col = find_col(['自營商買賣超股數','自營商買賣超','自營商買賣超(股)'])
    def parse_num(v):
        try:
            if pd.isna(v): return 0
            if isinstance(v, str):
                return int(float(v.replace(',','').replace('--','0').strip()))
            return int(float(v))
        except:
            return 0
    for _, row in df.iterrows():
        code = str(row[code_col]).strip()
        ext = parse_num(row[ext_buy_col]) if ext_buy_col else 0
        inv = parse_num(row[inv_buy_col]) if inv_buy_col else 0
        dealer = parse_num(row[dealer_buy_col]) if dealer_buy_col else 0
        total = ext + inv + dealer
        res[code] = {"外資": ext, "投信": inv, "自營商": dealer, "total": total}
    return res

# ===============================
# 股票打分邏輯
# ===============================
def score_stock(sym, three_dict):
    try:
        df = yf.download(sym, period="15d", interval="1d", progress=False)
        if df.empty: return None, None
        df = df.tail(10)
        avg_vol = df["Volume"].mean()
        vol_score = min(1.0, df["Volume"].iloc[-1] / avg_vol) if avg_vol > 0 else 0
        ma5 = df["Close"].rolling(5).mean().iloc[-1]
        ma10 = df["Close"].rolling(10).mean().iloc[-1]
        close = df["Close"].iloc[-1]
        ma_score = 1.0 if close > ma5 > ma10 else 0.5 if close > ma5 else 0.0
        pct_chg = (close - df["Close"].iloc[0]) / df["Close"].iloc[0] * 100
        pct_score = 1.0 if pct_chg > 5 else 0.5 if pct_chg > 0 else 0.0
        code = sym.split('.')[0]
        inst_entry = three_dict.get(code)
        inst_score = 0.0
        if inst_entry:
            total = inst_entry["total"]
            if total > 0:
                inst_score = min(1.0, total / 1000.0)
        total_score = 0.35*vol_score + 0.25*ma_score + 0.2*pct_score + 0.2*inst_score
        return {"symbol": sym, "score": total_score, "vol": vol_score, "ma": ma_score,
                "pct": pct_score, "inst": inst_score, "pct_chg": pct_chg, "close": close}
    except:
        return None

# ===============================
# Streamlit 介面
# ===============================
st.set_page_config(page_title="台股選股器", layout="wide")
st.title("📈 台股 AI 選股器")

# --- 三大法人 ---
with st.spinner("抓取三大法人資料中..."):
    three_df = get_twse_three_major_robust()
    three_dict = parse_twse_three_major_to_dict(three_df)
    if three_df is not None:
        st.success(f"三大法人資料日期：{three_df['_twse_date'].iloc[0]}")
    else:
        st.warning("無法取得三大法人資料，將只用價量指標。")

# --- 股票清單 ---
with st.spinner("抓取上市股票清單中..."):
    symbols = get_all_twse_symbols()
    st.write(f"共找到 {len(symbols)} 檔股票")

# --- 多線程抓股價 ---
results = []
max_try = 500
count_placeholder = st.empty()
with st.spinner("計算股票分數中..."):
    def worker(sym):
        return score_stock(sym, three_dict)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(worker, s): s for s in symbols[:max_try]}
        count = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            count += 1
            count_placeholder.text(f"已處理 {count}/{min(max_try,len(symbols))} 檔股票...")

# --- 顯示前10 ---
if results:
    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False).head(10)
    st.subheader("今日前10名潛力股")
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index("symbol")["score"])
else:
    st.warning("沒有可用的股票資料，請稍後再試或檢查網路")
