

import os
os.environ["STREAMLIT_WATCHDOG_DISABLE"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_WATCHDOG"] = "false"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"]     = "false"

from datetime import datetime, timedelta
import json

import requests
import pytz
import yfinance as yf
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf
import joblib
import gdown

from transformers import pipeline
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# ────────────────────────────────────────────────────────────────────────────────
# 0) STREAMLIT CONFIG
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Sentiment & Movement", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# 1) DOWNLOAD & LOAD MODELS + CONFIG
# ────────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "/tmp/FYP_Models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE_IDS = {
    "svm_model.pkl":          "1-68mv0vZlJRt9qhk-M2SBCZaeUB6DBT1",
    "rf_model.pkl":           "1--IBb2MPWufkGvighzRNl4E37x4ZaE-N",
    "gbm_model.pkl":          "1-CC_u_N3hPA6ZayMkVhIETpK-_SDNRC_",
    "ensemble_model.pkl":     "1YvNDbltoTMLTb7O33oiTmlqR3IY65Qns",
    "improved_lstm_model.h5": "1-NJMMwxSPJ0sqJHVMNrSoGFXDJ7xoCyP",
    "improved_cnn_model.h5":  "1-7_nbZA0uVlgxX2W7Of3f2M4n1KkpfPH",
    "feature_config.json":    "1DiROuG8KY5LupIOwI-lNIOT4YXnUkbYh",
    "imp_dl.pkl":             "1jayLETj4kIuapCMWo-CRPHr5xIQpXmkk",
    "scaler_dl.pkl":          "1aDcURAeSB9rd8MsdoJOdiN0KEGjQR4Co",
}

MODEL_MAP = {
    "Random Forest": "rf_model.pkl",
    "GBM":           "gbm_model.pkl",
    "SVM":           "svm_model.pkl",
    "Ensemble":      "ensemble_model.pkl",
    "CNN":           "improved_cnn_model.h5",
    "LSTM":          "improved_lstm_model.h5",
}

def download_if_needed(fname: str):
    dest = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(dest):
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_IDS[fname]}", dest, quiet=False)
    return dest

def load_model_file(name: str):
    path = download_if_needed(name)
    if name.endswith(".pkl"):
        return joblib.load(path)
    else:
        return tf.keras.models.load_model(path, compile=False, custom_objects={"InputLayer": InputLayer})

# feature config
cfg = json.load(open(download_if_needed("feature_config.json"), "r"))
TRAD_COLS, DL_COLS = cfg["trad_features"], cfg["dl_features"]
THRESHOLDS        = cfg["thresholds"]

# DL preprocessors
imp_dl    = joblib.load(download_if_needed("imp_dl.pkl"))
scaler_dl = joblib.load(download_if_needed("scaler_dl.pkl"))

# ────────────────────────────────────────────────────────────────────────────────
# 2) API CLASSES WITH FILTERED FALLBACK
# ────────────────────────────────────────────────────────────────────────────────
EST = pytz.timezone("US/Eastern")

class API:
    @staticmethod
    def get_news(ticker: str) -> pd.DataFrame:
        # 1) Try RapidAPI
        resp = requests.get(
            "https://mboum-finance.p.rapidapi.com/v1/markets/news",
            headers={
                "X-RapidAPI-Key":  "48c56bcad6msh9897ec585559d19p16c410jsn01e1017db67b",
                "X-RapidAPI-Host": "mboum-finance.p.rapidapi.com"
            },
            params={"symbol": ticker}
        ).json().get("body", [])
        rows = []
        for art in resp:
            dt    = datetime.strptime(art["pubDate"], "%a, %d %b %Y %H:%M:%S %z")
            rows.append([dt, art.get("title",""), art.get("description",""), art.get("link","")])
        df = pd.DataFrame(rows, columns=["Date Time","title","Description","link"])

        # 2) If empty, fallback to yfinance news—but **skip** providerPublishTime==0
        if df.empty:
            yf_news = yf.Ticker(ticker).news or []
            rows2 = []
            for art in yf_news:
                ts = art.get("providerPublishTime", 0)
                if not ts:
                    continue          # skip the epoch‐zero “news”
                dt = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(EST)
                rows2.append([dt, art.get("title",""), art.get("summary",""), art.get("link","")])
            df = pd.DataFrame(rows2, columns=["Date Time","title","Description","link"])

        # final cleanup & sort
        if not df.empty:
            df["Date Time"] = pd.to_datetime(df["Date Time"], utc=True).dt.tz_convert(EST)
            df.sort_values("Date Time", ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def get_price_history(ticker: str, earliest: datetime) -> pd.DataFrame:
        resp = requests.get(
            "https://mboum-finance.p.rapidapi.com/v1/markets/stock/history",
            headers={
                "X-RapidAPI-Key":  "…your key…",
                "X-RapidAPI-Host": "mboum-finance.p.rapidapi.com"
            },
            params={"symbol":ticker,"interval":"5m","diffandsplits":"false"}
        ).json().get("body", {})
        rows = []
        for rec in resp.values():
            dt = datetime.fromtimestamp(rec["date_utc"], tz=pytz.utc).astimezone(EST)
            if dt >= earliest:
                rows.append([dt, rec["open"], rec.get("volume",0)])
        df = pd.DataFrame(rows, columns=["Date Time","Price","Volume"])

        if df.empty:
            # fallback to last 7d via yfinance
            yf_df = yf.download(ticker, period="7d", interval="5m", auto_adjust=False)[["Open","Volume"]]
            # reset whatever index name it gave you into a real column
            yf_df = yf_df.reset_index()
            # grab that new first column name and remap it to "Date Time"
            date_col = yf_df.columns[0]
            yf_df = yf_df.rename(columns={ date_col: "Date Time", "Open": "Price" })

            # now consistently tz‐convert and typecast
            yf_df["Date Time"] = pd.to_datetime(yf_df["Date Time"], utc=True).dt.tz_convert(EST)
            yf_df["Price"]     = yf_df["Price"].astype(float)
            df = yf_df
        else:
            df.sort_values("Date Time", inplace=True)
            df.reset_index(drop=True, inplace=True)

        return df


class FinbertSentiment:
    def __init__(self):
        self.pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert",framework="pt")
        self.df   = pd.DataFrame()
    def set_data(self, df: pd.DataFrame):
        self.df = df.copy()
    def calc_sentiment_score(self):
        self.df["sentiment"] = self.df["title"].apply(self.pipe)
        self.df["sentiment_score"] = self.df["sentiment"].map(
            lambda x: (1 if x[0]["label"]=="positive" else -1 if x[0]["label"]=="negative" else 0)
                      * x[0]["score"]
        )
        self.df["sentiment_label"] = self.df["sentiment"].map(lambda x: x[0]["label"])
    def plot_sentiment(self):
        return px.bar(self.df, x="Date Time", y="sentiment_score", title="Sentiment over Time")

# ────────────────────────────────────────────────────────────────────────────────
# 3) FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────────────────────
def compute_features(news_df: pd.DataFrame, price_df: pd.DataFrame):
    base = {c: 0.0 for c in TRAD_COLS}
    if not price_df.empty:
        price_df = price_df.sort_values("Date Time")
        prices   = price_df["Price"].to_numpy().flatten()
        latest   = float(prices[-1])
        prev     = float(prices[-2]) if len(prices)>1 else latest
        lows, highs = prices.min(), prices.max()
        opens       = prices[0]
        volume      = float(price_df["Volume"].to_numpy().flatten()[-1])
        daily_ret   = (latest - prev)/prev if prev!=0 else 0.0
        ma10        = float(prices[-10:].mean()) if len(prices)>=10 else latest
        ma50        = float(prices[-50:].mean()) if len(prices)>=50 else latest
        rets        = pd.Series(prices).pct_change().fillna(0).to_numpy()
        volatility  = float(rets[-10:].std()) if len(rets)>=10 else 0.0
        close_scaled= (latest-lows)/(highs-lows) if highs>lows else 0.0
        price_delta = latest - prev
        base.update({
            "date_ord": int(price_df["Date Time"].iat[-1].toordinal()),
            "latest": latest, "high": highs, "low": lows,
            "open_p": opens,  "volume": volume, "daily_ret": daily_ret,
            "ma10": ma10,     "ma50": ma50,     "vol": volatility,
            "close_scaled": close_scaled, "price_delta": price_delta
        })
        for lag in (1,5,10):
            arr = pd.Series(prices).pct_change(lag).fillna(0).to_numpy()
            base[f"return_{lag}"] = float(arr[-1])
        delta = pd.Series(prices).diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        rs = up.rolling(14).mean() / down.rolling(14).mean()
        base["rsi_14"] = float((100 - (100/(1+rs))).iat[-1]) if len(rs)>=14 else 0.0
    now = datetime.now(EST)
    hr, dow = now.hour, now.weekday()
    base["hour_sin"], base["hour_cos"] = np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24)
    base["dow_sin"], base["dow_cos"]   = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    base["avg_sent"] = news_df["sentiment_score"].mean() if not news_df.empty else 0.0
    trad_vec = np.array([[base[c] for c in TRAD_COLS]], dtype=float)
    dl_vec   = np.array([[base[c] for c in DL_COLS]],   dtype=float)
    return trad_vec, dl_vec

# ────────────────────────────────────────────────────────────────────────────────
# 4) PREDICTION HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def predict_stock_movement(choice, num_feats, text_feats=None):
    mdl = load_model_file(MODEL_MAP[choice])
    if choice in ("CNN","LSTM"):
        if text_feats is None:
            text_feats = np.zeros((num_feats.shape[0],100), dtype=int)
        proba = mdl.predict([num_feats, text_feats]).flatten()
    else:
        proba = mdl.predict_proba(num_feats)[:,1]
    pred = (proba > THRESHOLDS.get(choice,0.5)).astype(int)
    return pred, proba

def plot_prediction_graph(proba):
    up, dn = float(proba[0]), 1-float(proba[0])
    fig = px.bar(x=["Down","Up"], y=[dn,up], title="Predicted Movement Probabilities")
    fig.update_layout(yaxis=dict(range=[0,1]))
    return fig

# ────────────────────────────────────────────────────────────────────────────────
# 5) STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📈 Stock Sentiment & Movement Prediction")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker","AAPL").upper().strip()
    with c2:
        model_choice = st.selectbox("Model", list(MODEL_MAP.keys()))

    if st.button("Analyze"):
        # News + sentiment
        news_df = API.get_news(ticker)
        if news_df.empty:
            st.warning("No recent news—using zeros & last 7 days of prices.")
            fs, padded = None, np.zeros((1,100), dtype=int)
        else:
            fs = FinbertSentiment()
            fs.set_data(news_df)
            fs.calc_sentiment_score()
            agg = (fs.df["title"]+" "+fs.df["Description"]).str.cat(sep=" ")
            tok = Tokenizer(num_words=10000)
            tok.fit_on_texts([agg])
            seq = tok.texts_to_sequences([agg])
            padded = pad_sequences(seq, maxlen=100, padding="post", truncating="post")

        # Price
        earliest = news_df["Date Time"].min() if not news_df.empty else datetime.now(pytz.utc)-timedelta(days=7)
        price_df = API.get_price_history(ticker, earliest)

        # Tabs
        tabs = st.tabs(["Sentiment","Headlines","Price","Prediction"])
        with tabs[0]:
            st.header("Sentiment Scores")
            if fs: st.plotly_chart(fs.plot_sentiment())
            else:  st.write("— no sentiment data —")
        with tabs[1]:
            st.header("Headlines")
            if fs:
                dfh = fs.df[["Date Time","title","sentiment_score","sentiment_label","link"]].copy()
                dfh["link"] = dfh["link"].apply(lambda u: f"[article]({u})")
                st.dataframe(dfh)
            else:
                st.write("— no headlines —")
        with tabs[2]:
            st.header("Price History")
            if price_df.empty:
                st.write("— no price data —")
            else:
                y = price_df["Price"].to_numpy().flatten()
                fig = px.line(x=price_df["Date Time"], y=y,
                              labels={"x":"Date Time","y":"Price"},
                              title=f"{ticker} Price")
                st.plotly_chart(fig)
        with tabs[3]:
            st.header("Model Prediction")
            trad_feats, dl_feats = compute_features(fs.df if fs else pd.DataFrame(), price_df)
            try:
                raw_vec    = dl_feats if model_choice in ("CNN","LSTM") else trad_feats
                imp_vec    = imp_dl.transform(raw_vec)
                vec_scaled = scaler_dl.transform(imp_vec)
                pred, proba = predict_stock_movement(model_choice, vec_scaled, padded)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                return

            direction = "UP" if int(pred[0])==1 else "DOWN"
            color     = "green" if direction=="UP" else "red"
            st.markdown(f"<h2 style='color:{color}'>{direction}</h2>", unsafe_allow_html=True)
            st.write(f"Probability Up:   {proba[0]:.2f}")
            st.write(f"Probability Down: {1-proba[0]:.2f}")
            st.plotly_chart(plot_prediction_graph(proba))

if __name__=="__main__":
    main()

