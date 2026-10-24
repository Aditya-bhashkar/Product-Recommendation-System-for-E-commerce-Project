"""
RecoPulse - Product Recommendation Analytics
Foundations of Data Science | Assignment 3

Run:  streamlit run app.py
Keep Product_Recommendation_Cleaned.csv in the same folder (or upload it in the sidebar).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="RecoPulse | Product Recommendation Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Palette + constants
# ============================================================================
NAVY, INDIGO, TEAL, AMBER, CORAL, SKY, VIOLET, SLATE = (
    "#0b1437", "#5b5bf0", "#19c3b1", "#ffb020", "#ff6b6b", "#38bdf8", "#9b7bff", "#94a3b8",
)
CAT_COLORS = [INDIGO, TEAL, AMBER, CORAL, SKY, VIOLET, SLATE]
SEQ = ["#eef0ff", "#aab0ff", INDIGO, "#2a2a9e"]

SAT_ORDER = ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"]
SAT_COLORS = {"Very Satisfied": TEAL, "Satisfied": SKY, "Neutral": SLATE, "Dissatisfied": AMBER, "Very Dissatisfied": CORAL}
FREQ_ORDER = ["Always", "Often", "Sometimes", "Rarely", "Never"]
FREQ_COLORS = {"Always": INDIGO, "Often": VIOLET, "Sometimes": SKY, "Rarely": AMBER, "Never": CORAL}
POS_SAT = {"Very Satisfied", "Satisfied"}
ADOPT = {"Always", "Often", "Sometimes"}
NA_LABEL = "Not recorded"

# column key -> phrases to look for in the (normalised) CSV header
SPEC = {
    "timestamp": ["timestamp"],
    "reg": ["registration"],
    "email": ["email"],
    "age": ["age group"],
    "gender": ["gender"],
    "occupation": ["occupation"],
    "income": ["income"],
    "shop_freq": ["how often do you shop"],
    "device": ["device"],
    "platform": ["e commerce platform"],
    "spending": ["monthly spending"],
    "category": ["product category"],
    "factor": ["important factor"],
    "discovery": ["discover new products"],
    "rec_freq": ["based on recommendations"],
    "payment": ["payment method"],
    "satisfaction": ["how satisfied"],
    "returned": ["returned a product"],
}
LABELS = {
    "age": "Age group", "gender": "Gender", "occupation": "Occupation", "income": "Monthly income",
    "shop_freq": "Shopping frequency", "device": "Device", "platform": "Platform",
    "spending": "Monthly spending", "category": "Top product category", "factor": "Key buying factor",
    "discovery": "Discovery channel", "rec_freq": "Buys via recommendations", "payment": "Payment method",
    "satisfaction": "Satisfaction with recommendations", "returned": "Returned a product",
    "reg": "Registration no.", "timestamp": "Timestamp",
}
ORDINAL = {"age", "income", "shop_freq", "spending", "satisfaction", "rec_freq"}
CORE = ["category", "satisfaction", "rec_freq"]

# ============================================================================
# Styling
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#0b1437; --muted:#5d6b88; --indigo:#5b5bf0; --teal:#19c3b1; --amber:#ffb020; --coral:#ff6b6b; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:
  radial-gradient(circle at 92% -5%, rgba(91,91,240,.10), transparent 30%),
  radial-gradient(circle at -5% 40%, rgba(25,195,177,.07), transparent 28%), #f5f7fd; color:var(--ink); }
.block-container { padding-top:1.2rem; padding-bottom:2.5rem; max-width:1520px; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"], [data-testid="stToolbar"] { visibility:hidden; }

/* sidebar */
[data-testid="stSidebar"] { background:linear-gradient(185deg,#0b1437 0%,#141f55 60%,#1d2a73 100%); }
[data-testid="stSidebar"] * { color:#eef1ff !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div { background:rgba(255,255,255,.96); }
[data-testid="stSidebar"] [data-baseweb="select"] * { color:#0b1437 !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] { background:#5b5bf0 !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] * { color:#fff !important; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:rgba(255,255,255,.07); border:1px dashed rgba(255,255,255,.35); }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background:#fff; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * { color:#0b1437 !important; }
[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.14); }
[data-testid="stSidebar"] .stButton button { background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.4); }
[data-testid="stSidebar"] .stButton button:hover { background:rgba(255,255,255,.22); border-color:#fff; }
.brand { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.45rem; letter-spacing:-.03em; }
.brand-sub { font-size:.68rem; letter-spacing:.16em; opacity:.7; margin-top:-4px; }

/* hero */
.hero { position:relative; overflow:hidden; border-radius:26px; padding:30px 36px; margin:0 0 22px 0; color:#fff;
  background:linear-gradient(118deg,#0b1437 0%,#1b2a78 52%,#5b5bf0 100%);
  box-shadow:0 18px 44px rgba(33,40,120,.22); animation:riseIn .6s ease-out both; }
.hero:before { content:""; position:absolute; width:340px; height:340px; right:-80px; top:-150px; border-radius:50%;
  background:radial-gradient(circle,rgba(25,195,177,.45),transparent 65%); }
.hero:after { content:""; position:absolute; width:260px; height:260px; right:120px; bottom:-170px; border-radius:50%;
  border:1px solid rgba(255,255,255,.22); box-shadow:0 0 0 26px rgba(255,255,255,.04),0 0 0 54px rgba(255,255,255,.03); }
.hero-tag { display:inline-block; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25);
  padding:5px 12px; border-radius:999px; font-size:.72rem; letter-spacing:.08em; margin-bottom:12px; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(1.8rem,3vw,2.7rem); line-height:1.1; margin:0 0 8px 0; font-weight:800; letter-spacing:-.04em; color:#fff !important; position:relative; z-index:1; }
.hero h1 .accent { color:#7ff0e0 !important; }
.hero p { margin:0; max-width:780px; color:#d6defc; font-size:1.02rem; position:relative; z-index:1; }
.hero-meta { margin-top:16px; display:flex; gap:10px; flex-wrap:wrap; position:relative; z-index:1; }
.chip { background:rgba(255,255,255,.1); border:1px solid rgba(255,255,255,.2); border-radius:999px; padding:4px 12px; font-size:.78rem; color:#eaf0ff; }

/* section heads */
.section-head { font-family:'Manrope',sans-serif; font-size:1.12rem; font-weight:800; color:var(--ink); margin:.3rem 0 .1rem 0; letter-spacing:-.02em; }
.section-sub { color:var(--muted); font-size:.86rem; margin-bottom:.6rem; }

/* KPI cards */
.kpi { position:relative; background:#fff; border:1px solid #e3e8f6; border-radius:20px; padding:18px 20px 16px 20px;
  box-shadow:0 8px 24px rgba(30,40,110,.06); overflow:hidden; transition:transform .2s ease, box-shadow .2s ease; animation:riseIn .6s ease-out both; }
.kpi:before { content:""; position:absolute; left:0; top:0; height:100%; width:5px; background:var(--accent); }
.kpi:hover { transform:translateY(-4px); box-shadow:0 14px 30px rgba(30,40,110,.12); }
.kpi-label { font-size:.78rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); }
.kpi-value { font-family:'Manrope',sans-serif; font-weight:800; font-size:2.15rem; letter-spacing:-.04em; line-height:1.15; margin:4px 0 2px 0; color:var(--ink); }
.kpi-value.small { font-size:1.45rem; padding-top:.45rem; padding-bottom:.35rem; }
.kpi-sub { font-size:.82rem; color:var(--muted); }
.up { color:#0f9d8a; font-weight:700; } .down { color:#e5484d; font-weight:700; } .flat { color:#7b88a6; font-weight:700; }

/* insight chips */
.insight { background:#fff; border:1px solid #e3e8f6; border-radius:16px; padding:13px 16px; font-size:.9rem; color:#2a3556;
  box-shadow:0 5px 16px rgba(30,40,110,.04); height:100%; }
.insight b { color:var(--ink); }
.insight .ico { font-size:1.15rem; margin-right:6px; }

/* charts / tables */
div[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e3e8f6; border-radius:20px; padding:8px; box-shadow:0 8px 22px rgba(30,40,110,.045); }
div[data-testid="stDataFrame"] { border:1px solid #e3e8f6; border-radius:14px; overflow:hidden; }
div[data-testid="stExpander"] { border:1px solid #e3e8f6; border-radius:16px; background:rgba(255,255,255,.85); }

/* tabs as pills */
div[data-testid="stTabs"] [role="tablist"] { gap:6px; border-bottom:none; background:#e9edfb; padding:5px; border-radius:16px; width:fit-content; flex-wrap:wrap; }
div[data-testid="stTabs"] button[role="tab"] { border-radius:12px; padding:8px 16px; font-weight:700; color:#4b5877; height:auto; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background:#fff; color:var(--indigo); box-shadow:0 3px 10px rgba(30,40,110,.12); }
div[data-testid="stTabs"] [data-baseweb="tab-highlight"], div[data-testid="stTabs"] [data-baseweb="tab-border"] { display:none; }

/* misc */
.note { border-left:4px solid var(--indigo); background:#eceeff; border-radius:10px; padding:12px 16px; color:#2f3a63; font-size:.88rem; }
.warn { border-left:4px solid var(--amber); background:#fff6e2; border-radius:10px; padding:12px 16px; color:#5a4410; font-size:.88rem; }
.simbadge { display:inline-block; background:#fff1d6; color:#8a5a00; border:1px solid #ffd58a; padding:3px 10px; border-radius:999px; font-size:.74rem; font-weight:700; letter-spacing:.06em; }
.step { background:#fff; border:1px solid #e3e8f6; border-radius:20px; padding:18px 20px; box-shadow:0 8px 22px rgba(30,40,110,.05); height:100%; }
.step .time { display:inline-block; background:var(--indigo); color:#fff; font-weight:700; font-size:.74rem; letter-spacing:.06em; padding:3px 10px; border-radius:999px; margin-bottom:8px; }
.step h4 { font-family:'Manrope',sans-serif; margin:0 0 6px 0; font-size:1.05rem; color:var(--ink); }
.step ul { margin:0; padding-left:1.1rem; color:#2a3556; font-size:.9rem; line-height:1.55; }
@keyframes riseIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helpers
# ============================================================================
_VER = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
_NEW_API = _VER >= (1, 50)
_chart_n = [0]


def show(fig):
    _chart_n[0] += 1
    if _NEW_API:
        st.plotly_chart(fig, width="stretch", key=f"pc{_chart_n[0]}")
    else:
        st.plotly_chart(fig, use_container_width=True, key=f"pc{_chart_n[0]}")


def show_table(df, height=420):
    if _NEW_API:
        st.dataframe(df, width="stretch", height=height)
    else:
        st.dataframe(df, use_container_width=True, height=height)


def head(title, sub=""):
    st.markdown(f'<div class="section-head">{title}</div><div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def style_fig(fig, height=360):
    fs = 15 if st.session_state.get("big_mode") else 12.5
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=16, r=16, t=40, b=16),
        font=dict(family="DM Sans, sans-serif", color="#263554", size=fs),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="#0b1437", font_color="#fff", font_size=fs),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf0f8", zerolinecolor="#e1e6f2")
    fig.update_yaxes(showgrid=True, gridcolor="#edf0f8", zerolinecolor="#e1e6f2")
    return fig


@st.cache_data(show_spinner=False)
def prepare(raw: pd.DataFrame):
    """Map messy survey headers to short keys, strip 'A) ' prefixes, remember answer order."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    norm = {c: normalize_name(c) for c in df.columns}
    mapping, used = {}, set()
    for key, needles in SPEC.items():
        for col, n in norm.items():
            if col not in used and any(nd in n for nd in needles):
                mapping[key] = col
                used.add(col)
                break
    out, order = pd.DataFrame(index=df.index), {}
    for key, col in mapping.items():
        s = df[col]
        if key == "email":
            continue  # privacy: never carried into the dashboard
        if key == "timestamp":
            out[key] = pd.to_datetime(s, errors="coerce")
            continue
        txt = s.astype("string").str.strip()
        letters = txt.str.extract(r"^([A-Za-z])\)\s*")[0]
        clean = txt.str.replace(r"^[A-Za-z]\)\s*", "", regex=True).str.strip().replace("", pd.NA)
        out[key] = clean.astype(object)
        if key in ORDINAL and letters.notna().any():
            pairs = pd.DataFrame({"l": letters, "c": clean}).dropna().drop_duplicates("c").sort_values("l")
            order[key] = pairs["c"].tolist()
    missing = [k for k in CORE if k not in out.columns]
    if missing:
        return None, order, missing
    if "reg" not in out.columns:
        out["reg"] = [f"R{i+1}" for i in range(len(out))]
    if "timestamp" not in out.columns:
        out["timestamp"] = pd.NaT
    for c in out.columns:
        if c in ("timestamp",):
            continue
        out[c] = out[c].fillna(NA_LABEL if c in ("satisfaction", "rec_freq") else "Unknown").astype(str)
    out = out[out["category"] != "Unknown"].reset_index(drop=True)
    return out, order, []


def load_csv(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    here = Path(__file__).resolve().parent
    for base in (here, Path.cwd()):
        for name in ("Product_Recommendation_Cleaned.csv", "QP2_Integrated_Dataset.csv"):
            p = base / name
            if p.exists():
                return pd.read_csv(p)
    return None


def pos_rate(df):
    v = df[df["satisfaction"] != NA_LABEL]
    return 100 * v["satisfaction"].isin(POS_SAT).mean() if len(v) else float("nan")


def adopt_rate(df):
    v = df[df["rec_freq"] != NA_LABEL]
    return 100 * v["rec_freq"].isin(ADOPT).mean() if len(v) else float("nan")


def delta_html(cur, base, filtered_on):
    if not filtered_on or np.isnan(cur) or np.isnan(base):
        return "all respondents"
    d = cur - base
    cls = "up" if d > 0.5 else "down" if d < -0.5 else "flat"
    arrow = "▲" if d > 0.5 else "▼" if d < -0.5 else "●"
    return f'<span class="{cls}">{arrow} {abs(d):.1f} pts</span> vs all respondents'


def kpi(label, value, sub, accent, tip, small=False):
    cls = "kpi-value small" if small else "kpi-value"
    return (f'<div class="kpi" style="--accent:{accent}" title="{tip}"><div class="kpi-label">{label}</div>'
            f'<div class="{cls}">{value}</div><div class="kpi-sub">{sub}</div></div>')


def ordered(key, values):
    values = list(dict.fromkeys(values))
    ref = order_map.get(key) or (SAT_ORDER if key == "satisfaction" else FREQ_ORDER if key == "rec_freq" else None)
    if ref:
        return [v for v in ref if v in values] + sorted(v for v in values if v not in ref)
    if key in data.columns:  # otherwise by popularity
        pop = data[key].value_counts().index.tolist()
        return [v for v in pop if v in values] + sorted(v for v in values if v not in pop)
    return sorted(values)


def rate_bar(df, dim, kind, height=330):
    """Horizontal bar of % positive satisfaction (kind='sat') or % rec-driven buying (kind='rec') by a dimension."""
    if kind == "sat":
        sub = df[df["satisfaction"] != NA_LABEL]
        flag, overall = sub["satisfaction"].isin(POS_SAT), pos_rate(df)
    else:
        sub = df[df["rec_freq"] != NA_LABEL]
        flag, overall = sub["rec_freq"].isin(ADOPT), adopt_rate(df)
    g = pd.DataFrame({"dim": sub[dim], "flag": flag}).groupby("dim")["flag"].agg(["mean", "size"]).reset_index()
    g.columns = ["dim", "rate", "n"]
    g["rate"] *= 100
    g = g.sort_values("rate")
    g["label"] = [f"{r:.0f}%  (n={n}){' ⚠' if n < 5 else ''}" for r, n in zip(g["rate"], g["n"])]
    fig = go.Figure(go.Bar(
        x=g["rate"], y=g["dim"], orientation="h", text=g["label"], textposition="outside", cliponaxis=False,
        marker=dict(color=g["rate"], colorscale=[[0, "#c9cffd"], [1, INDIGO]], cmin=0, cmax=100),
        customdata=g["n"], hovertemplate="<b>%{y}</b><br>Rate: %{x:.1f}%<br>Respondents: %{customdata}<extra></extra>",
    ))
    fig.add_vline(x=overall, line_dash="dash", line_color=CORAL,
                  annotation_text=f"overall {overall:.0f}%", annotation_position="bottom right", annotation_font_color=CORAL)
    fig.update_layout(xaxis=dict(range=[0, 125], title="% of respondents", ticksuffix="%", tickvals=[0, 25, 50, 75, 100]), yaxis_title="", showlegend=False)
    return style_fig(fig, height)


def wilson_moe(p, n):
    return 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))


# ============================================================================
# Sidebar: brand, data, filters
# ============================================================================
with st.sidebar:
    st.markdown('<div class="brand">🛍️ RecoPulse</div><div class="brand-sub">RECOMMENDATION ANALYTICS</div>', unsafe_allow_html=True)
    st.divider()
    uploaded_file = st.file_uploader("Upload survey CSV", type=["csv"], help="Optional if Product_Recommendation_Cleaned.csv sits next to app.py")
    st.toggle("Big-screen mode (larger fonts)", key="big_mode", help="Handy when projecting to the class.")

raw_df = load_csv(uploaded_file)

hero_chips = ""
st.markdown(f"""
<div class="hero">
<div class="hero-tag">FOUNDATIONS OF DATA SCIENCE · ASSIGNMENT 3</div>
<h1>Reco<span class="accent">Pulse</span>: do product recommendations actually move shoppers?</h1>
<p>An interactive look at what customers buy, how they discover products, and how much they trust the recommendations they are shown.</p>
</div>
""", unsafe_allow_html=True)

if raw_df is None or raw_df.empty:
    st.info("Upload `Product_Recommendation_Cleaned.csv` in the sidebar, or place it beside `app.py` and refresh.")
    st.stop()

data, order_map, missing = prepare(raw_df)
if data is None:
    st.error(f"Could not find these required questions in the CSV: {', '.join(missing)}")
    st.write("Columns found:", list(raw_df.columns))
    st.stop()
if data.empty:
    st.error("No usable rows after cleaning.")
    st.stop()

if st.session_state.get("big_mode"):
    st.markdown("""<style>
    .kpi-value{font-size:2.7rem!important}.kpi-value.small{font-size:1.8rem!important}.kpi-label{font-size:.9rem!important}.kpi-sub{font-size:.95rem!important}
    .section-head{font-size:1.4rem!important}.section-sub{font-size:1rem!important}.insight{font-size:1.05rem!important}.step ul{font-size:1.05rem!important}
    </style>""", unsafe_allow_html=True)

FILTERS = [(k, LABELS[k]) for k in ["age", "gender", "category", "platform", "shop_freq", "rec_freq"] if k in data.columns]
FILTER_OPTS = {k: ordered(k, data[k].unique().tolist()) for k, _ in FILTERS}


def reset_filters():
    for k, _ in FILTERS:
        st.session_state[f"f_{k}"] = FILTER_OPTS[k]


with st.sidebar:
    st.markdown("### 🎛️ Global filters")
    st.caption("Every KPI and chart on the Overview, Drivers and Profile tabs responds to these.")
    for k, lab in FILTERS:
        if f"f_{k}" not in st.session_state:
            st.session_state[f"f_{k}"] = FILTER_OPTS[k]
        st.multiselect(lab, FILTER_OPTS[k], key=f"f_{k}")
    st.button("↺ Reset all filters", on_click=reset_filters, use_container_width=True)
    st.divider()
    st.caption("Foundations of Data Science • Assignment 3")

mask = pd.Series(True, index=data.index)
for k, _ in FILTERS:
    mask &= data[k].isin(st.session_state.get(f"f_{k}", FILTER_OPTS[k]))
filtered = data[mask].copy()
filter_on = len(filtered) != len(data)

if filtered.empty:
    st.warning("No respondents match these filters — pick at least one value in every filter (or press Reset).")
    st.stop()

# ============================================================================
# KPIs
# ============================================================================
n_resp, n_total = len(filtered), len(data)
sat_f, sat_all = pos_rate(filtered), pos_rate(data)
ad_f, ad_all = adopt_rate(filtered), adopt_rate(data)
cat_counts = filtered["category"].value_counts()
top_cat, top_cat_share = cat_counts.index[0], 100 * cat_counts.iloc[0] / n_resp

head("Executive snapshot", "Headline numbers for the respondents currently selected in the sidebar.")
k1, k2, k3, k4 = st.columns(4)
k1.markdown(kpi("Respondents", f"{n_resp:,}", f"{100*n_resp/n_total:.0f}% of {n_total} surveyed", INDIGO,
                "Number of survey responses in the current selection."), unsafe_allow_html=True)
k2.markdown(kpi("Positive satisfaction", f"{sat_f:.0f}%", delta_html(sat_f, sat_all, filter_on), TEAL,
                "Share answering Satisfied or Very Satisfied about the recommendations they see."), unsafe_allow_html=True)
k3.markdown(kpi("Recommendation-driven buying", f"{ad_f:.0f}%", delta_html(ad_f, ad_all, filter_on), AMBER,
                "Share who buy based on platform recommendations Always or Sometimes."), unsafe_allow_html=True)
k4.markdown(kpi("Leading category", top_cat, f"{top_cat_share:.0f}% of selected respondents", CORAL,
                "Most frequently bought product category in the current selection.", small=True), unsafe_allow_html=True)
st.write("")

# ============================================================================
# Tabs
# ============================================================================
tabs = st.tabs(["📈 Overview", "🧭 Drivers", "👤 Shopper profile", "🧪 What-if lab", "🤖 ML predictor", "🔎 Explorer", "🎤 Present"])
tab_over, tab_drv, tab_prof, tab_lab, tab_ml, tab_exp, tab_pres = tabs

# ---------------------------------------------------------------- Overview
with tab_over:
    # auto insights
    ins = []
    if "discovery" in filtered.columns:
        d = filtered["discovery"].value_counts()
        ins.append(("🔎", f"<b>{d.index[0]}</b> is the #1 discovery channel ({100*d.iloc[0]/n_resp:.0f}% of respondents)."))
    cs = filtered[filtered["satisfaction"] != NA_LABEL].groupby("category")["satisfaction"].agg(lambda s: 100 * s.isin(POS_SAT).mean())
    cn = filtered.groupby("category").size()
    cs = cs[cn[cs.index] >= 3]
    if len(cs) >= 2:
        ins.append(("😊", f"<b>{cs.idxmax()}</b> shoppers are happiest with recommendations ({cs.max():.0f}% positive); <b>{cs.idxmin()}</b> lowest ({cs.min():.0f}%)."))
    if "returned" in filtered.columns:
        r = 100 * filtered["returned"].str.startswith("Yes").mean()
        ins.append(("↩️", f"<b>{r:.0f}%</b> have returned an online purchase at least once."))
    if ins:
        cols = st.columns(len(ins))
        for c, (ico, txt) in zip(cols, ins):
            c.markdown(f'<div class="insight"><span class="ico">{ico}</span>{txt}</div>', unsafe_allow_html=True)
        st.write("")

    c1, c2 = st.columns([1.35, 1])
    with c1:
        head("Response timeline", "When survey answers arrived (survey activity — not sales).")
        ts = filtered.dropna(subset=["timestamp"])
        if ts.empty:
            st.info("No valid timestamps in the data.")
        else:
            span_days = (ts["timestamp"].max() - ts["timestamp"].min()).total_seconds() / 86400
            freq, unit = ("30min", "30 min") if span_days <= 1.5 else ("D", "day")
            per = ts.set_index("timestamp").resample(freq).size()
            cum = per.cumsum()
            fig = go.Figure()
            fig.add_bar(x=per.index, y=per.values, name=f"Responses per {unit}", marker_color="#c9cffd",
                        hovertemplate="%{x|%d %b, %H:%M}<br>%{y} responses<extra></extra>")
            fig.add_scatter(x=cum.index, y=cum.values, name="Cumulative responses", mode="lines+markers",
                            line=dict(color=INDIGO, width=3), marker=dict(size=7, color=INDIGO), yaxis="y2",
                            hovertemplate="%{x|%d %b, %H:%M}<br>%{y} so far<extra></extra>")
            fig.update_layout(yaxis=dict(title=f"Per {unit}", rangemode="tozero"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Cumulative", rangemode="tozero"),
                              xaxis_title="")
            show(style_fig(fig, 360))
    with c2:
        head("Category share", "What respondents buy most often online.")
        pie = cat_counts.rename_axis("Category").reset_index(name="Respondents")
        fig = px.pie(pie, names="Category", values="Respondents", hole=.6, color_discrete_sequence=CAT_COLORS)
        fig.update_traces(textposition="outside", textinfo="percent", marker=dict(line=dict(color="#fff", width=2)),
                          hovertemplate="<b>%{label}</b><br>%{value} respondents<br>%{percent}<extra></extra>")
        fig.update_layout(annotations=[dict(text=f"<b>{n_resp}</b><br><span style='font-size:11px;color:#5d6b88'>respondents</span>",
                                            x=.5, y=.5, font_size=24, showarrow=False)],
                          legend=dict(orientation="h", y=-.08))
        show(style_fig(fig, 360))

    c3, c4 = st.columns(2)
    with c3:
        head("How often do they buy via recommendations?", "Self-reported frequency of purchases driven by platform suggestions.")
        fc = filtered["rec_freq"].value_counts().reindex(ordered("rec_freq", filtered["rec_freq"].unique())).dropna().rename_axis("Frequency").reset_index(name="Respondents")
        fig = px.bar(fc, x="Frequency", y="Respondents", text="Respondents", color="Frequency",
                     color_discrete_map={k: FREQ_COLORS.get(k, SLATE) for k in fc["Frequency"]})
        fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False,
                          hovertemplate="<b>%{x}</b><br>%{y} respondents<extra></extra>")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Respondents")
        show(style_fig(fig, 350))
    with c4:
        head("Satisfaction by category", "Row = category; each cell is the share of that category's shoppers (n shown).")
        ct = pd.crosstab(filtered["category"], filtered["satisfaction"])
        cols_ord = ordered("satisfaction", ct.columns.tolist())
        ct = ct[cols_ord]
        ct = ct.loc[filtered["category"].value_counts().index[::-1]]
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        text = [[f"{pct.iloc[i, j]:.0f}%<br>n={ct.iloc[i, j]}" for j in range(ct.shape[1])] for i in range(ct.shape[0])]
        fig = go.Figure(go.Heatmap(z=pct.values, x=ct.columns, y=ct.index, text=text, texttemplate="%{text}",
                                   colorscale=SEQ, zmin=0, zmax=100, showscale=False, xgap=3, ygap=3,
                                   hovertemplate="<b>%{y}</b> · %{x}<br>%{z:.1f}% of category<extra></extra>"))
        fig.update_layout(xaxis_title="", yaxis_title="")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        show(style_fig(fig, 350))

# ---------------------------------------------------------------- Drivers
with tab_drv:
    st.markdown('<div class="note">Dashed red line = overall rate for the current selection. ⚠ marks groups with fewer than 5 respondents — treat those as anecdotes, not findings.</div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        if "discovery" in filtered.columns:
            head("Satisfaction by discovery channel", "% Satisfied / Very satisfied, by how people usually find products.")
            show(rate_bar(filtered, "discovery", "sat"))
    with b:
        head("Recommendation-driven buying by category", "% who buy via recommendations Always or Sometimes.")
        show(rate_bar(filtered, "category", "rec"))
    c, d = st.columns(2)
    with c:
        if "factor" in filtered.columns:
            head("Satisfaction by key buying factor", "Do shoppers who value price, brand, reviews or quality feel differently?")
            show(rate_bar(filtered, "factor", "sat"))
    with d:
        if "shop_freq" in filtered.columns:
            head("Recommendation-driven buying by shopping frequency", "Do heavy online shoppers lean on recommendations more?")
            show(rate_bar(filtered, "shop_freq", "rec"))

# ---------------------------------------------------------------- Shopper profile
with tab_prof:
    a, b = st.columns(2)
    with a:
        if "spending" in filtered.columns:
            head("Monthly online spend", "Self-reported spending bands (₹).")
            sp = filtered["spending"].value_counts().reindex(ordered("spending", filtered["spending"].unique())).dropna().rename_axis("Band").reset_index(name="Respondents")
            fig = px.bar(sp, x="Band", y="Respondents", text="Respondents", color_discrete_sequence=[VIOLET])
            fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
            fig.update_layout(xaxis_title="", yaxis_title="Respondents")
            show(style_fig(fig, 340))
    with b:
        if "platform" in filtered.columns:
            head("Platform share", "Where respondents shop most frequently.")
            pl = filtered["platform"].value_counts().rename_axis("Platform").reset_index(name="Respondents")
            fig = px.pie(pl, names="Platform", values="Respondents", hole=.55, color_discrete_sequence=CAT_COLORS)
            fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="#fff", width=2)))
            fig.update_layout(showlegend=False)
            show(style_fig(fig, 340))
    c, d = st.columns(2)
    with c:
        if "device" in filtered.columns:
            head("Device used", "Primary device for online shopping.")
            dv = filtered["device"].value_counts().sort_values().rename_axis("Device").reset_index(name="Respondents")
            fig = px.bar(dv, x="Respondents", y="Device", orientation="h", text="Respondents", color_discrete_sequence=[TEAL])
            fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
            fig.update_layout(xaxis_title="Respondents", yaxis_title="")
            show(style_fig(fig, 320))
    with d:
        if "payment" in filtered.columns:
            head("Payment preference", "How people like to pay.")
            pm = filtered["payment"].value_counts().sort_values().rename_axis("Method").reset_index(name="Respondents")
            fig = px.bar(pm, x="Respondents", y="Method", orientation="h", text="Respondents", color_discrete_sequence=[AMBER])
            fig.update_traces(textposition="outside", marker_line_width=0, cliponaxis=False)
            fig.update_layout(xaxis_title="Respondents", yaxis_title="")
            show(style_fig(fig, 320))

# ---------------------------------------------------------------- What-if lab
with tab_lab:
    st.markdown('<div class="note"><b>Demo lab.</b> Everything here is <b>simulated</b> and labelled as such — it shows what the KPIs <i>could</i> look like under different assumptions. It never overwrites your real survey data.</div>', unsafe_allow_html=True)
    st.write("")
    mode = st.radio("Choose a demo", ["🎚️ Scenario slider: improve the recommender", "🎲 Random respondent generator"], horizontal=True, label_visibility="collapsed")

    if mode.startswith("🎚️"):
        head("If the recommender got better…", "Move shoppers between answer levels and watch both KPIs respond. Uses the respondents currently selected in the sidebar.")
        base_sat = filtered[filtered["satisfaction"] != NA_LABEL]["satisfaction"].value_counts().astype(float)
        base_rec = filtered[filtered["rec_freq"] != NA_LABEL]["rec_freq"].value_counts().astype(float)
        s1, s2, s3 = st.columns(3)
        up = s1.slider("Neutral → Satisfied (%)", 0, 100, 30, 5, help="Share of 'Neutral' respondents who become 'Satisfied'.")
        lift = s2.slider("Rarely/Never → Sometimes (%)", 0, 100, 25, 5, help="Share of low-adoption respondents who start buying via recommendations sometimes.")
        base_n = s3.number_input("Scale to a customer base of", 100, 10_000_000, 10_000, 1000)

        new_sat, new_rec = base_sat.copy(), base_rec.copy()
        moved = base_sat.get("Neutral", 0) * up / 100
        if "Neutral" in new_sat.index:
            new_sat["Neutral"] -= moved
        new_sat["Satisfied"] = new_sat.get("Satisfied", 0) + moved
        low = [x for x in ("Rarely", "Never") if x in new_rec.index]
        low_total = sum(base_rec[x] for x in low)
        for x in low:
            new_rec[x] -= base_rec[x] * lift / 100
        new_rec["Sometimes"] = new_rec.get("Sometimes", 0) + low_total * lift / 100

        pos_new = 100 * sum(new_sat.get(x, 0) for x in POS_SAT) / new_sat.sum()
        ad_new = 100 * sum(new_rec.get(x, 0) for x in ADOPT) / new_rec.sum()
        m1, m2, m3 = st.columns(3)
        m1.markdown(kpi("Positive satisfaction", f"{pos_new:.0f}%", f'<span class="up">▲ {pos_new - sat_f:.1f} pts</span> from {sat_f:.0f}% today', TEAL, "Simulated"), unsafe_allow_html=True)
        m2.markdown(kpi("Recommendation-driven buying", f"{ad_new:.0f}%", f'<span class="up">▲ {ad_new - ad_f:.1f} pts</span> from {ad_f:.0f}% today', AMBER, "Simulated"), unsafe_allow_html=True)
        m3.markdown(kpi("Extra rec-driven shoppers", f"+{(ad_new - ad_f) / 100 * base_n:,.0f}", f"per {base_n:,} customers <span class='simbadge'>SIMULATED</span>", INDIGO, "Simulated"), unsafe_allow_html=True)
        st.write("")

        g1, g2 = st.columns(2)
        for col, (title, before, after, ordr, cmap) in zip(
            (g1, g2),
            [("Satisfaction mix: today vs scenario", base_sat, new_sat, ordered("satisfaction", set(base_sat.index) | set(new_sat.index)), SAT_COLORS),
             ("Buying via recommendations: today vs scenario", base_rec, new_rec, ordered("rec_freq", set(base_rec.index) | set(new_rec.index)), FREQ_COLORS)],
        ):
            with col:
                head(title)
                fig = go.Figure()
                fig.add_bar(name="Today (real)", x=ordr, y=[100 * before.get(o, 0) / before.sum() for o in ordr], marker_color="#c9cffd",
                            hovertemplate="%{x}: %{y:.1f}%<extra>Today</extra>")
                fig.add_bar(name="Scenario (simulated)", x=ordr, y=[100 * after.get(o, 0) / after.sum() for o in ordr], marker_color=INDIGO,
                            hovertemplate="%{x}: %{y:.1f}%<extra>Scenario</extra>")
                fig.update_layout(barmode="group", yaxis=dict(title="% of respondents", ticksuffix="%"), xaxis_title="")
                show(style_fig(fig, 340))
    else:
        head("Generate random respondents", "Create a synthetic cohort and see how the KPIs would read if the survey looked like this. Great for showing 'what if we had more / different data?'.")
        g1, g2, g3 = st.columns([1, 1, 1.4])
        n_new = g1.slider("How many simulated respondents?", 10, 500, 100, 10)
        seed = g2.number_input("Random seed", 0, 9999, 42, help="Same seed = same synthetic data. Change it for a fresh draw.")
        gen_mode = g3.radio("Distribution", ["Mirror the real survey", "Custom mix"], horizontal=True)

        src_cols = [c for c in data.columns if c not in ("timestamp", "reg")]
        overrides = {}
        if gen_mode == "Custom mix":
            st.caption("Set relative weights (they are normalised). Everything else still mirrors the real survey.")
            ec1, ec2, ec3 = st.columns(3)
            for ecol, key, levels in [
                (ec1, "satisfaction", ordered("satisfaction", [x for x in data["satisfaction"].unique() if x != NA_LABEL])),
                (ec2, "rec_freq", ordered("rec_freq", [x for x in data["rec_freq"].unique() if x != NA_LABEL])),
                (ec3, "category", ordered("category", data["category"].unique())),
            ]:
                with ecol:
                    st.markdown(f"**{LABELS[key]}**")
                    obs = data[key].value_counts(normalize=True)
                    w = [st.slider(lv, 0, 100, int(round(100 * obs.get(lv, 0))), key=f"w_{key}_{lv}") for lv in levels]
                    overrides[key] = (levels, w)

        b1, b2 = st.columns([1, 5])
        gen = b1.button("▶ Generate", type="primary", key="gen_btn")
        if b2.button("Clear simulated data", key="clr_btn"):
            st.session_state.pop("sim_df", None)

        if gen:
            rng = np.random.default_rng(int(seed))
            out = {}
            for c in src_cols:
                if c in overrides:
                    levels, w = overrides[c]
                    p = np.array(w, float)
                    p = p / p.sum() if p.sum() > 0 else np.ones(len(levels)) / len(levels)
                    out[c] = rng.choice(levels, size=n_new, p=p)
                else:
                    vc = filtered[c].value_counts(normalize=True)
                    out[c] = rng.choice(vc.index.to_numpy(dtype=object), size=n_new, p=vc.values)
            sim = pd.DataFrame(out)
            sim.insert(0, "reg", [f"SIM-{i+1:04d}" for i in range(n_new)])
            sim["timestamp"] = pd.NaT
            st.session_state["sim_df"] = sim

        sim = st.session_state.get("sim_df")
        if sim is not None:
            sp_, sa_ = pos_rate(sim), adopt_rate(sim)
            m1, m2, m3 = st.columns(3)
            m1.markdown(kpi("Simulated respondents", f"{len(sim):,}", "<span class='simbadge'>SIMULATED</span>", VIOLET, "Synthetic cohort"), unsafe_allow_html=True)
            m2.markdown(kpi("Positive satisfaction", f"{sp_:.0f}%", f"real survey: {sat_f:.0f}%", TEAL, "Simulated cohort"), unsafe_allow_html=True)
            m3.markdown(kpi("Recommendation-driven buying", f"{sa_:.0f}%", f"real survey: {ad_f:.0f}%", AMBER, "Simulated cohort"), unsafe_allow_html=True)
            st.write("")
            v1, v2 = st.columns(2)
            for col, key, ttl in [(v1, "satisfaction", "Satisfaction: real vs simulated"), (v2, "category", "Category mix: real vs simulated")]:
                with col:
                    head(ttl)
                    lv = ordered(key, [x for x in set(filtered[key]) | set(sim[key]) if x != NA_LABEL])
                    rr = filtered[key].value_counts(normalize=True) * 100
                    ss = sim[key].value_counts(normalize=True) * 100
                    fig = go.Figure()
                    fig.add_bar(name="Real survey", x=lv, y=[rr.get(x, 0) for x in lv], marker_color="#c9cffd")
                    fig.add_bar(name="Simulated", x=lv, y=[ss.get(x, 0) for x in lv], marker_color=VIOLET)
                    fig.update_layout(barmode="group", yaxis=dict(title="% of respondents", ticksuffix="%"), xaxis_title="")
                    fig.update_xaxes(tickangle=-12)
                    show(style_fig(fig, 330))
            with st.expander("Peek at the synthetic rows"):
                st.markdown('<span class="simbadge">SIMULATED — not real people</span>', unsafe_allow_html=True)
                show_table(sim.drop(columns=["timestamp"]).rename(columns=LABELS), 280)
            st.markdown('<div class="warn"><b>Caveat:</b> each column is sampled independently, so real-world relationships between answers (e.g. category ↔ satisfaction) are lost. That is fine for demoing KPI sensitivity, not for drawing conclusions.</div>', unsafe_allow_html=True)

        with st.expander("📐 Why sample size matters (margin of error)"):
            p0 = sat_f / 100
            ns = np.arange(10, 501, 5)
            fig = go.Figure()
            fig.add_scatter(x=ns, y=[100 * wilson_moe(p0, n) for n in ns], mode="lines", line=dict(color=INDIGO, width=3), name="95% margin of error",
                            hovertemplate="n=%{x}<br>±%{y:.1f} pts<extra></extra>")
            fig.add_scatter(x=[len(filtered)], y=[100 * wilson_moe(p0, len(filtered))], mode="markers+text", marker=dict(size=13, color=CORAL),
                            text=[f"Your survey: ±{100 * wilson_moe(p0, len(filtered)):.0f} pts"], textposition="top right", name="Your survey")
            if sim is not None:
                fig.add_scatter(x=[len(sim)], y=[100 * wilson_moe(p0, len(sim))], mode="markers+text", marker=dict(size=13, color=VIOLET),
                                text=[f"Simulated n={len(sim)}: ±{100 * wilson_moe(p0, len(sim)):.0f} pts"], textposition="top right", name="Simulated")
            fig.update_layout(xaxis_title="Sample size (n)", yaxis=dict(title="± percentage points", ticksuffix=" pts"))
            show(style_fig(fig, 320))
            st.caption(f"With the current positive-satisfaction rate of {sat_f:.0f}%, a survey of {len(filtered)} people is accurate only to roughly ±{100 * wilson_moe(p0, len(filtered)):.0f} points — a key limitation to mention in your talk.")

# ---------------------------------------------------------------- ML predictor
with tab_ml:
    st.markdown('<div class="note"><b>Mini machine-learning demo.</b> Train a small classifier on the survey, check honestly whether it beats guessing, then predict for a made-up shopper. Uses the <b>full</b> dataset (not the sidebar filters).</div>', unsafe_allow_html=True)
    st.write("")
    t1, t2, t3 = st.columns([1.3, 1, 1])
    target_name = t1.selectbox("What should the model predict?", ["Satisfied with recommendations?", "Buys via recommendations (Always/Sometimes)?"])
    model_name = t2.radio("Model", ["Logistic Regression", "Random Forest"], horizontal=True)
    if target_name.startswith("Satisfied"):
        valid = data[data["satisfaction"] != NA_LABEL]
        y = valid["satisfaction"].isin(POS_SAT).astype(int)
        banned, defaults = {"satisfaction"}, ["category", "discovery", "factor", "shop_freq", "platform", "spending"]
    else:
        valid = data[data["rec_freq"] != NA_LABEL]
        y = valid["rec_freq"].isin(ADOPT).astype(int)
        banned, defaults = {"rec_freq"}, ["category", "discovery", "factor", "shop_freq", "satisfaction", "spending"]
    feat_pool = [k for k in LABELS if k in data.columns and k not in banned and k not in ("reg", "timestamp")]
    feats = st.multiselect("Features the model may use", feat_pool, default=[d for d in defaults if d in feat_pool],
                           format_func=lambda k: LABELS[k], help="Fewer features = less overfitting on a small survey.")
    t3.metric("Training rows", len(valid))

    if len(feats) < 1:
        st.info("Pick at least one feature.")
    elif y.nunique() < 2 or y.value_counts().min() < 2:
        st.warning("The target has only one class (or too few examples of one) — can't train a classifier on this selection.")
    else:
        X = valid[feats].astype(str)

        @st.cache_data(show_spinner="Training…")
        def train_eval(X, y, model_name):
            pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), list(X.columns))])
            est = (LogisticRegression(C=0.5, max_iter=2000) if model_name == "Logistic Regression"
                   else RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=2, random_state=7))
            pipe = Pipeline([("pre", pre), ("m", est)])
            n_splits = int(min(5, y.value_counts().min()))
            cv = RepeatedStratifiedKFold(n_splits=max(n_splits, 2), n_repeats=5, random_state=7)
            acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            pipe.fit(X, y)
            names = pipe.named_steps["pre"].get_feature_names_out()
            raw = (np.abs(pipe.named_steps["m"].coef_[0]) if model_name == "Logistic Regression"
                   else pipe.named_steps["m"].feature_importances_)
            imp = {f: float(sum(v for n, v in zip(names, raw) if n.startswith(f"oh__{f}_"))) for f in X.columns}
            imp = pd.Series(imp)
            imp = 100 * imp / imp.sum() if imp.sum() > 0 else imp
            return float(acc.mean()), float(acc.std()), float(max(y.mean(), 1 - y.mean())), pipe, imp.sort_values()

        acc, acc_sd, base, pipe, imp = train_eval(X, y, model_name)
        mm1, mm2, mm3 = st.columns(3)
        mm1.markdown(kpi("Cross-validated accuracy", f"{acc*100:.0f}%", f"± {acc_sd*100:.0f} across folds", INDIGO, "Repeated stratified k-fold on the survey"), unsafe_allow_html=True)
        mm2.markdown(kpi("'Always guess majority' baseline", f"{base*100:.0f}%", "the bar a model must beat", SLATE, "Accuracy of predicting the most common answer for everyone"), unsafe_allow_html=True)
        lift = (acc - base) * 100
        real_gain = lift >= max(5.0, acc_sd * 100)  # must clear both 5 pts and the fold-to-fold noise
        mm3.markdown(kpi("Lift over baseline", f"{lift:+.0f} pts", "clearly better than guessing" if real_gain else "within noise — not clearly better than guessing",
                         TEAL if real_gain else CORAL, "Accuracy minus baseline"), unsafe_allow_html=True)
        st.write("")
        if not real_gain:
            st.markdown('<div class="warn"><b>Honest finding:</b> with so few responses the model is not reliably better than always guessing the majority answer (the gain is smaller than the fold-to-fold noise). That is a legitimate data-science result — it tells you more data is needed before trusting any prediction.</div>', unsafe_allow_html=True)
            st.write("")

        p1, p2 = st.columns([1.15, 1])
        with p1:
            head("Try a what-if shopper", "Choose a profile and see the predicted probability.")
            row = {}
            fc_ = st.columns(2)
            for i, f in enumerate(feats):
                opts = ordered(f, X[f].unique().tolist())
                default_idx = opts.index(X[f].mode().iloc[0]) if X[f].mode().iloc[0] in opts else 0
                row[f] = fc_[i % 2].selectbox(LABELS[f], opts, index=default_idx, key=f"ml_{f}")
            proba = float(pipe.predict_proba(pd.DataFrame([row]))[0][list(pipe.classes_).index(1)])
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=100 * proba, number=dict(suffix="%", font=dict(size=42, family="Manrope")),
                title=dict(text="Predicted probability of 'Yes'", font=dict(size=14)),
                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=INDIGO), bgcolor="#eef0ff",
                           steps=[dict(range=[0, 40], color="#ffe3e3"), dict(range=[40, 60], color="#fff3d6"), dict(range=[60, 100], color="#d9f7f2")],
                           threshold=dict(line=dict(color=NAVY, width=3), value=50)),
            ))
            show(style_fig(fig, 270))
            st.caption(f"Verdict: **{'Yes' if proba >= .5 else 'No'}** for “{target_name}” — treat as an illustration, not a forecast.")
        with p2:
            head("What the model leans on", "Relative importance of each question (sums to 100%).")
            fig = go.Figure(go.Bar(x=imp.values, y=[LABELS[i] for i in imp.index], orientation="h", marker_color=VIOLET,
                                   text=[f"{v:.0f}%" for v in imp.values], textposition="outside", cliponaxis=False))
            fig.update_layout(xaxis=dict(title="Importance (%)", range=[0, max(imp.max() * 1.25, 10)]), yaxis_title="", showlegend=False)
            show(style_fig(fig, 340))

        sim = st.session_state.get("sim_df")
        if sim is not None and all(f in sim.columns for f in feats):
            pr = pipe.predict_proba(sim[feats].astype(str))[:, list(pipe.classes_).index(1)]
            st.markdown(f'<div class="insight"><span class="ico">🎲</span>Scoring your <b>{len(sim)} simulated respondents</b> from the What-if lab: the model predicts <b>{100 * (pr >= .5).mean():.0f}%</b> would answer “Yes”. <span class="simbadge">SIMULATED</span></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Explorer
with tab_exp:
    head("Explore any question", "Pick a column and a chart; optionally cross it with a second column.")
    use_all = st.checkbox("Use all respondents (ignore sidebar filters)", value=True)
    ex = data if use_all else filtered
    cols_ex = [k for k in LABELS if k in ex.columns and k not in ("reg", "timestamp")]
    e1, e2, e3 = st.columns(3)
    c_main = e1.selectbox("Column", cols_ex, index=cols_ex.index("category") if "category" in cols_ex else 0, format_func=lambda k: LABELS[k])
    chart = e2.selectbox("Chart", ["Bar", "Donut", "Treemap", "Cross-tab heatmap"])
    c_cross = e3.selectbox("Cross with", [k for k in cols_ex if k != c_main], format_func=lambda k: LABELS[k]) if chart == "Cross-tab heatmap" else None
    if chart == "Cross-tab heatmap":
        ct = pd.crosstab(ex[c_main], ex[c_cross])
        ct = ct.loc[ordered(c_main, ct.index.tolist()), ordered(c_cross, ct.columns.tolist())]
        fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale=SEQ, labels=dict(x=LABELS[c_cross], y=LABELS[c_main], color="Respondents"))
        fig.update_xaxes(showgrid=False, tickangle=-15)
        fig.update_yaxes(showgrid=False)
        show(style_fig(fig, 460))
    else:
        vc = ex[c_main].value_counts()
        vc = vc.reindex(ordered(c_main, vc.index.tolist())) if c_main in ORDINAL else vc
        cdf = vc.rename_axis("Value").reset_index(name="Respondents")
        if chart == "Bar":
            fig = px.bar(cdf, x="Respondents", y="Value", orientation="h", text="Respondents", color="Respondents", color_continuous_scale=["#c9cffd", INDIGO])
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"), yaxis_title="")
        elif chart == "Donut":
            fig = px.pie(cdf, names="Value", values="Respondents", hole=.55, color_discrete_sequence=CAT_COLORS)
            fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="#fff", width=2)))
            fig.update_layout(showlegend=False)
        else:
            fig = px.treemap(cdf, path=["Value"], values="Respondents", color="Respondents", color_continuous_scale=SEQ)
            fig.update_layout(coloraxis_showscale=False)
        show(style_fig(fig, 440))
    with st.expander("View respondent table (emails removed)"):
        tbl = ex.drop(columns=["reg"]).rename(columns=LABELS)
        show_table(tbl, 380)
        st.download_button("⬇ Download as CSV", tbl.to_csv(index=False).encode("utf-8"), "recopulse_view.csv", "text/csv")

# ---------------------------------------------------------------- Present
with tab_pres:
    full = data
    pos_a, ad_a = pos_rate(full), adopt_rate(full)
    neutral_share = 100 * (full["satisfaction"] == "Neutral").mean()
    cat_stats = (full[full["satisfaction"] != NA_LABEL].groupby("category")["satisfaction"].agg(lambda s: 100 * s.isin(POS_SAT).mean()).to_frame("pos")
                 .join(full.groupby("category").size().rename("n")))
    cat_stats = cat_stats[cat_stats["n"] >= 5]
    best_cat = cat_stats["pos"].idxmax() if len(cat_stats) else "—"
    worst_cat = cat_stats["pos"].idxmin() if len(cat_stats) else "—"
    disc_top, disc_share, plat_share = "—", 0.0, 0.0
    if "discovery" in full.columns:
        dvc = full["discovery"].value_counts(normalize=True) * 100
        disc_top, disc_share = dvc.index[0], dvc.iloc[0]
        plat_share = float(sum(v for k, v in dvc.items() if k.lower().startswith("platform")))
    anom = ""
    if "discovery" in full.columns:
        dsat = (full[full["satisfaction"] != NA_LABEL].groupby("discovery")["satisfaction"].agg(lambda s_: 100 * s_.isin(POS_SAT).mean()).to_frame("pos")
                .join(full.groupby("discovery").size().rename("n")))
        dsat = dsat[dsat["n"] >= 5]
        if len(dsat):
            lo = dsat["pos"].idxmin()
            anom = f" Anomaly to call out: <b>{lo}</b> shoppers are only {dsat['pos'].min():.0f}% positive vs {pos_a:.0f}% overall."
    fac_top, fac_share = ("—", 0.0)
    if "factor" in full.columns:
        fvc = full["factor"].value_counts(normalize=True) * 100
        fac_top, fac_share = fvc.index[0], fvc.iloc[0]
    stu = 100 * (full["occupation"] == "Student").mean() if "occupation" in full.columns else float("nan")
    young = 100 * full["age"].str.startswith("18").mean() if "age" in full.columns else float("nan")
    tsv = full["timestamp"].dropna()
    span = f"{(tsv.max() - tsv.min()).total_seconds() / 3600:.0f} hours" if len(tsv) > 1 else "a short window"

    st.markdown('<div class="note">Your 5-minute plan, pre-filled with live numbers from the full dataset. Re-check the Assignment 1 baseline numbers against these before presenting.</div>', unsafe_allow_html=True)
    st.write("")
    s1, s2 = st.columns(2)
    s1.markdown(f"""<div class="step"><span class="time">0:00 – 1:00</span><h4>Core problem & baseline</h4><ul>
<li><b>Question:</b> do platform recommendations actually influence what shoppers buy, and do they trust them?</li>
<li><b>Baseline:</b> {ad_a:.0f}% buy via recommendations Always/Sometimes; {pos_a:.0f}% are satisfied; <b>{neutral_share:.0f}%</b> are only neutral.</li>
<li>Survey of <b>{len(full)}</b> online shoppers; {disc_top} is the top discovery channel ({disc_share:.0f}%) vs only {plat_share:.0f}% via platform recommendations.</li></ul></div>""", unsafe_allow_html=True)
    s2.markdown(f"""<div class="step"><span class="time">1:00 – 3:00</span><h4>Live demo: reveal a pattern</h4><ul>
<li><b>Click 1:</b> sidebar → Top product category → keep only <b>{best_cat}</b>, watch the KPI deltas turn green. Then switch to <b>{worst_cat}</b>.</li>
<li><b>Click 2:</b> Drivers tab → satisfaction by discovery channel; point at the dashed overall line.{anom}</li>
<li><b>Click 3:</b> What-if lab → drag “Neutral → Satisfied” to show the upside of better recommendations.</li>
<li><b>Click 4 (bonus):</b> ML predictor → show the model vs baseline honestly.</li></ul></div>""", unsafe_allow_html=True)
    st.write("")
    s3, s4 = st.columns(2)
    s3.markdown(f"""<div class="step"><span class="time">3:00 – 4:00</span><h4>Two actionable recommendations</h4><ul>
<li><b>1 · Fix relevance where trust is lowest.</b> {worst_cat} has only {cat_stats['pos'].get(worst_cat, float('nan')):.0f}% positive satisfaction vs {pos_a:.0f}% overall — tune the recommender and add “why this?” explanations there first.</li>
<li><b>2 · Lean into social proof.</b> “{fac_top}” is the top buying factor ({fac_share:.0f}%) and {disc_top} is the top discovery route ({disc_share:.0f}%) versus {plat_share:.0f}% for platform suggestions — surface ratings, reviews and “friends bought this” on recommendation tiles.</li></ul></div>""", unsafe_allow_html=True)
    s4.markdown(f"""<div class="step"><span class="time">4:00 – 5:00</span><h4>Key limitation + Q&A</h4><ul>
<li><b>Small, narrow sample:</b> n = {len(full)}; ≈{stu:.0f}% students and ≈{young:.0f}% aged 18–24, collected over {span} — not representative of all Indian shoppers.</li>
<li>Self-reported answers; satisfaction is a 3-level scale; spending is in bands, not exact ₹.</li>
<li>Patterns are <b>correlational</b>; the what-if lab and ML demo are illustrations, not causal estimates.</li></ul></div>""", unsafe_allow_html=True)
    st.write("")
    with st.expander("🙋 Likely questions from the class (with answers)"):
        st.markdown(f"""
**Why did you count “Always + Sometimes” as recommendation-driven?** The survey has no “Often” option, so “Sometimes” is the middle level of actually acting on recommendations; “Always” alone would be only a handful of people.

**Can you trust the ML model?** With {len(full)} responses it barely beats the majority-answer baseline — which is exactly why it is a demo, and why more data is the first next step.

**Is this causal?** No — we see associations in self-reported data. Proving that recommendations *cause* purchases would need an experiment (A/B test).

**What would you do with more data?** Collect a larger and more varied sample, add real click/purchase logs, and test the two recommendations with an A/B experiment.
""")

st.divider()
st.caption("RecoPulse • Foundations of Data Science, Assignment 3 • Small self-reported survey — read patterns as directional, not conclusive.")
