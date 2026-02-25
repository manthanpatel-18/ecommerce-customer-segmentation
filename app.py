# FIXED VERSION v4 - All Plotly update_layout conflicts resolved
import joblib
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG — must be first Streamlit call
# ============================================================================
st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD ML MODEL
# ============================================================================
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('models/kmeans_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        with open('models/model_summary.json', 'r') as f:
            model_info = json.load(f)
        return model, scaler, model_info
    except FileNotFoundError:
        return None, None, None

kmeans_model, scaler, model_info = load_ml_model()

# ============================================================================
# THEME & CSS
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:          #080b12;
    --bg-card:     #0e1320;
    --bg-card2:    #111827;
    --bg-input:    #151d2e;
    --cyan:        #06b6d4;
    --cyan-dim:    rgba(6,182,212,0.10);
    --amber:       #f59e0b;
    --amber-dim:   rgba(245,158,11,0.10);
    --violet:      #8b5cf6;
    --green:       #10b981;
    --red:         #ef4444;
    --blue:        #3b82f6;
    --pink:        #ec4899;
    --slate:       #475569;
    --txt:         #e2e8f0;
    --txt2:        #94a3b8;
    --txt3:        #475569;
    --border:      rgba(148,163,184,0.08);
    --border2:     rgba(6,182,212,0.20);
    --glow:        rgba(6,182,212,0.15);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"],
.main, .main .block-container {
    background: var(--bg) !important;
    color: var(--txt) !important;
    font-family: 'DM Sans', system-ui, sans-serif !important;
}

.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1480px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #080b12 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* ── Sidebar text ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: var(--txt2) !important; font-size: 0.78rem !important; }

/* ── Metric override ── */
[data-testid="stMetric"] { background: transparent !important; }
[data-testid="stMetricLabel"] p { color: var(--txt3) !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: var(--txt) !important; font-family: 'Space Mono', monospace !important; font-size: 1.4rem !important; letter-spacing: -0.5px; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--txt2) !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--cyan-dim) !important;
    color: var(--cyan) !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
.stDataFrame iframe { background: var(--bg-card) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 99px; }

/* ── Hide white top header bar ── */
[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stHeader"]::before {
    background: var(--bg) !important;
}
/* Deploy button area */
[data-testid="stToolbar"] {
    background: var(--bg) !important;
}
[data-testid="stDecoration"] {
    background: var(--bg) !important;
    display: none !important;
}
/* Top rainbow line */
[data-testid="stDecorationColoredLine"] {
    display: none !important;
}
/* Main menu and other header elements */
#MainMenu, footer { visibility: hidden !important; }
header { background: rgba(8,11,18,0.95) !important; backdrop-filter: blur(8px); }

/* ── Components ── */
.hero {
    background: linear-gradient(135deg, #0a0f1e 0%, #0e1320 60%, #080d18 100%);
    border: 1px solid var(--border2);
    border-radius: 20px;
    padding: 2.8rem 2.5rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(6,182,212,0.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--cyan-dim);
    border: 1px solid var(--border2);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 0.7rem; font-weight: 600;
    color: var(--cyan); letter-spacing: 1px; text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    color: var(--txt) !important;
    line-height: 1.1 !important;
    letter-spacing: -1px;
}
.hero h1 .accent { color: var(--cyan); }
.hero-sub { color: var(--txt3) !important; font-size: 0.88rem !important; margin-top: 0.6rem; }

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.6rem;
}
.kpi {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi:hover { border-color: var(--border2); }
.kpi-accent { position: absolute; top: 0; left: 0; right: 0; height: 2px; }
.kpi-label {
    font-size: 0.65rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: var(--txt3); margin-bottom: 0.4rem;
}
.kpi-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.55rem; font-weight: 700; color: var(--txt);
    line-height: 1; letter-spacing: -0.5px;
}
.kpi-sub { font-size: 0.7rem; color: var(--txt3); margin-top: 0.3rem; }

.section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1.2rem; margin-top: 0.5rem;
}
.section-header h2 {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important; font-weight: 700 !important;
    color: var(--txt) !important; margin: 0 !important;
}
.section-icon {
    width: 34px; height: 34px;
    background: var(--cyan-dim);
    border: 1px solid var(--border2);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}

.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1.2rem;
}

.badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 99px;
    font-size: 0.68rem; font-weight: 600;
}
.badge-cyan  { background: var(--cyan-dim);  color: var(--cyan);  border: 1px solid var(--border2); }
.badge-amber { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }

.metric-mini {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
}
.metric-mini .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem; font-weight: 800; color: var(--txt);
}
.metric-mini .lbl { font-size: 0.64rem; color: var(--txt3); text-transform: uppercase; letter-spacing: 1px; }

.sidebar-logo {
    padding: 1.4rem 1rem 1.2rem;
    margin-bottom: 0.5rem;
    text-align: center;
    background: linear-gradient(180deg, rgba(6,182,212,0.05) 0%, transparent 100%);
    border-bottom: 1px solid var(--border2);
    position: relative;
}
.sidebar-logo::after {
    content: '';
    position: absolute;
    bottom: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.sidebar-logo .icon-wrap {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(6,182,212,0.05));
    border: 1px solid var(--border2);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.7rem;
    font-size: 1.6rem;
    box-shadow: 0 0 20px rgba(6,182,212,0.12);
}
.sidebar-logo .name {
    font-family: 'Playfair Display', serif;
    font-size: 1rem; font-weight: 800;
    background: linear-gradient(90deg, #e2e8f0 0%, #06b6d4 60%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block; margin-bottom: 0.45rem;
    letter-spacing: -0.3px;
    line-height: 1.25;
}
.sidebar-logo .sub {
    display: inline-block;
    background: var(--cyan-dim);
    border: 1px solid var(--border2);
    border-radius: 99px;
    padding: 3px 10px;
    font-size: 0.58rem; color: var(--cyan);
    text-transform: uppercase; letter-spacing: 1.8px;
    font-weight: 600;
}
.sidebar-section {
    font-size: 0.6rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1.8px;
    color: var(--cyan) !important;
    padding: 0.8rem 1rem 0.3rem;
    border-top: 1px solid var(--border);
    margin-top: 0.5rem;
}

/* ── Warning / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border-left: 3px solid var(--amber) !important;
    background: var(--amber-dim) !important;
}

/* ── Multiselect tags ── */
[data-baseweb="tag"] {
    background: var(--cyan-dim) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
}
[data-baseweb="tag"] span { color: var(--cyan) !important; font-size: 0.72rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================
DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0e1320",
    font=dict(family="DM Sans, system-ui, sans-serif", color="#94a3b8", size=11),
)
MARGIN = dict(l=50, r=30, t=30, b=50)  # default margin — use explicitly

# Default axis style (use explicitly, never spread inside DARK)
AXIS = dict(gridcolor="rgba(148,163,184,0.06)", zerolinecolor="rgba(148,163,184,0.08)")

SEG_COLORS = {
    "Champions":           "#06b6d4",
    "Cannot Lose Them":    "#f59e0b",
    "Loyal Customers":     "#8b5cf6",
    "Potential Loyalists": "#10b981",
    "At Risk":             "#ef4444",
    "Need Attention":      "#3b82f6",
    "New Customers":       "#ec4899",
    "Hibernating":         "#475569",
}

CLUSTER_PALETTE = [
    "#06b6d4","#f59e0b","#10b981","#8b5cf6",
    "#ef4444","#3b82f6","#ec4899","#f97316"
]

# ============================================================================
# DATA
# ============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('data/ecommerce_transactions.csv')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], dayfirst=True, format='mixed')
    return df

@st.cache_data
def calculate_rfm(df):
    analysis_date = df['transaction_date'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_id').agg(
        recency=('transaction_date', lambda x: (analysis_date - x.max()).days),
        frequency=('transaction_id', 'count'),
        monetary=('amount', 'sum')
    ).reset_index()

    rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')

    def segment(row):
        r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
        if   r >= 4 and f >= 4 and m >= 4: return 'Champions'
        elif r <= 2 and m >= 4:            return 'Cannot Lose Them'
        elif f >= 4:                        return 'Loyal Customers'
        elif r >= 4 and f >= 2:            return 'Potential Loyalists'
        elif r >= 4 and f == 1:            return 'New Customers'
        elif f >= 3 and r <= 2:            return 'At Risk'
        elif r <= 2:                        return 'Hibernating'
        else:                               return 'Need Attention'

    rfm['segment'] = rfm.apply(segment, axis=1)

    cd = df.groupby('customer_id').agg(
        first_purchase=('transaction_date', 'min'),
        last_purchase=('transaction_date', 'max')
    ).reset_index()
    cd['lifespan_days'] = (cd['last_purchase'] - cd['first_purchase']).dt.days
    rfm = rfm.merge(cd, on='customer_id')
    rfm['avg_purchase_value'] = rfm['monetary'] / rfm['frequency']
    rfm['lifespan_years']     = rfm['lifespan_days'].clip(lower=30) / 365.25
    rfm['purchases_per_year'] = rfm['frequency'] / rfm['lifespan_years']
    rfm['estimated_clv']      = (rfm['avg_purchase_value'] * rfm['purchases_per_year'] * 3).round(2)
    return rfm

df  = load_data()
rfm = calculate_rfm(df)

# Apply ML clusters
if kmeans_model is not None and scaler is not None:
    features = rfm[['recency', 'frequency', 'monetary']].values
    rfm['ml_cluster'] = kmeans_model.predict(scaler.transform(features))

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("""
<div class="sidebar-logo">
  <div class="icon-wrap">📊</div>
  <span class="name">Customer<br>Segmentation</span>
  <div class="sub">Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)

if model_info:
    st.sidebar.markdown('<div class="sidebar-section">🤖 ML Model</div>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Algorithm", "K-Means")
    c2.metric("Clusters", model_info['n_clusters'])
    st.sidebar.metric("Silhouette Score", f"{model_info['metrics']['silhouette_score']:.3f}")
else:
    st.sidebar.info("Run ml_clustering.ipynb to load ML model")

st.sidebar.markdown('<div class="sidebar-section">📅 Time Period</div>', unsafe_allow_html=True)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df['transaction_date'].min().date(), df['transaction_date'].max().date()),
    min_value=df['transaction_date'].min().date(),
    max_value=df['transaction_date'].max().date(),
    label_visibility="hidden"
)

st.sidebar.markdown('<div class="sidebar-section">👥 Segments</div>', unsafe_allow_html=True)
all_segs = sorted(rfm['segment'].unique())
seg_counts = rfm['segment'].value_counts().to_dict()
seg_opts = [f"{s} ({seg_counts.get(s,0):,})" for s in all_segs]
seg_map  = {f"{s} ({seg_counts.get(s,0):,})": s for s in all_segs}

sel_display = st.sidebar.multiselect("Segments", options=seg_opts, default=seg_opts, label_visibility="hidden")
selected_segments = [seg_map[s] for s in sel_display]

st.sidebar.markdown('<div class="sidebar-section">💰 Revenue Filter</div>', unsafe_allow_html=True)
min_revenue = st.sidebar.slider(
    "Min Revenue", min_value=0, max_value=int(rfm['monetary'].max()),
    value=0, step=100, format="$%d", label_visibility="hidden"
)

# ============================================================================
# FILTER DATA
# ============================================================================
if len(date_range) == 2:
    df_f = df[(df['transaction_date'].dt.date >= date_range[0]) &
              (df['transaction_date'].dt.date <= date_range[1])]
else:
    df_f = df

rfm_f = rfm[rfm['segment'].isin(selected_segments) & (rfm['monetary'] >= min_revenue)]

total_rev  = rfm_f['monetary'].sum()
total_cust = len(rfm_f)
avg_order  = df_f['amount'].mean() if len(df_f) > 0 else 0
avg_freq   = rfm_f['frequency'].mean() if total_cust > 0 else 0
avg_clv    = rfm_f['estimated_clv'].mean() if total_cust > 0 else 0

# ============================================================================
# HERO
# ============================================================================
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">✦ Live Analytics</div>
  <h1>E-Commerce Customer<br><span class="accent">Segmentation</span> Dashboard</h1>
  <p class="hero-sub">
    {len(rfm):,} customers · RFM scoring + K-Means ML clustering ·
    Data from {df['transaction_date'].min().strftime('%b %Y')} to {df['transaction_date'].max().strftime('%b %Y')}
  </p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# KPI ROW
# ============================================================================
rev_str = f"${total_rev/1e6:.2f}M" if total_rev >= 1e6 else f"${total_rev:,.0f}"
clv_str = f"${avg_clv:,.0f}"

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-accent" style="background:linear-gradient(90deg,#06b6d4,transparent)"></div>
    <div class="kpi-label">Total Customers</div>
    <div class="kpi-val">{total_cust:,}</div>
    <div class="kpi-sub">{total_cust/len(rfm)*100:.1f}% of base</div>
  </div>
  <div class="kpi">
    <div class="kpi-accent" style="background:linear-gradient(90deg,#10b981,transparent)"></div>
    <div class="kpi-label">Total Revenue</div>
    <div class="kpi-val">{rev_str}</div>
    <div class="kpi-sub">Filtered selection</div>
  </div>
  <div class="kpi">
    <div class="kpi-accent" style="background:linear-gradient(90deg,#8b5cf6,transparent)"></div>
    <div class="kpi-label">Avg Order Value</div>
    <div class="kpi-val">${avg_order:.2f}</div>
    <div class="kpi-sub">Per transaction</div>
  </div>
  <div class="kpi">
    <div class="kpi-accent" style="background:linear-gradient(90deg,#f59e0b,transparent)"></div>
    <div class="kpi-label">Avg Est. CLV</div>
    <div class="kpi-val">{clv_str}</div>
    <div class="kpi-sub">3-year projection</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 RFM Segmentation",
    "🤖 ML Clustering",
    "📈 Revenue & Pareto",
    "🔄 Cohort Analysis"
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — RFM SEGMENTATION
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">📊</div>
      <h2>RFM Segment Overview</h2>
    </div>
    """, unsafe_allow_html=True)

    if total_cust == 0:
        st.warning("No customers match the current filters.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            sc = rfm_f['segment'].value_counts().reset_index()
            sc.columns = ['Segment', 'Count']
            colors = [SEG_COLORS.get(s, "#475569") for s in sc['Segment']]

            fig = go.Figure(go.Pie(
                labels=sc['Segment'], values=sc['Count'], hole=0.58,
                marker=dict(colors=colors, line=dict(color='#080b12', width=2)),
                textfont=dict(size=11, color='#e2e8f0'),
                hovertemplate='<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>'
            ))
            fig.update_layout(**DARK, height=360, margin=dict(**MARGIN),
                annotations=[dict(text=f"<b>{total_cust:,}</b><br><span style='font-size:10px'>customers</span>",
                                  x=0.5, y=0.5, font_size=16, showarrow=False,
                                  font_color='#e2e8f0')])
            fig.update_layout(legend=dict(font_color='#94a3b8', bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            sr = rfm_f.groupby('segment')['monetary'].sum().reset_index().sort_values('monetary')
            bar_colors = [SEG_COLORS.get(s, "#475569") for s in sr['segment']]

            fig = go.Figure(go.Bar(
                y=sr['segment'], x=sr['monetary'],
                orientation='h',
                marker=dict(color=bar_colors, opacity=0.85),
                text=[f"${v/1e6:.1f}M" if v >= 1e6 else f"${v:,.0f}" for v in sr['monetary']],
                textposition='outside',
                textfont=dict(color='#94a3b8', size=10),
                hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>'
            ))
            fig.update_layout(**DARK, height=360, margin=dict(**MARGIN),
                              xaxis_title="Total Revenue ($)",
                              xaxis=dict(**AXIS, tickformat='$,.0f'),
                              yaxis=dict(**AXIS))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Segment deep-dive table
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">📋</div>
          <h2>Segment Deep Dive</h2>
        </div>
        """, unsafe_allow_html=True)

        tbl = rfm_f.groupby('segment').agg(
            Customers=('customer_id', 'count'),
            Avg_Recency=('recency', 'mean'),
            Avg_Frequency=('frequency', 'mean'),
            Total_Revenue=('monetary', 'sum'),
            Avg_CLV=('estimated_clv', 'mean')
        ).round(1).reset_index()
        tbl['% of Total'] = (tbl['Customers'] / total_cust * 100).round(1)
        tbl['Total_Revenue'] = tbl['Total_Revenue'].apply(lambda x: f"${x:,.0f}")
        tbl['Avg_CLV'] = tbl['Avg_CLV'].apply(lambda x: f"${x:,.0f}")
        tbl.columns = ['Segment', 'Customers', 'Avg Recency (d)', 'Avg Freq', 'Revenue', 'Avg CLV', '% Total']
        st.dataframe(tbl.set_index('Segment'), use_container_width=True)

        # RFM Score Heatmap
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🔥</div>
          <h2>RFM Score Heatmap (Recency vs Frequency → Avg Monetary)</h2>
        </div>
        """, unsafe_allow_html=True)

        heatmap_data = rfm_f.groupby(['r_score', 'f_score'])['monetary'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='r_score', columns='f_score', values='monetary').fillna(0)

        fig = go.Figure(go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"F={c}" for c in heatmap_pivot.columns],
            y=[f"R={i}" for i in heatmap_pivot.index],
            colorscale=[[0, '#0e1320'], [0.5, '#0e4d6b'], [1, '#06b6d4']],
            text=[[f"${v:,.0f}" for v in row] for row in heatmap_pivot.values],
            texttemplate='%{text}',
            textfont=dict(size=10, color='#e2e8f0'),
            hovertemplate='Recency: %{y}<br>Frequency: %{x}<br>Avg Monetary: $%{z:,.0f}<extra></extra>'
        ))
        fig.update_layout(**DARK, height=340, margin=dict(**MARGIN),
                          xaxis_title="Frequency Score", yaxis_title="Recency Score",
                          xaxis=dict(**AXIS), yaxis=dict(**AXIS))
        st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if 'ml_cluster' not in rfm_f.columns or model_info is None:
        st.markdown("""
        <div class="card" style="text-align:center; padding: 3rem;">
          <div style="font-size:3rem; margin-bottom:1rem">🤖</div>
          <h3 style="color:#e2e8f0; font-family:'Playfair Display',serif;">ML Model Not Loaded</h3>
          <p style="color:#475569; margin-top:0.5rem;">
            Run <code>ml_clustering.ipynb</code> and ensure <code>models/</code> folder contains
            <code>kmeans_model.pkl</code>, <code>scaler.pkl</code>, and <code>model_summary.json</code>
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🤖</div>
          <h2>Machine Learning Clustering Results</h2>
        </div>
        """, unsafe_allow_html=True)

        # Model metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Algorithm", "K-Means")
        m2.metric("Clusters", model_info['n_clusters'])
        m3.metric("Silhouette Score", f"{model_info['metrics']['silhouette_score']:.3f}",
                  help="0 = random, 1 = perfect. ≥0.5 is good; 0.39 is moderate.")
        m4.metric("Davies-Bouldin", f"{model_info['metrics']['davies_bouldin_index']:.3f}",
                  help="Lower is better. < 1.0 is acceptable.")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Cluster Distribution**")
            cc = rfm_f['ml_cluster'].value_counts().sort_index()
            fig = go.Figure(go.Bar(
                x=cc.index.astype(str), y=cc.values,
                marker=dict(color=CLUSTER_PALETTE[:len(cc)], opacity=0.85,
                            line=dict(color='#080b12', width=1.5)),
                text=cc.values,
                textposition='outside',
                textfont=dict(color='#94a3b8', size=10),
                hovertemplate='Cluster %{x}<br>Customers: %{y:,}<extra></extra>'
            ))
            fig.update_layout(**DARK, height=300, margin=dict(**MARGIN),
                              xaxis_title="ML Cluster", yaxis_title="Customers",
                              xaxis=dict(**AXIS), yaxis=dict(**AXIS))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Revenue by Cluster**")
            rc = rfm_f.groupby('ml_cluster')['monetary'].sum().sort_values(ascending=True)
            fig = go.Figure(go.Bar(
                y=rc.index.astype(str), x=rc.values,
                orientation='h',
                marker=dict(color=[CLUSTER_PALETTE[i] for i in rc.index], opacity=0.85,
                            line=dict(color='#080b12', width=1.5)),
                text=[f"${v/1e6:.2f}M" if v >= 1e6 else f"${v:,.0f}" for v in rc.values],
                textposition='outside',
                textfont=dict(color='#94a3b8', size=10),
                hovertemplate='Cluster %{y}<br>Revenue: $%{x:,.0f}<extra></extra>'
            ))
            fig.update_layout(**DARK, height=300, margin=dict(**MARGIN),
                              xaxis_title="Revenue ($)", yaxis_title="ML Cluster",
                              xaxis=dict(**AXIS, tickformat='$,.0f'),
                              yaxis=dict(**AXIS))
            st.plotly_chart(fig, use_container_width=True)

        # Cluster profile table
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">📋</div>
          <h2>Cluster Profiles</h2>
        </div>
        """, unsafe_allow_html=True)

        cp = rfm_f.groupby('ml_cluster').agg(
            Customers=('customer_id', 'count'),
            Avg_Recency=('recency', 'mean'),
            Avg_Frequency=('frequency', 'mean'),
            Avg_Monetary=('monetary', 'mean'),
            Total_Revenue=('monetary', 'sum')
        ).round(2).reset_index()
        cp['% of Total'] = (cp['Customers'] / total_cust * 100).round(1)
        cp['Total_Revenue'] = cp['Total_Revenue'].apply(lambda x: f"${x:,.0f}")
        cp['Avg_Monetary']  = cp['Avg_Monetary'].apply(lambda x: f"${x:,.0f}")
        cp.columns = ['Cluster', 'Customers', 'Avg Recency (d)', 'Avg Frequency',
                      'Avg Spend', 'Total Revenue', '% of Total']
        st.dataframe(cp.set_index('Cluster'), use_container_width=True)

        # Scatter: Recency vs Monetary coloured by cluster
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">🔵</div>
          <h2>Cluster Scatter — Recency vs Spend</h2>
        </div>
        """, unsafe_allow_html=True)

        sample = rfm_f.sample(min(1500, len(rfm_f)), random_state=42)
        fig = go.Figure()
        for cid in sorted(sample['ml_cluster'].unique()):
            sub = sample[sample['ml_cluster'] == cid]
            fig.add_trace(go.Scatter(
                x=sub['recency'], y=sub['monetary'],
                mode='markers',
                name=f'Cluster {cid}',
                marker=dict(color=CLUSTER_PALETTE[cid % len(CLUSTER_PALETTE)],
                            size=5, opacity=0.65,
                            line=dict(width=0)),
                hovertemplate='Recency: %{x}d<br>Spend: $%{y:,.0f}<extra>Cluster ' + str(cid) + '</extra>'
            ))
        fig.update_layout(**DARK, height=380, margin=dict(**MARGIN),
                          xaxis_title="Recency (days)", yaxis_title="Total Spend ($)",
                          xaxis=dict(**AXIS), yaxis=dict(**AXIS),
                          legend=dict(font_color='#94a3b8', bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

        # ML vs RFM heatmap image
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">⚖️</div>
          <h2>ML Clusters vs RFM Segments</h2>
        </div>
        """, unsafe_allow_html=True)

        # Try multiple paths for the heatmap
        heatmap_paths = [
            'outputs/ml_results/ml_vs_rfm_heatmap.png',
            'models/ml_vs_rfm_heatmap.png',
            'ml_vs_rfm_heatmap.png'
        ]
        heatmap_loaded = False
        for path in heatmap_paths:
            try:
                st.image(path, use_container_width=True)
                heatmap_loaded = True
                break
            except:
                continue
        if not heatmap_loaded:
            st.info("Place `ml_vs_rfm_heatmap.png` in the `models/` folder to display it here.")

        # Elbow + 3D viz
        st.markdown("---")
        st.markdown("""
        <div class="section-header">
          <div class="section-icon">📈</div>
          <h2>Model Evaluation Plots</h2>
        </div>
        """, unsafe_allow_html=True)

        # Load images as base64 for controlled height display
        import base64
        from pathlib import Path

        def load_img_b64(paths):
            for p in paths:
                try:
                    data = Path(p).read_bytes()
                    return base64.b64encode(data).decode()
                except: continue
            return None

        elbow_paths = ['outputs/ml_results/elbow_analysis.png', 'models/elbow_analysis.png', 'elbow_analysis.png']
        cluster3d_paths = ['outputs/ml_results/3d_cluster_viz.png', 'models/3d_cluster_viz.png', '3d_cluster_viz.png']

        elbow_b64   = load_img_b64(elbow_paths)
        cluster_b64 = load_img_b64(cluster3d_paths)

        # Use equal-height image cards with CSS object-fit
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0.5rem;">
          <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:14px; overflow:hidden;">
            <div style="height:340px; display:flex; align-items:center; justify-content:center; padding:0.8rem;">
              {'<img src="data:image/png;base64,' + elbow_b64 + '" style="max-width:100%; max-height:100%; object-fit:contain; border-radius:8px;">' if elbow_b64 else '<p style="color:#475569; text-align:center;">elbow_analysis.png not found</p>'}
            </div>
            <div style="padding:0.6rem 1rem 0.8rem; text-align:center; font-size:0.75rem; color:#64748b; border-top:1px solid var(--border);">
              📉 Elbow Method — Optimal k selection
            </div>
          </div>
          <div style="background:var(--bg-card); border:1px solid var(--border); border-radius:14px; overflow:hidden;">
            <div style="height:340px; display:flex; align-items:center; justify-content:center; padding:0.8rem;">
              {'<img src="data:image/png;base64,' + cluster_b64 + '" style="max-width:100%; max-height:100%; object-fit:contain; border-radius:8px;">' if cluster_b64 else '<p style="color:#475569; text-align:center;">3d_cluster_viz.png not found</p>'}
            </div>
            <div style="padding:0.6rem 1rem 0.8rem; text-align:center; font-size:0.75rem; color:#64748b; border-top:1px solid var(--border);">
              🔵 3D Cluster Visualization (K-Means)
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — REVENUE & PARETO
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">💰</div>
      <h2>Revenue Distribution & Pareto Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    # Pareto
    cust_rev = rfm_f[['customer_id','monetary']].sort_values('monetary', ascending=False).reset_index(drop=True)
    cust_rev['cum_revenue'] = cust_rev['monetary'].cumsum()
    cust_rev['cum_pct']     = cust_rev['cum_revenue'] / cust_rev['monetary'].sum() * 100
    cust_rev['cust_pct']    = (cust_rev.index + 1) / len(cust_rev) * 100

    top20_rev = cust_rev[cust_rev['cust_pct'] <= 20]['cum_pct'].max()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cust_rev['cust_pct'], y=cust_rev['monetary'],
        name='Individual Revenue',
        marker=dict(color='#06b6d4', opacity=0.3),
        hovertemplate='Customer rank: %{x:.1f}%<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=cust_rev['cust_pct'], y=cust_rev['cum_pct'],
        name='Cumulative Revenue %',
        yaxis='y2',
        line=dict(color='#f59e0b', width=2.5),
        hovertemplate='Top %{x:.1f}% customers → %{y:.1f}% revenue<extra></extra>'
    ))
    fig.add_vline(x=20, line=dict(color='#10b981', dash='dash', width=1.5))
    fig.add_annotation(x=22, y=top20_rev, yref='y2',
                       text=f"Top 20% → {top20_rev:.1f}%",
                       bgcolor='#0e1320', bordercolor='#10b981',
                       font=dict(color='#10b981', size=11), showarrow=False)
    fig.update_layout(**DARK, height=380, margin=dict(**MARGIN),
                      xaxis_title="Customers (ranked by revenue) %",
                      yaxis_title="Revenue ($)",
                      xaxis=dict(**AXIS), yaxis=dict(**AXIS),
                      yaxis2=dict(title="Cumulative Revenue %", overlaying='y', side='right',
                                  gridcolor='rgba(0,0,0,0)', ticksuffix='%',
                                  color='#94a3b8'),
                      legend=dict(font_color='#94a3b8', bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Monthly Revenue Trend
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">📅</div>
      <h2>Monthly Revenue Trend</h2>
    </div>
    """, unsafe_allow_html=True)

    df_f['month'] = df_f['transaction_date'].dt.to_period('M').dt.to_timestamp()
    monthly = df_f.groupby('month').agg(
        revenue=('amount', 'sum'),
        transactions=('transaction_id', 'count')
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly['month'], y=monthly['revenue'],
        fill='tozeroy',
        fillcolor='rgba(6,182,212,0.07)',
        line=dict(color='#06b6d4', width=2),
        name='Monthly Revenue',
        hovertemplate='%{x|%b %Y}<br>Revenue: $%{y:,.0f}<extra></extra>'
    ))
    fig.update_layout(**DARK, height=320, margin=dict(**MARGIN),
                      xaxis_title="Month", yaxis_title="Revenue ($)",
                      xaxis=dict(**AXIS),
                      yaxis=dict(**AXIS, tickformat='$,.0f'))
    st.plotly_chart(fig, use_container_width=True)

    # Revenue by Category
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">🏷️</div>
      <h2>Revenue by Product Category</h2>
    </div>
    """, unsafe_allow_html=True)

    cat_rev = df_f.groupby('category')['amount'].sum().reset_index().sort_values('amount', ascending=True)
    fig = go.Figure(go.Bar(
        y=cat_rev['category'], x=cat_rev['amount'],
        orientation='h',
        marker=dict(
            color=['#06b6d4','#8b5cf6','#10b981','#f59e0b','#ef4444'],
            opacity=0.85
        ),
        text=[f"${v/1e6:.2f}M" for v in cat_rev['amount']],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=10),
        hovertemplate='%{y}<br>Revenue: $%{x:,.0f}<extra></extra>'
    ))
    fig.update_layout(**DARK, height=280, margin=dict(**MARGIN),
                      xaxis_title="Revenue ($)",
                      xaxis=dict(**AXIS, tickformat='$,.0f'),
                      yaxis=dict(**AXIS))
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — COHORT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">🔄</div>
      <h2>Cohort Retention Analysis</h2>
    </div>
    """, unsafe_allow_html=True)

    # Build cohort table
    df_cohort = df_f.copy()
    df_cohort['cohort_month'] = df_cohort.groupby('customer_id')['transaction_date'].transform('min').dt.to_period('M')
    df_cohort['order_month']  = df_cohort['transaction_date'].dt.to_period('M')
    df_cohort['cohort_index'] = (
        df_cohort['order_month'].dt.start_time.dt.to_period('M').astype(int) -
        df_cohort['cohort_month'].dt.start_time.dt.to_period('M').astype(int)
    )

    cohort_data = df_cohort.groupby(['cohort_month', 'cohort_index'])['customer_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='cohort_month', columns='cohort_index', values='customer_id')
    cohort_size  = cohort_pivot[0]
    cohort_pct   = cohort_pivot.divide(cohort_size, axis=0) * 100
    cohort_pct   = cohort_pct.iloc[:12, :12]   # first 12 cohorts × 12 months

    z_vals   = cohort_pct.values.tolist()
    x_labels = [f"Month {i}" for i in cohort_pct.columns]
    y_labels = [str(c) for c in cohort_pct.index]
    text_vals = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in cohort_pct.values]

    fig = go.Figure(go.Heatmap(
        z=z_vals, x=x_labels, y=y_labels,
        colorscale=[[0,'#1a0a0a'],[0.3,'#7f1d1d'],[0.6,'#0e4d2f'],[1,'#064e3b']],
        text=text_vals,
        texttemplate='%{text}',
        textfont=dict(size=9.5, color='#e2e8f0'),
        zmin=0, zmax=100,
        colorbar=dict(title=dict(text="Retention %", font=dict(color='#94a3b8')),
                      tickfont=dict(color='#94a3b8'))
    ))
    fig.update_layout(**DARK, height=520,
                      xaxis=dict(**AXIS, side='top', tickfont=dict(size=10, color='#94a3b8')),
                      yaxis=dict(**AXIS, tickfont=dict(size=10, color='#94a3b8')),
                      margin=dict(l=80, r=60, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Key insight callouts
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    avg_m1 = cohort_pct[1].mean() if 1 in cohort_pct.columns else 0
    avg_m6 = cohort_pct[6].mean() if 6 in cohort_pct.columns else 0
    avg_m11 = cohort_pct.iloc[:, -1].mean()

    with c1:
        st.markdown(f"""
        <div class="metric-mini">
          <div class="val">{avg_m1:.1f}%</div>
          <div class="lbl">Avg Month-1 Retention</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-mini">
          <div class="val">{avg_m6:.1f}%</div>
          <div class="lbl">Avg Month-6 Retention</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-mini">
          <div class="val">{avg_m11:.1f}%</div>
          <div class="lbl">Avg Month-11 Retention</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # CLV Distribution
    st.markdown("""
    <div class="section-header">
      <div class="section-icon">💎</div>
      <h2>Estimated CLV Distribution by Segment</h2>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure()
    for seg in rfm_f['segment'].unique():
        sub = rfm_f[rfm_f['segment'] == seg]['estimated_clv'].clip(upper=rfm_f['estimated_clv'].quantile(0.98))
        fig.add_trace(go.Box(
            y=sub, name=seg,
            marker_color=SEG_COLORS.get(seg, '#475569'),
            line=dict(width=1.5),
            boxmean=True,
            hovertemplate='%{y:$,.0f}<extra>' + seg + '</extra>'
        ))
    fig.update_layout(**DARK, height=360, margin=dict(**MARGIN),
                      yaxis_title="Estimated CLV ($)",
                      xaxis=dict(**AXIS),
                      yaxis=dict(**AXIS, tickformat='$,.0f'),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#334155; font-size:0.72rem; letter-spacing:0.5px;">
  📊 E-Commerce Customer Segmentation &nbsp;·&nbsp;
  Built with Streamlit, Scikit-learn & Plotly &nbsp;·&nbsp;
  6,000 customers · RFM + K-Means
</div>
""", unsafe_allow_html=True)