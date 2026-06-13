# ============================================================
#  app.py  —  Dashboard Analisis & Prediksi Risiko Obesitas
#  Tugas Besar Data Mining
# ============================================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# KONFIG HALAMAN
# ------------------------------------------------------------
st.set_page_config(
    page_title="Obesity Risk Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")
TARGET = "NObeyesdad"

# Palet warna
ACCENT   = "#7c5cff"
ACCENT2  = "#22d3ee"
GOOD     = "#22c55e"
WARN     = "#f59e0b"
BAD      = "#ef4444"
TXT      = "#e6e9f2"
MUTED    = "#9aa3b2"
CARD     = "rgba(255,255,255,0.04)"

# Metadata kelas (urut dari ringan -> berat)
CLASS_INFO = {
    "Insufficient_Weight": ("Berat Badan Kurang", "#38bdf8", "🔵",
        "Berat di bawah ideal. Fokus pada penambahan asupan gizi seimbang dan konsultasi gizi."),
    "Normal_Weight": ("Berat Badan Normal", "#22c55e", "🟢",
        "Kondisi ideal. Pertahankan pola makan dan aktivitas fisik yang sudah baik."),
    "Overweight_Level_I": ("Kelebihan Berat Tk. I", "#eab308", "🟡",
        "Mulai kelebihan berat. Tingkatkan aktivitas fisik dan kontrol porsi makan."),
    "Overweight_Level_II": ("Kelebihan Berat Tk. II", "#f97316", "🟠",
        "Kelebihan berat lebih lanjut. Disarankan program penurunan berat terstruktur."),
    "Obesity_Type_I": ("Obesitas Tipe I", "#ef4444", "🔴",
        "Obesitas tingkat awal. Perlu intervensi diet, olahraga, dan pemantauan kesehatan."),
    "Obesity_Type_II": ("Obesitas Tipe II", "#dc2626", "🔴",
        "Obesitas tingkat menengah. Risiko penyakit metabolik meningkat; konsultasi medis dianjurkan."),
    "Obesity_Type_III": ("Obesitas Tipe III", "#991b1b", "⛔",
        "Obesitas berat (morbid). Risiko kesehatan tinggi; perlu penanganan medis intensif."),
}
CLASS_ORDER = list(CLASS_INFO.keys())

# ------------------------------------------------------------
# CSS — tampilan advanced
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }

.stApp {
    background:
      radial-gradient(900px 500px at 12% -8%, rgba(124,92,255,0.18), transparent 60%),
      radial-gradient(800px 500px at 95% 0%, rgba(34,211,238,0.14), transparent 55%),
      linear-gradient(160deg, #0b0d17 0%, #0e1120 55%, #0a0c14 100%);
    color: #e6e9f2;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1280px;}

/* Hero */
.hero {
    border-radius: 22px; padding: 26px 30px; margin-bottom: 18px;
    background: linear-gradient(120deg, rgba(124,92,255,0.22), rgba(34,211,238,0.12));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    position: relative; overflow: hidden;
}
.hero:before{
    content:""; position:absolute; inset:0;
    background: radial-gradient(600px 200px at 90% -40%, rgba(255,255,255,0.10), transparent 60%);
}
.hero h1 {font-size: 30px; font-weight: 800; margin: 0 0 6px 0; letter-spacing:-0.5px;}
.hero p  {color:#c7cddd; margin:0; font-size: 14.5px; max-width: 820px;}
.hero .tag {display:inline-block; margin-top:12px; padding:5px 12px; font-size:12px;
    border-radius:999px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12);
    color:#dfe5f2;}

/* Section title */
.sec {font-size: 20px; font-weight: 700; margin: 26px 0 4px 0;
    background: linear-gradient(90deg,#c9b8ff,#8ee9f7); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;}
.sec-sub {color:#9aa3b2; font-size:13px; margin-bottom:14px;}

/* Metric card */
.kpi {
    border-radius:18px; padding:18px 18px 16px 18px; height:100%;
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    border:1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 22px rgba(0,0,0,0.35); transition: transform .18s ease, box-shadow .18s ease;
}
.kpi:hover {transform: translateY(-3px); box-shadow: 0 14px 30px rgba(124,92,255,0.22);}
.kpi .lbl {color:#9aa3b2; font-size:12.5px; font-weight:600; letter-spacing:.3px; text-transform:uppercase;}
.kpi .val {font-size:26px; font-weight:800; margin-top:6px; letter-spacing:-0.5px;}
.kpi .sub {color:#7f8aa0; font-size:12px; margin-top:2px;}

/* Insight box */
.insight {
    border-left: 3px solid #7c5cff; background: rgba(124,92,255,0.07);
    border-radius: 12px; padding: 13px 16px; margin: 8px 0 4px 0;
    font-size: 13.6px; color:#d4dbec; line-height:1.55;
}
.insight b {color:#fff;}
.note {
    border-left: 3px solid #f59e0b; background: rgba(245,158,11,0.07);
    border-radius: 12px; padding: 12px 16px; margin: 8px 0; font-size:13px; color:#f3e3c4;
}

/* Panel card wrapper */
.panel {
    background: rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07);
    border-radius:18px; padding:14px 16px 6px 16px; margin-bottom:6px;
}

/* Result badge (prediksi) */
.result {
    border-radius:20px; padding:24px 26px; margin-top:6px; color:#fff;
    box-shadow: 0 14px 44px rgba(0,0,0,0.5); border:1px solid rgba(255,255,255,0.15);
}
.result .small {font-size:13px; opacity:.9; letter-spacing:.4px; text-transform:uppercase;}
.result .big {font-size:30px; font-weight:800; margin:4px 0 8px 0;}
.result .desc {font-size:14px; opacity:.95; line-height:1.55;}

/* Tabs tweak */
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04); border-radius:10px 10px 0 0;
    padding: 8px 16px; border:1px solid rgba(255,255,255,0.06);
}
.stTabs [aria-selected="true"] {background: rgba(124,92,255,0.18); color:#fff;}

div[data-testid="stSidebar"] {background: linear-gradient(180deg,#0c0e1a,#0a0c14);
    border-right:1px solid rgba(255,255,255,0.06);}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# PIPELINE (cached) — replikasi penuh dari notebook
# ------------------------------------------------------------
def make_preprocessor(num_f, cat_f):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scl", StandardScaler())]), num_f),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_f),
    ])

@st.cache_resource(show_spinner="Menjalankan pipeline Data Mining (sekali saja)...")
def run_pipeline():
    R = {}
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ File CSV tidak ditemukan di:\n{DATA_PATH}\n\n"
                 f"Taruh 'ObesityDataSet_raw_and_data_sinthetic.csv' di folder yang sama dengan app.py.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    R["raw_shape"] = df.shape

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns
                        if pd.api.types.is_string_dtype(df[c]) and c != TARGET]

    R["numeric_cols"] = numeric_cols
    R["categorical_cols"] = categorical_cols
    R["dup"] = int(df.duplicated().sum())
    R["class_dist"] = df[TARGET].value_counts()
    R["missing"] = pd.DataFrame({
        "Kolom": df.columns,
        "Missing": df.isnull().sum().values,
        "Persen (%)": (df.isnull().mean().values * 100).round(2)
    })
    R["describe"] = df[numeric_cols].describe().T.round(2)
    R["head"] = df.head(8)

    # BMI per kelas
    dvis = df.copy()
    dvis["BMI"] = dvis["Weight"] / (dvis["Height"] ** 2)
    R["bmi_df"] = dvis[[TARGET, "BMI"]]

    # cleaning
    dclean = df.copy().drop_duplicates()
    for c in numeric_cols:
        dclean[c] = dclean[c].fillna(dclean[c].median())
    for c in categorical_cols + [TARGET]:
        dclean[c] = dclean[c].fillna(dclean[c].mode()[0])
    dcap = dclean.copy()
    for c in numeric_cols:
        q1, q3 = dcap[c].quantile(0.25), dcap[c].quantile(0.75)
        iqr = q3 - q1
        dcap[c] = dcap[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    R["clean_shape"] = dcap.shape
    R["corr"] = dcap[numeric_cols].corr().round(2)
    R["cat_unique"] = {c: sorted(dcap[c].unique().tolist()) for c in categorical_cols}
    R["num_stats"] = {c: (float(dcap[c].min()), float(dcap[c].max()),
                          float(dcap[c].median())) for c in numeric_cols}

    # feature selection
    denc = dcap.copy()
    le = LabelEncoder()
    for c in categorical_cols + [TARGET]:
        denc[c] = le.fit_transform(denc[c].astype(str))
    Xfs, yfs = denc.drop(columns=[TARGET]), denc[TARGET]
    sel = SelectKBest(score_func=f_classif, k="all").fit(Xfs, yfs)
    fdf = pd.DataFrame({"Feature": Xfs.columns, "Score": sel.scores_}).sort_values("Score", ascending=False)
    rffs = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xfs, yfs)
    rdf = pd.DataFrame({"Feature": Xfs.columns, "Importance": rffs.feature_importances_}).sort_values("Importance", ascending=False)
    R["fscore"], R["rfimp"] = fdf, rdf

    topf, topr = set(fdf.head(10)["Feature"]), set(rdf.head(10)["Feature"])
    selected = rdf[rdf["Feature"].isin(list(topf & topr))]["Feature"].tolist()
    R["selected"] = selected

    num_f = [c for c in selected if not pd.api.types.is_string_dtype(dcap[c])]
    cat_f = [c for c in selected if pd.api.types.is_string_dtype(dcap[c])]
    R["num_f"], R["cat_f"] = num_f, cat_f

    X, y = dcap[selected], dcap[TARGET]
    R["classes"] = sorted(y.unique().tolist())
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # clustering
    cl_pre = make_preprocessor(num_f, cat_f)
    Xc = cl_pre.fit_transform(X)
    Krange = list(range(2, 9))
    inertias, sils = [], []
    for k in Krange:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(Xc)
        inertias.append(km.inertia_); sils.append(silhouette_score(Xc, lab))
    best_k = int(Krange[int(np.argmax(sils))])
    R["Krange"], R["inertias"], R["sils"], R["best_k"] = Krange, inertias, sils, best_k

    kmf = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clab = kmf.fit_predict(Xc)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xc)
    R["pca"] = Xp; R["clab"] = clab
    R["pca_var"] = (pca.explained_variance_ratio_ * 100).round(1)

    dcl = dcap.copy(); dcl["Cluster"] = clab
    R["cluster_count"] = dcl["Cluster"].value_counts().sort_index()
    R["cluster_profile"] = dcl.groupby("Cluster")[num_f].mean().round(2)
    R["cluster_obes"] = pd.crosstab(dcl["Cluster"], dcl[TARGET], normalize="index").round(3)

    # classification
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    accs, cms, reports = {}, {}, {}
    best_acc, best_name, best_pipe = 0, None, None
    labels_sorted = sorted(y.unique())
    for name, mdl in models.items():
        pipe = Pipeline([("pre", make_preprocessor(num_f, cat_f)), ("model", mdl)])
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xte)
        acc = accuracy_score(yte, yp)
        accs[name] = round(acc, 4)
        cms[name] = confusion_matrix(yte, yp, labels=labels_sorted)
        reports[name] = classification_report(yte, yp, output_dict=True, zero_division=0)
        if acc > best_acc:
            best_acc, best_name, best_pipe = acc, name, pipe
    R["accs"], R["cms"], R["reports"] = accs, cms, reports
    R["best_name"], R["best_acc"] = best_name, round(best_acc, 4)
    R["labels_sorted"] = labels_sorted

    # cross-val pada model terbaik
    cv = cross_val_score(best_pipe, X, y, cv=5, scoring="accuracy")
    R["cv"] = cv
    R["cv_mean"], R["cv_std"] = float(cv.mean()), float(cv.std())

    # regression FAF
    rtgt = "FAF"
    allf = [c for c in dcap.columns if c not in [TARGET, rtgt]]
    rnum = [c for c in allf if not pd.api.types.is_string_dtype(dcap[c])]
    rcat = [c for c in allf if pd.api.types.is_string_dtype(dcap[c])]
    Xr, yr = dcap[allf], dcap[rtgt]
    Xrt, Xre, yrt, yre = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg_rows = []
    for name, mdl in {"Linear Regression": LinearRegression(),
                      "Random Forest Regressor": RandomForestRegressor(random_state=42)}.items():
        p = Pipeline([("pre", make_preprocessor(rnum, rcat)), ("model", mdl)])
        p.fit(Xrt, yrt); yp = p.predict(Xre)
        reg_rows.append({"Model": name,
                         "MAE": round(mean_absolute_error(yre, yp), 4),
                         "RMSE": round(np.sqrt(mean_squared_error(yre, yp)), 4),
                         "R2": round(r2_score(yre, yp), 4)})
    R["reg"] = pd.DataFrame(reg_rows)

    # model final di-fit ulang pada SELURUH data untuk dipakai prediksi
    final = Pipeline([("pre", make_preprocessor(num_f, cat_f)), ("model", models[best_name])])
    final.fit(X, y)
    R["final_model"] = final
    return R

R = run_pipeline()

# ------------------------------------------------------------
# Helper visual
# ------------------------------------------------------------
def style_fig(fig, h=360, legend=True):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TXT, family="Plus Jakarta Sans", size=12),
        margin=dict(l=10, r=10, t=40, b=10), height=h,
        title=dict(font=dict(size=15, color="#dfe5f2")),
        legend=dict(bgcolor="rgba(0,0,0,0)") if legend else None,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig

def insight(txt):
    st.markdown(f'<div class="insight">{txt}</div>', unsafe_allow_html=True)

def note(txt):
    st.markdown(f'<div class="note">⚠️ {txt}</div>', unsafe_allow_html=True)

def section(title, sub=""):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)

def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="lbl">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)

SEQ = px.colors.sequential.Plasma

# ------------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h2 style='font-weight:800;letter-spacing:-.5px;margin-bottom:0'>🩺 Obesity<span style='color:{ACCENT}'>IQ</span></h2>"
                f"<p style='color:{MUTED};font-size:12.5px;margin-top:2px'>Health Profile & Risk Mining</p>",
                unsafe_allow_html=True)
    st.markdown("---")
    page = option_menu(
        None,
        ["Dashboard Analisis", "Prediksi Risiko"],
        icons=["bar-chart-line-fill", "activity"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"color": ACCENT2, "font-size": "17px"},
            "nav-link": {"font-size": "14px", "font-weight": "600",
                         "color": "#c4cbdc", "border-radius": "12px",
                         "margin": "4px 0", "--hover-color": "rgba(124,92,255,0.15)"},
            "nav-link-selected": {"background": "linear-gradient(90deg,#7c5cff,#5b8def)",
                                  "color": "white"},
        },
    )
    st.markdown("---")
    kc1, kc2 = st.columns(2)
    kc1.metric("Akurasi", f"{R['best_acc']*100:.1f}%")
    kc2.metric("Data", f"{R['clean_shape'][0]:,}")
    st.caption(f"Model final: **{R['best_name']}**")
    st.caption("Tugas Besar Data Mining")

# ============================================================
#  HALAMAN 1 — DASHBOARD ANALISIS
# ============================================================
if page == "Dashboard Analisis":
    st.markdown(f"""
    <div class="hero">
      <h1>Analisis Segmentasi Profil Kesehatan & Risiko Obesitas</h1>
      <p>Eksplorasi penuh dari pipeline CRISP-DM: identifikasi data, EDA, feature selection,
      clustering segmentasi, hingga perbandingan model klasifikasi & regresi —
      lengkap dengan interpretasi tiap visual.</p>
      <span class="tag">Dataset: Estimation of Obesity Levels (UCI / Kaggle)</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Total Data", f"{R['clean_shape'][0]:,}", f"raw {R['raw_shape'][0]:,} · -{R['dup']} duplikat")
    kpi(c2, "Kelas Target", f"{len(R['classes'])}", "tingkat obesitas")
    kpi(c3, "Fitur Terpilih", f"{len(R['selected'])}", "irisan ANOVA ∩ RF")
    kpi(c4, "Cluster Optimal", f"{R['best_k']}", f"silhouette {max(R['sils']):.3f}")
    kpi(c5, "Akurasi Terbaik", f"{R['best_acc']*100:.1f}%", R['best_name'])

    tabs = st.tabs(["📋 Ringkasan Data", "📊 Eksplorasi (EDA)",
                    "🎯 Feature Selection", "🧩 Clustering", "🤖 Modeling"])

    # ---------- TAB 1: Ringkasan ----------
    with tabs[0]:
        section("Distribusi Kelas Target", "Sebaran 7 tingkat obesitas pada dataset.")
        cd = R["class_dist"].reset_index()
        cd.columns = ["Kelas", "Jumlah"]
        fig = px.bar(cd, x="Kelas", y="Jumlah", color="Jumlah", color_continuous_scale=SEQ, text="Jumlah")
        fig.update_traces(textposition="outside")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        insight("Dataset relatif <b>seimbang</b> antar kelas — efek dari proses sintetis "
                "(SMOTE) pada data asli. Keseimbangan ini bagus karena model tidak bias ke kelas mayoritas.")

        cL, cR = st.columns([1.4, 1])
        with cL:
            section("Cuplikan Dataset", "8 baris pertama data mentah.")
            st.dataframe(R["head"], use_container_width=True, height=320)
        with cR:
            section("Kualitas Data", "Cek missing value & duplikat.")
            miss = R["missing"][R["missing"]["Missing"] > 0]
            if len(miss) == 0:
                st.success(f"✅ Tidak ada missing value. Duplikat ditemukan: {R['dup']} baris (sudah dihapus).")
            else:
                st.dataframe(miss, use_container_width=True)
            st.markdown(f"<div class='panel'><b>Numerik:</b> {len(R['numeric_cols'])} kolom &nbsp;·&nbsp; "
                        f"<b>Kategorikal:</b> {len(R['categorical_cols'])} kolom</div>", unsafe_allow_html=True)

        section("Statistik Deskriptif Numerik")
        st.dataframe(R["describe"], use_container_width=True)
        insight("Ringkasan ini dipakai untuk menangkap skala tiap fitur. Terlihat <b>Weight</b> punya rentang "
                "paling lebar (≈39–173 kg), sehingga <b>scaling wajib</b> sebelum modeling agar fitur berskala besar "
                "tidak mendominasi.")

    # ---------- TAB 2: EDA ----------
    with tabs[1]:
        section("Distribusi Fitur Numerik", "Histogram + kurva kepadatan tiap fitur numerik.")
        ncols = R["numeric_cols"]
        df_raw = pd.read_csv(DATA_PATH); df_raw.columns = df_raw.columns.str.strip()
        rows = int(np.ceil(len(ncols) / 2))
        fig = make_subplots(rows=rows, cols=2, subplot_titles=ncols)
        for i, c in enumerate(ncols):
            r, cc = i // 2 + 1, i % 2 + 1
            fig.add_trace(go.Histogram(x=df_raw[c], marker_color=ACCENT, opacity=0.8, name=c), row=r, col=cc)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 150 * rows + 60, legend=False), use_container_width=True)
        insight("<b>Age</b> condong ke kanan (mayoritas dewasa muda ≈18–26 th). "
                "<b>FCVC, NCP, CH2O, FAF, TUE</b> tampak diskret/bertingkat karena diisi pada skala terbatas. "
                "<b>Weight</b> & <b>Height</b> mendekati normal.")

        section("Deteksi Outlier (Boxplot)")
        sel_box = st.selectbox("Pilih fitur untuk dilihat detail:", ncols, index=ncols.index("Weight"))
        fig = px.box(df_raw, x=sel_box, points="outliers", color_discrete_sequence=[ACCENT2])
        st.plotly_chart(style_fig(fig, 240), use_container_width=True)
        insight("Outlier yang muncul <b>tidak dibuang</b>, melainkan dibatasi (capping IQR) agar jumlah data "
                "tidak berkurang banyak — pilihan tepat untuk dataset berukuran sedang.")

        section("Distribusi Fitur Kategorikal")
        ccols = R["categorical_cols"]
        sel_cat = st.selectbox("Pilih fitur kategorikal:", ccols, index=0)
        vc = df_raw[sel_cat].value_counts().reset_index()
        vc.columns = ["Kategori", "Jumlah"]
        fig = px.bar(vc, x="Kategori", y="Jumlah", color="Jumlah", color_continuous_scale=SEQ, text="Jumlah")
        fig.update_traces(textposition="outside")
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        insight("Pola kebiasaan terlihat jelas: misalnya mayoritas responden punya <b>riwayat keluarga overweight</b> "
                "dan memilih <b>transportasi umum</b> — sinyal awal bahwa faktor genetik & gaya hidup berperan.")

        cL, cR = st.columns(2)
        with cL:
            section("Heatmap Korelasi")
            corr = R["corr"]
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(style_fig(fig, 420, legend=False), use_container_width=True)
            insight("Korelasi <b>Weight–Height</b> positif moderat. Antar fitur lain korelasinya rendah, "
                    "artinya minim <b>multikolinearitas</b> — bagus untuk model linear.")
        with cR:
            section("Distribusi BMI per Tingkat Obesitas")
            bmi = R["bmi_df"].copy()
            order = bmi.groupby(TARGET)["BMI"].median().sort_values().index.tolist()
            fig = px.box(bmi, x=TARGET, y="BMI", color=TARGET,
                         category_orders={TARGET: order}, color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_xaxes(tickangle=30)
            st.plotly_chart(style_fig(fig, 420, legend=False), use_container_width=True)
            insight("BMI naik <b>monoton & rapi</b> mengikuti tingkat obesitas — label target sangat konsisten "
                    "dengan rumus BMI. Inilah alasan <b>Weight & Height</b> jadi prediktor terkuat dan akurasi model tinggi.")

    # ---------- TAB 3: Feature Selection ----------
    with tabs[2]:
        section("Metode 1 — ANOVA F-Score", "Mengukur seberapa kuat tiap fitur memisahkan antar kelas secara statistik.")
        fdf = R["fscore"]
        fig = px.bar(fdf, x="Score", y="Feature", orientation="h", color="Score",
                     color_continuous_scale=SEQ)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 420, legend=False), use_container_width=True)

        section("Metode 2 — Random Forest Importance", "Kontribusi tiap fitur dalam ensemble pohon keputusan.")
        rdf = R["rfimp"]
        fig = px.bar(rdf, x="Importance", y="Feature", orientation="h", color="Importance",
                     color_continuous_scale="Magma")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 420, legend=False), use_container_width=True)

        section("Hasil Seleksi — Irisan Top-10 Kedua Metode")
        cols = st.columns(len(R["selected"]))
        for i, f in enumerate(R["selected"]):
            cols[i].markdown(f"<div class='kpi' style='text-align:center'><div class='val' "
                             f"style='font-size:17px'>{f}</div></div>", unsafe_allow_html=True)
        st.write("")
        insight(f"Hanya fitur yang masuk <b>top-10 di KEDUA metode</b> yang dipakai → "
                f"<b>{', '.join(R['selected'])}</b>. Strategi irisan ini membuang fitur yang kebetulan "
                f"tinggi di satu metode saja, sehingga seleksi lebih <b>robust</b>. "
                f"Weight, Height, Age konsisten di puncak keduanya.")

    # ---------- TAB 4: Clustering ----------
    with tabs[3]:
        section("Penentuan Jumlah Cluster Optimal", "Elbow Method (inertia) & Silhouette Score otomatis.")
        cL, cR = st.columns(2)
        with cL:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=R["Krange"], y=R["inertias"], mode="lines+markers",
                                     line=dict(color=ACCENT, width=3), marker=dict(size=9)))
            fig.update_layout(title="Elbow Method", xaxis_title="K", yaxis_title="Inertia")
            st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)
        with cR:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=R["Krange"], y=R["sils"], mode="lines+markers",
                                     line=dict(color=ACCENT2, width=3), marker=dict(size=9)))
            fig.add_vline(x=R["best_k"], line_dash="dash", line_color=GOOD,
                          annotation_text=f"Best K={R['best_k']}")
            fig.update_layout(title="Silhouette Score", xaxis_title="K", yaxis_title="Score")
            st.plotly_chart(style_fig(fig, 340, legend=False), use_container_width=True)
        note(f"K terbaik dipilih otomatis dari silhouette tertinggi → <b>K={R['best_k']}</b>. "
             f"Catatan jujur: silhouette di sini terus naik sampai batas atas rentang uji (K=8), "
             f"dan nilainya tergolong sedang (≈{max(R['sils']):.2f}). Artinya segmen <b>saling overlap</b> "
             f"dan batas antar cluster tidak super tajam — wajar untuk data profil manusia.")

        section("Visualisasi Cluster (PCA 2D)", "Reduksi dimensi agar cluster bisa dilihat di 2 sumbu.")
        pca_df = pd.DataFrame({"PCA1": R["pca"][:, 0], "PCA2": R["pca"][:, 1],
                               "Cluster": R["clab"].astype(str)})
        fig = px.scatter(pca_df, x="PCA1", y="PCA2", color="Cluster", opacity=0.75,
                         color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(title=f"Segmentasi K-Means (K={R['best_k']})")
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)
        insight(f"Dua komponen PCA menjelaskan ≈{R['pca_var'][0]}% + {R['pca_var'][1]}% variansi. "
                f"Beberapa cluster terpisah jelas, sebagian saling bersinggungan — konsisten dengan nilai silhouette tadi.")

        cL, cR = st.columns([1, 1])
        with cL:
            section("Profil Rata-rata per Cluster")
            st.dataframe(R["cluster_profile"], use_container_width=True)
        with cR:
            section("Komposisi Obesitas per Cluster")
            co = R["cluster_obes"]
            fig = go.Figure()
            for col in co.columns:
                fig.add_trace(go.Bar(name=col, x=co.index.astype(str), y=co[col]))
            fig.update_layout(barmode="stack", title="Proporsi tingkat obesitas",
                              xaxis_title="Cluster", yaxis_title="Proporsi")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
        insight("Tiap cluster punya <b>komposisi obesitas berbeda</b> — membuktikan segmentasi menangkap "
                "kelompok profil kesehatan yang berbeda (mis. cluster yang didominasi berat normal vs cluster "
                "yang condong ke obesitas berat).")

    # ---------- TAB 5: Modeling ----------
    with tabs[4]:
        section("Perbandingan Akurasi Model Klasifikasi", "Tiga algoritma diuji pada data test (20%).")
        acc_df = pd.DataFrame([{"Model": k, "Accuracy": v} for k, v in R["accs"].items()]).sort_values("Accuracy")
        fig = px.bar(acc_df, x="Accuracy", y="Model", orientation="h", text="Accuracy",
                     color="Accuracy", color_continuous_scale=SEQ, range_x=[0.8, 1.0])
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        st.plotly_chart(style_fig(fig, 300, legend=False), use_container_width=True)
        insight(f"<b>{R['best_name']}</b> menang dengan akurasi <b>{R['best_acc']*100:.2f}%</b>, "
                f"mengungguli Decision Tree & Logistic Regression. Ensemble RF unggul karena mampu menangkap "
                f"interaksi non-linear antar fitur (Weight, Height, Age, dll).")

        section("Confusion Matrix per Model", "Diagonal = prediksi benar. Pilih model di bawah.")
        msel = st.radio("Model:", list(R["cms"].keys()), horizontal=True,
                        index=list(R["cms"].keys()).index(R["best_name"]))
        cm = R["cms"][msel]
        labs = [l.replace("_", " ") for l in R["labels_sorted"]]
        fig = px.imshow(cm, text_auto=True, x=labs, y=labs, color_continuous_scale="Blues", aspect="auto")
        fig.update_layout(title=f"Confusion Matrix — {msel}", xaxis_title="Prediksi", yaxis_title="Aktual")
        fig.update_xaxes(tickangle=30)
        st.plotly_chart(style_fig(fig, 480, legend=False), use_container_width=True)

        rep = R["reports"][msel]
        rep_df = pd.DataFrame(rep).T
        rep_df = rep_df.loc[[c for c in R["labels_sorted"] if c in rep_df.index]][["precision", "recall", "f1-score", "support"]].round(3)
        st.dataframe(rep_df, use_container_width=True)
        insight("Mayoritas prediksi jatuh di diagonal. Kesalahan kecil biasanya terjadi antar kelas "
                "<b>bertetangga</b> (mis. Overweight II ↔ Obesity I) karena profilnya memang berdekatan.")

        cL, cR = st.columns([1.1, 1])
        with cL:
            section("Validasi Cross-Validation (5-Fold)")
            cvv = R["cv"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(len(cvv))], y=cvv,
                                 marker_color=ACCENT, text=[f"{v:.3f}" for v in cvv], textposition="outside"))
            fig.add_hline(y=R["cv_mean"], line_dash="dash", line_color=GOOD,
                          annotation_text=f"Mean={R['cv_mean']:.3f}")
            fig.update_layout(title=f"CV Accuracy — {R['best_name']}", yaxis=dict(range=[0.8, 1.0]))
            st.plotly_chart(style_fig(fig, 360, legend=False), use_container_width=True)
            insight(f"Rata-rata CV <b>{R['cv_mean']*100:.2f}%</b> (std ±{R['cv_std']*100:.2f}%). "
                    f"Std kecil = model <b>stabil & tidak overfit</b>; performa konsisten di berbagai potongan data.")
        with cR:
            section("Regresi — Prediksi FAF")
            st.dataframe(R["reg"], use_container_width=True)
            best_reg = R["reg"].sort_values("R2", ascending=False).iloc[0]
            note(f"Target FAF (frekuensi aktivitas fisik) <b>sulit diprediksi</b>: R² terbaik hanya "
                 f"≈{best_reg['R2']:.2f} ({best_reg['Model']}). Artinya fitur yang ada belum cukup "
                 f"menjelaskan kebiasaan olahraga seseorang — sebuah temuan yang jujur, bukan kegagalan model.")

# ============================================================
#  HALAMAN 2 — PREDIKSI
# ============================================================
else:
    st.markdown(f"""
    <div class="hero">
      <h1>Prediksi Tingkat Risiko Obesitas</h1>
      <p>Masukkan profil fisik & gaya hidup, model <b>{R['best_name']}</b> (akurasi {R['best_acc']*100:.1f}%)
      akan memprediksi tingkat obesitas beserta probabilitas tiap kelas.</p>
      <span class="tag">Input = 7 fitur terpilih yang dipakai model final</span>
    </div>
    """, unsafe_allow_html=True)

    selected = R["selected"]
    nstats = R["num_stats"]
    catopt = R["cat_unique"]

    # label ramah untuk input
    LBL = {
        "Weight": "Berat Badan (kg)", "Height": "Tinggi Badan (m)", "Age": "Usia (tahun)",
        "FCVC": "Frekuensi Makan Sayur (1–3)", "NCP": "Jumlah Makan Utama/hari (1–4)",
        "Gender": "Jenis Kelamin", "CALC": "Konsumsi Alkohol",
    }
    GENDER_MAP = {"Female": "Perempuan", "Male": "Laki-laki"}
    CALC_MAP = {"no": "Tidak Pernah", "Sometimes": "Kadang-kadang",
                "Frequently": "Sering", "Always": "Selalu"}

    section("Input Profil", "Lengkapi data berikut lalu klik Prediksi.")
    inputs = {}
    cols = st.columns(3)
    i = 0
    for f in selected:
        col = cols[i % 3]; i += 1
        if f in catopt:  # kategorikal
            opts = catopt[f]
            if f == "Gender":
                disp = [GENDER_MAP.get(o, o) for o in opts]
                pick = col.selectbox(LBL.get(f, f), disp)
                inputs[f] = opts[disp.index(pick)]
            elif f == "CALC":
                opts_sorted = [o for o in ["no", "Sometimes", "Frequently", "Always"] if o in opts]
                disp = [CALC_MAP.get(o, o) for o in opts_sorted]
                pick = col.selectbox(LBL.get(f, f), disp)
                inputs[f] = opts_sorted[disp.index(pick)]
            else:
                inputs[f] = col.selectbox(LBL.get(f, f), opts)
        else:  # numerik
            lo, hi, med = nstats[f]
            if f in ("FCVC", "NCP"):
                inputs[f] = col.slider(LBL.get(f, f), float(lo), float(hi), float(round(med)), step=1.0)
            elif f == "Height":
                inputs[f] = col.number_input(LBL.get(f, f), float(lo), float(hi), float(round(med, 2)), step=0.01, format="%.2f")
            else:
                inputs[f] = col.number_input(LBL.get(f, f), float(lo), float(hi), float(round(med)), step=1.0)

    st.write("")
    go_pred = st.button("🔍  Prediksi Sekarang", type="primary", use_container_width=True)

    if go_pred:
        Xnew = pd.DataFrame([inputs])[selected]
        model = R["final_model"]
        pred = model.predict(Xnew)[0]
        proba = model.predict_proba(Xnew)[0]
        classes = list(model.classes_)

        label, color, emoji, desc = CLASS_INFO.get(pred, (pred, ACCENT, "•", ""))
        bmi = inputs["Weight"] / (inputs["Height"] ** 2)
        conf = proba[classes.index(pred)] * 100

        st.markdown(f"""
        <div class="result" style="background:linear-gradient(120deg,{color},#1f2433);">
          <div class="small">Hasil Prediksi {emoji}</div>
          <div class="big">{label}</div>
          <div class="desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        m1, m2, m3 = st.columns(3)
        kpi(m1, "Tingkat (label asli)", pred.replace("_", " "))
        kpi(m2, "Keyakinan Model", f"{conf:.1f}%")
        kpi(m3, "Estimasi BMI", f"{bmi:.1f}", "Weight / Height²")

        cL, cR = st.columns([1.15, 1])
        with cL:
            section("Probabilitas per Kelas")
            pe = pd.DataFrame({"Kelas": [c.replace("_", " ") for c in classes],
                               "Prob": (proba * 100).round(2),
                               "raw": classes}).sort_values("Prob", ascending=True)
            cmap = [CLASS_INFO.get(c, ("", ACCENT))[1] for c in pe["raw"]]
            fig = go.Figure(go.Bar(x=pe["Prob"], y=pe["Kelas"], orientation="h",
                                   marker_color=cmap, text=[f"{v:.1f}%" for v in pe["Prob"]],
                                   textposition="outside"))
            fig.update_layout(title="Distribusi probabilitas (%)", xaxis=dict(range=[0, 105]))
            st.plotly_chart(style_fig(fig, 420, legend=False), use_container_width=True)
        with cR:
            section("Indikator BMI")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=round(bmi, 1),
                number={"suffix": "", "font": {"color": TXT, "size": 40}},
                gauge={
                    "axis": {"range": [10, 50], "tickcolor": MUTED},
                    "bar": {"color": color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [10, 18.5], "color": "rgba(56,189,248,0.35)"},
                        {"range": [18.5, 25], "color": "rgba(34,197,94,0.35)"},
                        {"range": [25, 30], "color": "rgba(245,158,11,0.35)"},
                        {"range": [30, 50], "color": "rgba(239,68,68,0.35)"},
                    ],
                }))
            st.plotly_chart(style_fig(fig, 420, legend=False), use_container_width=True)
            st.caption("Zona: <18.5 kurus · 18.5–25 normal · 25–30 overweight · >30 obesitas")

        insight("Probabilitas tertinggi menentukan label akhir. Jika dua kelas bertetangga punya probabilitas "
                "berdekatan, artinya profil berada di <b>ambang transisi</b> antar tingkat — perhatikan tren "
                "berat & pola makan.")
        st.caption("⚠️ Hasil ini adalah estimasi berbasis model statistik untuk keperluan akademik, "
                   "bukan diagnosis medis. Konsultasikan ke tenaga kesehatan untuk penilaian sebenarnya.")
