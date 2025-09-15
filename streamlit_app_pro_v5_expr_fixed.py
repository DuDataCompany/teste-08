import json
import math
import os
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Optional: scikit-learn for linear regression forecast
try:
    from sklearn.linear_model import LinearRegression
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
try:
    import matplotlib
    matplotlib.use("Agg", force=True)
except Exception:
    pass

DEFAULT_CONFIG
# ===== Helper: avaliar expressões numéricas com segurança =====
import ast

ALLOWED_BINOP = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
ALLOWED_UNARY = (ast.UAdd, ast.USub)

def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Num):  # Python <3.8 compat
        return float(node.n)
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Apenas números são permitidos.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ALLOWED_UNARY):
        return +_eval_node(node.operand) if isinstance(node.op, ast.UAdd) else -_eval_node(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ALLOWED_BINOP):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.FloorDiv): return left // right
        if isinstance(node.op, ast.Mod): return left % right
        if isinstance(node.op, ast.Pow): return left ** right
    raise ValueError("Expressão inválida. Use apenas números e operadores: + - * / // % ** e parênteses.")

def eval_numeric_expr(expr: str, *, minimum: float = 0.0) -> float:
    expr = (expr or "").strip()
    if expr == "":
        return float(minimum)
    try:
        tree = ast.parse(expr, mode="eval")
        val = float(_eval_node(tree))
        if val < minimum:
            val = minimum
        return val
    except Exception as e:
        raise ValueError(f"Não foi possível avaliar a expressão: {e}")
 = {
    "metas": {"mencoes": 500, "engajamentos": 6000, "sentimento": 70.0, "brandfit": 7.0},
    "pesos_iniciais": {"mencoes": 0.5, "engajamentos": 2.0, "sentimento": 1.0, "brandfit": 0.75},
    "caps": {"ratio_cap": 2.0},
    "logistic_k": 4.0,
}

def normalize_features(mencoes, engaj, sentimento, brandfit, CONFIG):
    metas = CONFIG["metas"]; caps = CONFIG["caps"]
    r_menc = min(max(0.0, mencoes / max(1.0, metas["mencoes"])), caps["ratio_cap"])
    r_eng  = min(max(0.0, engaj   / max(1.0, metas["engajamentos"])), caps["ratio_cap"])
    r_sent = min(max(0.0, sentimento / max(1e-6, metas["sentimento"])), caps["ratio_cap"])
    r_bfit = min(max(0.0, brandfit / max(1e-6, metas["brandfit"])), caps["ratio_cap"])
    menc_log = math.log1p(mencoes) / math.log1p(metas["mencoes"])
    eng_log  = math.log1p(engaj)   / math.log1p(metas["engajamentos"])
    sent_smooth = math.tanh(r_sent)
    bfit_smooth = math.tanh(r_bfit)
    return np.array([r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth], dtype=float)

def compute_composite(features, CONFIG):
    pesos = CONFIG["pesos_iniciais"]
    r_menc, menc_log, r_eng, eng_log, r_sent, sent_smooth, r_bfit, bfit_smooth = features
    comp_menc = 0.5*r_menc + 0.5*menc_log
    comp_eng  = 0.5*r_eng  + 0.5*eng_log
    comp_sent = 0.5*r_sent + 0.5*sent_smooth
    comp_bfit = 0.5*r_bfit + 0.5*bfit_smooth
    w = np.array([pesos["mencoes"], pesos["engajamentos"], pesos["sentimento"], pesos["brandfit"]], dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    composite = w[0]*comp_menc + w[1]*comp_eng + w[2]*comp_sent + w[3]*comp_bfit
    comps = {"mencoes": comp_menc, "engajamentos": comp_eng, "sentimento": comp_sent, "brandfit": comp_bfit}
    weights = {"mencoes": w[0], "engajamentos": w[1], "sentimento": w[2], "brandfit": w[3]}
    return composite, comps, weights

def composite_to_score(composite, CONFIG):
    k = CONFIG["logistic_k"]
    prob_like = 1.0 / (1.0 + math.exp(-k*(composite - 1.0)))
    return float(100.0 * prob_like)

def classify(score):
    if score <= 40:  return "FRIA"
    if score <= 74:  return "MORNA"
    return "QUENTE"

def pct_of_goal(value, goal):
    try:
        return max(0.0, min(100.0, (float(value) / float(goal)) * 100.0))
    except Exception:
        return 0.0

# ---- Color thresholds ----
def value_color(v: float) -> str:
    """Return hex color by thresholds: <40 red, 40-70 yellow, >70 green."""
    try:
        v = float(v)
    except Exception:
        v = 0.0
    if v < 40:
        return "#D9534F"  # red
    if v < 70:
        return "#F0AD4E"  # yellow
    return "#5CB85C"      # green

# ---- Donut helper with dynamic color and no % labels ----
def donut_plotly(value_0_100: float, title: str):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    v = float(np.clip(value_0_100, 0, 100))
    color = value_color(v)
    fig = go.Figure(data=[go.Pie(
        values=[v, 100 - v],
        labels=["", ""],
        hole=0.7,
        textinfo="none",
        sort=False,
        direction="clockwise",
        showlegend=False
    )])
    fig.update_traces(marker=dict(colors=[color, "#E6E6E6"]))
    fig.update_layout(
        title={'text': title, 'y':0.92},
        margin=dict(l=10, r=10, t=40, b=10),
        height=300, width=300,
        annotations=[dict(text=f"{v:.0f}", x=0.5, y=0.5, font_size=32, showarrow=False)]
    )
    return fig

# =========================
# App
# =========================
st.set_page_config(page_title="Temperatura da Pauta — Pro v5+Forecast", page_icon="🔥", layout="wide")
st.title("🔥 Temperatura da Pauta — Pro v5 (+ Previsão 15 dias)")

# Sidebar
st.sidebar.header("Configurações")
with st.sidebar.expander("Metas (alvos)", expanded=True):
    meta_menc = st.number_input("Meta de Menções", min_value=0, value=int(DEFAULT_CONFIG["metas"]["mencoes"]), step=50)
    meta_eng_expr  = st.text_input("Meta de Engajamentos (aceita expressão)", value=str(int(DEFAULT_CONFIG["metas"]["engajamentos"])), help="Ex.: 2*3000 + 500")
    try:
        meta_eng_val = eval_numeric_expr(meta_eng_expr, minimum=0)
    except Exception as e:
        st.warning(f"Meta de Engajamentos inválida: {e}")
        meta_eng_val = float(DEFAULT_CONFIG["metas"]["engajamentos"])
    meta_eng = int(meta_eng_val)
    meta_sent = st.number_input("Meta de Sentimento", min_value=0.0, max_value=100.0, value=float(DEFAULT_CONFIG["metas"]["sentimento"]), step=1.0)
    meta_bfit = st.number_input("Meta de Brandfit", min_value=0.0, max_value=10.0, value=float(DEFAULT_CONFIG["metas"]["brandfit"]), step=0.1)

with st.sidebar.expander("Pesos (importância relativa)", expanded=True):
    st.caption("Defina a importância relativa de cada pilar (valores serão normalizados internamente).")
    peso_menc = st.number_input("Peso - Menções",  value=float(DEFAULT_CONFIG["pesos_iniciais"]["mencoes"]), step=0.1, format="%.2f")
    peso_eng  = st.number_input("Peso - Engajamentos", value=float(DEFAULT_CONFIG["pesos_iniciais"]["engajamentos"]), step=0.1, format="%.2f")
    peso_sent = st.number_input("Peso - Sentimento", value=float(DEFAULT_CONFIG["pesos_iniciais"]["sentimento"]), step=0.1, format="%.2f")
    peso_bfit = st.number_input("Peso - Brandfit", value=float(DEFAULT_CONFIG["pesos_iniciais"]["brandfit"]), step=0.1, format="%.2f")

ratio_cap = st.sidebar.slider("Cap de razão (limite superior)", 1.0, 5.0, float(DEFAULT_CONFIG["caps"]["ratio_cap"]), 0.1)
log_k     = st.sidebar.slider("Inclinação logística (k)", 1.0, 8.0, float(DEFAULT_CONFIG["logistic_k"]), 0.5)

CONFIG = {
    "metas": {"mencoes": meta_menc, "engajamentos": meta_eng, "sentimento": meta_sent, "brandfit": meta_bfit},
    "pesos_iniciais": {"mencoes": peso_menc, "engajamentos": peso_eng, "sentimento": peso_sent, "brandfit": peso_bfit},
    "caps": {"ratio_cap": ratio_cap},
    "logistic_k": log_k
}

# Inputs
st.subheader("Entradas da Pauta")
col_a, col_b = st.columns(2)
with col_a:
    pauta = st.text_input("Nome da pauta", value="Minha Pauta")
    menc_ig = st.number_input("Menções no Instagram", min_value=0, value=120, step=10)
    eng_ig_expr  = st.text_input("Engajamentos no Instagram (aceita expressão)", value="3000", help="Ex.: 1500*2")
    try:
        eng_ig_val = eval_numeric_expr(eng_ig_expr, minimum=0)
    except Exception as e:
        st.warning(f"Engajamentos IG inválido: {e}")
        eng_ig_val = 3000.0
    eng_ig = int(eng_ig_val)
with col_b:
    menc_tw = st.number_input("Menções no Twitter/X", min_value=0, value=80, step=10)
    eng_tw_expr  = st.text_input("Engajamentos no Twitter/X (aceita expressão)", value="1800", help="Ex.: 1200 + 3*200")
    try:
        eng_tw_val = eval_numeric_expr(eng_tw_expr, minimum=0)
    except Exception as e:
        st.warning(f"Engajamentos X inválido: {e}")
        eng_tw_val = 1800.0
    eng_tw = int(eng_tw_val)
    brand   = st.number_input("Brandfit (0-10)", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
    sent    = st.number_input("Sentimento (0-100)", min_value=0.0, max_value=100.0, value=72.0, step=1.0)

menc_total = menc_ig + menc_tw
eng_total  = eng_ig + eng_tw

# Cálculos principais
feats = normalize_features(menc_total, eng_total, sent, brand, CONFIG)
composite, comps, weights = compute_composite(feats, CONFIG)
score = composite_to_score(composite, CONFIG)
classe = classify(score)

# KPIs
st.markdown("---")
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Score Final", f"{score:.1f}")
kpi2.metric("Classificação", classe)
kpi3.metric("Menções (IG+X)", f"{menc_total:,}".replace(",", "."))
kpi4.metric("Engajamentos (IG+X)", f"{eng_total:,}".replace(",", "."))
kpi5.metric("Sent/Brandfit", f"{sent:.0f} / {brand:.1f}")

# Score donut (colored by threshold)
st.subheader("Score — Visão Geral")
try:
    import plotly
    fig_score = donut_plotly(score, title=f"Score — {pauta}")
    st.plotly_chart(fig_score, use_container_width=False)
except Exception:
    st.info("Instale `plotly` para ver os donuts.")

# Four donuts with thresholds
st.subheader("Pilares — Visão 0 a 100")
men_0100 = pct_of_goal(menc_total, meta_menc)
eng_0100 = pct_of_goal(eng_total, meta_eng)
sent_0100 = float(np.clip(sent, 0, 100))
bfit_0100 = float(np.clip(brand * 10.0, 0, 100))

cols = st.columns(4)
titles = ["Menções", "Engajamentos", "Sentimento", "Brandfit"]
vals = [men_0100, eng_0100, sent_0100, bfit_0100]
for c, t, v in zip(cols, titles, vals):
    with c:
        fig = donut_plotly(v, title=t)
        st.plotly_chart(fig, use_container_width=True)

# Google Trends (histórico + previsão 15 dias)
st.subheader("Google Trends — Últimos 15 dias + Previsão (15 dias)")

@st.cache_data(show_spinner=False)
def parse_trends_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # detectar colunas
    date_col = None
    for c in df.columns:
        if str(c).strip().lower() in ['date', 'data']:
            date_col = c; break
    if date_col is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                date_col = c; break
            except: pass
    if date_col is None:
        raise ValueError("Não foi possível identificar a coluna de data.")
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.rename(columns={date_col: 'Data'})
    # detectar volume
    value_cols = [c for c in df.columns if c != 'Data' and pd.api.types.is_numeric_dtype(df[c])]
    if not value_cols:
        raise ValueError("Coluna de volume não encontrada ou não numérica.")
    return df[['Data', value_cols[0]]].rename(columns={value_cols[0]: 'Volume de Pesquisas'})

trends_file = st.file_uploader("Envie um CSV com as colunas 'data/date' e 'volume'", type=['csv'])
if trends_file is not None:
    try:
        df_trends = parse_trends_csv(trends_file)

        # Mantém apenas os últimos 15 dias no histórico visual (se desejar)
        if len(df_trends) > 15:
            df_last15 = df_trends.tail(15).copy()
        else:
            df_last15 = df_trends.copy()

        st.caption("Prévia dos dados (últimas linhas):")
        st.dataframe(df_trends.tail(10), use_container_width=True)

        # --------- Regressão Linear para 15 dias à frente ---------
        if _SKLEARN_OK and len(df_trends) >= 2:
            df_model = df_trends.copy()
            df_model['Dias'] = (df_model['Data'] - df_model['Data'].min()).dt.days  # eixo numérico

            X = df_model[['Dias']].values
            y = df_model['Volume de Pesquisas'].values

            model = LinearRegression()
            model.fit(X, y)

            last_day = df_model['Dias'].max()
            future_days = np.arange(last_day + 1, last_day + 16).reshape(-1, 1)
            future_dates = pd.date_range(df_model['Data'].max() + pd.Timedelta(days=1), periods=15)

            predictions = model.predict(future_days)
            # Evita valores negativos por ruído do modelo
            predictions = np.maximum(predictions, 0)

            df_future = pd.DataFrame({'Data': future_dates, 'Volume de Pesquisas': predictions})
            df_total = pd.concat([df_trends[['Data', 'Volume de Pesquisas']], df_future], ignore_index=True)
            df_total['Tipo'] = ['Histórico'] * len(df_trends) + ['Previsão'] * len(df_future)

            # --------- Gráfico combinado (Altair) ---------
            try:
                import altair as alt
                chart = (
                    alt.Chart(df_total)
                    .mark_line()
                    .encode(
                        x=alt.X('Data:T', title='Data'),
                        y=alt.Y('Volume de Pesquisas:Q', title='Volume de Pesquisas'),
                        color=alt.Color('Tipo', scale=alt.Scale(domain=['Histórico', 'Previsão'], range=['gray', 'blue'])),
                        tooltip=['Data:T', 'Volume de Pesquisas:Q', 'Tipo']
                    )
                    .properties(title='Tendências de Pesquisa e Previsão (15 dias)')
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.line_chart(df_total.set_index('Data')['Volume de Pesquisas'])

            st.subheader("Previsão — Próximos 15 dias")
            st.dataframe(df_future, use_container_width=True)

        else:
            if not _SKLEARN_OK:
                st.error("scikit-learn não está disponível no ambiente. Instale `scikit-learn` para habilitar a previsão por regressão linear (pip install scikit-learn).")
            else:
                st.warning("Dados insuficientes para ajustar a regressão linear (mínimo de 2 pontos).")

            # Fallback: mostra apenas o histórico (cinza)
            try:
                import altair as alt
                chart = (
                    alt.Chart(df_last15)
                    .mark_line(color='gray')
                    .encode(
                        x=alt.X('Data:T', title='Data'),
                        y=alt.Y('Volume de Pesquisas:Q', title='Volume de Pesquisas'),
                        tooltip=['Data:T', 'Volume de Pesquisas:Q']
                    )
                    .properties(title='Tendências de Pesquisa (histórico)')
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.line_chart(df_last15.set_index('Data')['Volume de Pesquisas'])

    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
else:
    st.info("Envie um CSV para visualizar o Google Trends e gerar a previsão (se scikit-learn estiver instalado).")

st.markdown("---")
st.caption("Creative Data • Temperatura da Pauta — Pro v5 (+ previsão linear de 15 dias)")