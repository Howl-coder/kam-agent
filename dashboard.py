"""
dashboard.py - Rappi KAM Early Warning System
Solo UI — toda la lógica vive en agent/

Ejecutar: streamlit run dashboard.py
"""

import re
import sys
import pickle
import streamlit as st
import pandas as pd
import os
sys.path.append(".")
from agent.features import build_features, FEATURE_SET
from agent.alerter import (
    get_kam_portfolio,
    flag_border_cases,
    TOOLS,
)
import anthropic

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Rappi KAM Agent", page_icon="🟠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Nunito', sans-serif; }

.rappi-header {
    background: linear-gradient(135deg, #FF441F 0%, #ff6b35 100%);
    border-radius: 16px; padding: 24px 32px; margin-bottom: 16px;
}
.rappi-header h1 { color:white!important; font-size:28px!important;
    font-weight:800!important; margin:0 0 4px 0!important; }
.rappi-header p  { color:rgba(255,255,255,0.85)!important;
    font-size:14px!important; margin:0!important; }

.metric-card { border-radius:12px; padding:16px 20px;
    margin-bottom:8px; font-weight:700; }
.card-critico { background:#fde8e8; border-left:5px solid #FF441F; color:#822727; }
.card-riesgo  { background:#fef3cd; border-left:5px solid #d69e2e; color:#744210; }
.card-estable { background:#e6f4ea; border-left:5px solid #38a169; color:#1c4532; }
.todo-critico { border-left:4px solid #FF441F; padding:12px 16px;
    border-radius:0 8px 8px 0; background:#fff5f5; margin-bottom:8px;
    color:#822727 !important; }
.todo-riesgo  { border-left:4px solid #d69e2e; padding:12px 16px;
    border-radius:0 8px 8px 0; background:#fffbeb; margin-bottom:8px;
    color:#744210 !important; }
.todo-revisar { border-left:4px solid #718096; padding:12px 16px;
    border-radius:0 8px 8px 0; background:#f7fafc; margin-bottom:8px;
    color:#2d3748 !important; }

.agent-log { background:#1a1a1a; color:#FF441F; font-family:monospace;
    font-size:13px; padding:16px; border-radius:8px;
    max-height:180px; overflow-y:auto; border:1px solid #FF441F44; }

.stButton > button[kind="primary"] {
    background:#FF441F!important; border:none!important; font-weight:700!important; }
.stButton > button[kind="primary"]:hover { background:#e63a18!important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers UI ───────────────────────────────────────────────────────────────

EMOJI_PATTERN = re.compile("\U0001F7E2|\U0001F7E1|\U0001F534")


@st.cache_data(show_spinner="Cargando dataset y clasificando restaurantes...")
def load_and_predict(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df['semaforo_riesgo'] = (
        df['semaforo_riesgo']
        .apply(lambda x: EMOJI_PATTERN.sub("", str(x)))
        .str.strip().str.upper()
    )
    with open("./Notebooks/models/decision_tree.pkl", "rb") as f:
        clf = pickle.load(f)
    with open("./Notebooks/models/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    df_feat = build_features(df)
    X = df_feat[FEATURE_SET].fillna(df_feat[FEATURE_SET].median())
    df_feat['nivel_riesgo']     = le.inverse_transform(clf.predict(X))
    df_feat['prob_critico']     = clf.predict_proba(X)[:, le.transform(['CRÍTICO'])[0]]
    df_feat['confianza_modelo'] = clf.predict_proba(X).max(axis=1)
    return df_feat


def parse_todo_items(raw: str) -> list[dict]:
    """Parsea output del LLM en items de to-do."""
    items = []
    if not raw:
        return items
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        nivel = None
        if line.upper().startswith("CRÍTICO:") or line.upper().startswith("CRITICO:"):
            nivel = "CRÍTICO"
            line = line.split(":", 1)[1].strip()
        elif line.upper().startswith("EN RIESGO:"):
            nivel = "EN RIESGO"
            line = line.split(":", 1)[1].strip()
        elif line.upper().startswith("REVISAR:"):
            nivel = "REVISAR"
            line = line.split(":", 1)[1].strip()
        else:
            continue
        parts = [p.strip() for p in line.split("|")]
        items.append({
            "nivel":    nivel,
            "nombre":   parts[0] if len(parts) > 0 else "—",
            "problema": parts[1] if len(parts) > 1 else "—",
            "accion":   parts[2] if len(parts) > 2 else "Revisar con el restaurante",
        })
    orden = {"CRÍTICO": 0, "EN RIESGO": 1, "REVISAR": 2}
    items.sort(key=lambda x: orden.get(x["nivel"], 3))
    return items


# ─── Session state ────────────────────────────────────────────────────────────
if "raw_output"    not in st.session_state: st.session_state.raw_output    = None
if "kam_analizado" not in st.session_state: st.session_state.kam_analizado = None
if "todo_checks"   not in st.session_state: st.session_state.todo_checks   = {}


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    data_path = st.text_input(
        "Ruta del dataset",
        value="data/Rappi_AI_Builder_Challenge_Dataset.xlsx"
    )
    api_key = st.text_input("Anthropic API Key", value=os.environ.get("ANTHROPIC_API_KEY", ""),type="password", placeholder="sk-ant-...")
    st.divider()
    st.caption("Agente 1 · Árbol de Decisión")
    st.caption("Agente 2 · Claude + Tool Use")


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rappi-header">
    <h1>🛵 Rappi · KAM Agent</h1>
    <p>Early Warning System — detección proactiva de restaurantes en riesgo</p>
</div>
""", unsafe_allow_html=True)


# ─── Carga de datos ───────────────────────────────────────────────────────────
if not data_path:
    st.info("Ingresa la ruta del dataset en el sidebar.")
    st.stop()

try:
    df = load_and_predict(data_path)
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()


# ─── Selector de KAM ─────────────────────────────────────────────────────────
kams = sorted(df['kam_asignado'].unique().tolist())

col_select, col_btn = st.columns([3, 1])
with col_select:
    kam_sel = st.selectbox(
        "KAM",
        options=["— Selecciona un KAM —"] + kams,
        label_visibility="collapsed"
    )
with col_btn:
    analizar = st.button("🔍 Analizar", type="primary",
                         use_container_width=True,
                         disabled=(kam_sel == "— Selecciona un KAM —"))

# Si cambia el KAM, resetear análisis previo
if kam_sel != st.session_state.kam_analizado:
    st.session_state.raw_output    = None
    st.session_state.todo_checks   = {}
    st.session_state.kam_analizado = kam_sel

# Nada más que mostrar si no hay KAM seleccionado
if kam_sel == "— Selecciona un KAM —":
    st.markdown("### Selecciona un KAM para ver su portfolio.")
    st.stop()


# ─── Info del KAM seleccionado ────────────────────────────────────────────────
st.divider()
df_kam = df[df['kam_asignado'] == kam_sel]

n_critico = (df_kam['nivel_riesgo'] == 'CRÍTICO').sum()
n_riesgo  = (df_kam['nivel_riesgo'] == 'EN RIESGO').sum()
n_estable = (df_kam['nivel_riesgo'] == 'ESTABLE').sum()

col_a, col_b = st.columns([1, 2])

with col_a:
    st.markdown(f"**Portfolio · {kam_sel}**")
    st.markdown(f'<div class="metric-card card-critico">🔴 Críticos: {n_critico}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card card-riesgo">🟡 En Riesgo: {n_riesgo}</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card card-estable">🟢 Estables: {n_estable}</div>',
                unsafe_allow_html=True)

with col_b:
    st.markdown("**Por vertical**")
    for vertical in sorted(df_kam['vertical'].unique()):
        dv = df_kam[df_kam['vertical'] == vertical]
        vc = (dv['nivel_riesgo'] == 'CRÍTICO').sum()
        vr = (dv['nivel_riesgo'] == 'EN RIESGO').sum()
        ve = (dv['nivel_riesgo'] == 'ESTABLE').sum()
        st.markdown(
            f"**{vertical}** — "
            f"🔴 {vc} · 🟡 {vr} · 🟢 {ve}"
        )

    # Restaurantes críticos del KAM
    criticos_df = df_kam[df_kam['nivel_riesgo'] == 'CRÍTICO'][
        ['nombre', 'ciudad', 'vertical', 'tasa_cancelacion_pct', 'var_ordenes_pct']
    ].sort_values('tasa_cancelacion_pct', ascending=False)

    if not criticos_df.empty:
        st.markdown("**Críticos**")
        st.dataframe(criticos_df.rename(columns={
            'nombre': 'Restaurante', 'ciudad': 'Ciudad', 'vertical': 'Vertical',
            'tasa_cancelacion_pct': 'Cancelaciones %', 'var_ordenes_pct': 'Var. Volumen %'
        }), use_container_width=True, hide_index=True)


# ─── Agente en vivo ───────────────────────────────────────────────────────────
st.divider()

if analizar:
    if not api_key:
        st.error("Ingresa tu Anthropic API Key en el sidebar.")
        st.stop()

    df_alertas = df_kam[df_kam['nivel_riesgo'] != 'ESTABLE']
    if df_alertas.empty:
        st.success(f"✅ {kam_sel} no tiene restaurantes en alerta.")
        st.stop()

    st.markdown("**🤖 Agente procesando...**")
    log_placeholder = st.empty()
    log_lines = []

    def update_log(line):
        log_lines.append(line)
        log_placeholder.markdown(
            '<div class="agent-log">' + "<br>".join(log_lines[-10:]) + '</div>',
            unsafe_allow_html=True
        )

    # Loop del agente
    client = anthropic.Anthropic(api_key=api_key)
    notifications = {}

    prompt = f"""Eres un agente de alertas para KAMs de Rappi.
Genera una lista de tareas priorizadas para el KAM: {kam_sel}

Pasos:
1. Llama get_kam_portfolio('{kam_sel}')
2. Llama flag_border_cases('{kam_sel}')
3. Llama generate_kam_alert con toda la información"""

    messages = [{"role": "user", "content": prompt}]
    update_log(f"🤖 Iniciando análisis de {kam_sel}...")

    max_iter = 10
    for _ in range(max_iter):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            update_log("✅ Análisis completo")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                update_log(f"  → {block.name}({block.input.get('kam_name', '')})")

                if block.name == "get_kam_portfolio":
                    res = get_kam_portfolio(block.input["kam_name"], df)

                elif block.name == "flag_border_cases":
                    res = flag_border_cases(block.input["kam_name"], df)

                elif block.name == "generate_kam_alert":
                    from agent.alerter import generate_kam_alert_llm
                    update_log("  📝 Generando notificación...")
                    res = generate_kam_alert_llm(
                        block.input["kam_name"],
                        block.input["resumen_criticos"],
                        block.input["resumen_riesgo"],
                        block.input["casos_frontera"],
                        api_key
                    )
                    st.session_state.raw_output = res
                    update_log("  ✅ Listo")
                else:
                    res = f"Tool desconocida: {block.name}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": res
                })

            messages.append({"role": "user", "content": tool_results})


# ─── To-Do List ───────────────────────────────────────────────────────────────
if st.session_state.raw_output and st.session_state.kam_analizado == kam_sel:
    st.markdown(f"### ✅ To-Do — {kam_sel}")
    st.caption("Ordenado por severidad · Marca las tareas completadas")

    todo_items = parse_todo_items(st.session_state.raw_output)

    if not todo_items:
        st.warning("El agente no devolvió items en el formato esperado.")
        st.code(st.session_state.raw_output)
    else:
        completados = 0
        for i, item in enumerate(todo_items):
            key = f"check_{kam_sel}_{i}"

            # Inicializar checkbox en session_state si no existe
            if key not in st.session_state.todo_checks:
                st.session_state.todo_checks[key] = False

            col_check, col_content = st.columns([0.5, 9.5])

            with col_check:
                checked = st.checkbox(" ", key=key, label_visibility="collapsed")
                st.session_state.todo_checks[key] = checked

            with col_content:
                if item["nivel"] == "CRÍTICO":
                    css = "todo-critico"
                    badge = "🔴 CRÍTICO"
                elif item["nivel"] == "EN RIESGO":
                    css = "todo-riesgo"
                    badge = "🟡 EN RIESGO"
                else:
                    css = "todo-revisar"
                    badge = "⚠️ REVISAR"

                done_class = "todo-done" if checked else ""
                st.markdown(
                    f'<div class="{css} {done_class}">'
                    f'<span style="color:#333333">'
                    f'<strong>{badge} · {item["nombre"]}</strong><br>'
                    f'<em>{item["problema"]}</em><br>'
                    f'→ {item["accion"]}'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True

                )

            if checked:
                completados += 1

        st.caption(f"Progreso: {completados}/{len(todo_items)} tareas completadas")