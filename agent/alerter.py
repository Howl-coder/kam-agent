"""
alerter.py - Agente de notificaciones por KAM
Rappi KAM Early Warning System

Usa tool use nativo de la API de Anthropic.
Claude decide qué tools llamar y en qué orden para generar
la notificación más relevante para cada KAM.

Tools disponibles:
    get_kam_portfolio    → restaurantes de un KAM con su clasificación
    flag_border_cases    → casos en frontera que requieren revisión humana
    generate_kam_alert   → genera notificación accionable en lenguaje natural

Loop del agente:
    1. Claude recibe el prompt con los KAMs disponibles
    2. Claude llama get_kam_portfolio para entender el contexto
    3. Claude llama flag_border_cases para identificar casos ambiguos
    4. Claude genera la notificación final con todo el contexto
    5. Repite por cada KAM con alertas
"""

import json
import anthropic
import pandas as pd
from typing import Optional


TOOLS = [
    {
        "name": "get_kam_portfolio",
        "description": (
            "Retorna los restaurantes críticos y en riesgo de un KAM específico. "
            "Incluye métricas clave: cancelaciones, volumen, ticket y vertical."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kam_name": {"type": "string", "description": "Nombre exacto del KAM"}
            },
            "required": ["kam_name"]
        }
    },
    {
        "name": "flag_border_cases",
        "description": (
            "Identifica restaurantes en la frontera de clasificación "
            "que requieren revisión humana del KAM. "
            "Son casos donde el modelo tiene baja confianza."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kam_name": {"type": "string", "description": "Nombre exacto del KAM"}
            },
            "required": ["kam_name"]
        }
    },
    {
        "name": "generate_kam_alert",
        "description": (
            "Genera y guarda la notificación final accionable para el KAM. "
            "Llama esta tool DESPUÉS de get_kam_portfolio y flag_border_cases."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "kam_name": {"type": "string"},
                "resumen_criticos": {"type": "string"},
                "resumen_riesgo": {"type": "string"},
                "casos_frontera": {"type": "string"}
            },
            "required": ["kam_name", "resumen_criticos", "resumen_riesgo", "casos_frontera"]
        }
    }
]


def get_kam_portfolio(kam_name: str, df: pd.DataFrame) -> str:
    subset = df[
        (df['kam_asignado'] == kam_name) &
        (df['nivel_riesgo'] != 'ESTABLE')
    ].sort_values('prob_critico', ascending=False)

    if subset.empty:
        return f"KAM {kam_name} no tiene restaurantes en alerta."

    criticos  = subset[subset['nivel_riesgo'] == 'CRÍTICO']
    en_riesgo = subset[subset['nivel_riesgo'] == 'EN RIESGO']

    lines = [
        f"Portfolio de {kam_name}:",
        f"Total en alerta: {len(subset)} ({len(criticos)} críticos, {len(en_riesgo)} en riesgo)",
        ""
    ]

    if not criticos.empty:
        lines.append("CRÍTICOS:")
        for _, r in criticos.iterrows():
            lines.append(
                f"  • {r['nombre']} | {r['vertical']} | {r['ciudad']}\n"
                f"    Cancelaciones: {r['tasa_cancelacion_pct']:.1f}% | "
                f"Volumen: {r['var_ordenes_pct']:.1f}% | "
                f"Ticket: ${r['valor_ticket_prom_mxn']:.0f} MXN"
            )

    if not en_riesgo.empty:
        lines.append("\nEN RIESGO:")
        for _, r in en_riesgo.iterrows():
            lines.append(
                f"  • {r['nombre']} | {r['vertical']} | {r['ciudad']}\n"
                f"    Cancelaciones: {r['tasa_cancelacion_pct']:.1f}% | "
                f"Volumen: {r['var_ordenes_pct']:.1f}%"
            )

    return "\n".join(lines)


def flag_border_cases(kam_name: str, df: pd.DataFrame) -> str:
    subset = df[
        (df['kam_asignado'] == kam_name) &
        (df['nivel_riesgo'] != 'ESTABLE') &
        (df['confianza_modelo'] < 0.80)
    ]

    if subset.empty:
        return f"No hay casos en frontera para {kam_name}."

    lines = [f"Casos que requieren revisión humana ({kam_name}):"]
    for _, r in subset.iterrows():
        lines.append(
            f"  • {r['nombre']} | Clasificado como: {r['nivel_riesgo']} | "
            f"Confianza: {r['confianza_modelo']:.0%}\n"
            f"    Cancelaciones: {r['tasa_cancelacion_pct']:.1f}% | "
            f"Volumen: {r['var_ordenes_pct']:.1f}%"
        )

    return "\n".join(lines)


def generate_kam_alert(
    kam_name: str,
    resumen_criticos: str,
    resumen_riesgo: str,
    casos_frontera: str,
    client: anthropic.Anthropic
) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system="""Responde ÚNICAMENTE con líneas en este formato exacto, sin ningún texto adicional:

    CRÍTICO: [nombre restaurante] | [problema específico con número] | [acción concreta + por qué es urgente]
    EN RIESGO: [nombre restaurante] | [problema específico con número] | [acción concreta + qué señal vigilar]
    REVISAR: [nombre restaurante] | [por qué es ambiguo] | [qué evaluar para decidir]

    REGLAS:
    - El problema DEBE incluir el número (ej: "41% cancelaciones 3 semanas seguidas")
    - La acción DEBE explicar el por qué (ej: "Llamar hoy — cancelaciones altas indican problema operativo antes de que caiga el rating")
    - Máximo 20 palabras por sección
    - PROHIBIDO: texto introductorio, bullets, markdown como ** o ##, cualquier línea que no empiece con CRÍTICO:, EN RIESGO: o REVISAR:

    Ejemplo correcto:
    CRÍTICO: Pizzería La Abuela | 41% cancelaciones — doble del umbral normal | Llamar hoy para revisar cocina — a este ritmo el rating cae en 2 semanas
    EN RIESGO: Tortas Clásico | volumen bajó 15% vs semana anterior | Check-in esta semana — aún reversible antes de que impacte rating
    REVISAR: Café El Norte | cancelaciones en 20% — justo en frontera | Evaluar si es pico temporal o tendencia antes de escalar
    
    PROHIBIDO agrupar así:
    EN RIESGO: 3 restaurantes México | cancelaciones 10-18% | Revisar esta semana
    
    CORRECTO — uno por uno:
    EN RIESGO: Pizzería 100% Natural | 18.8% cancelaciones subiendo | Llamar esta semana — si no baja en 7 días escala a crítico
    EN RIESGO: El Asadero La Familia | 16.4% cancelaciones | Check-in telefónico — verificar si es problema de staffing en horas pico
    EN RIESGO: Burger La Familia | 11.4% cancelaciones | Monitorear — está en umbral, una semana más así y requiere intervención
    """,

        messages=[{
            "role": "user",
            "content": f"""KAM: {kam_name}

        {resumen_criticos}

        {resumen_riesgo}

        {casos_frontera}

        IMPORTANTE: Una línea por restaurante individual. 
        NO agrupes restaurantes por ciudad ni por nivel.
        Cada restaurante debe tener su propia línea CRÍTICO:, EN RIESGO: o REVISAR:"""
            }]
    )
    return response.content[0].text

def execute_tool(tool_name, tool_input, df, client, notifications):
    if tool_name == "get_kam_portfolio":
        return get_kam_portfolio(tool_input["kam_name"], df)

    elif tool_name == "flag_border_cases":
        return flag_border_cases(tool_input["kam_name"], df)

    elif tool_name == "generate_kam_alert":
        alert = generate_kam_alert(
            kam_name=tool_input["kam_name"],
            resumen_criticos=tool_input["resumen_criticos"],
            resumen_riesgo=tool_input["resumen_riesgo"],
            casos_frontera=tool_input["casos_frontera"],
            client=client
        )
        notifications[tool_input["kam_name"]] = alert
        return f"Notificación generada para {tool_input['kam_name']}."

    return f"Tool desconocida: {tool_name}"


def run_agent(df: pd.DataFrame, api_key: str, verbose: bool = True) -> dict:
    """
    Loop principal del agente.
    Claude decide qué tools llamar y en qué orden para cada KAM.
    """
    client = anthropic.Anthropic(api_key=api_key)
    notifications = {}

    if 'confianza_label' not in df.columns:
        df['confianza_label'] = df['confianza_modelo'].apply(
            lambda p: 'Alta' if p >= 0.90 else 'Media' if p >= 0.70 else 'Revisar'
        )

    kams = (
        df[df['nivel_riesgo'] != 'ESTABLE']['kam_asignado']
        .unique().tolist()
    )

    if not kams:
        print("✅ No hay restaurantes en alerta.")
        return {}

    prompt = f"""Eres un agente de alertas tempranas para KAMs de Rappi.

Tu tarea es generar una notificación accionable para cada KAM con restaurantes en alerta.

KAMs que requieren atención hoy: {', '.join(kams)}

Para cada KAM debes:
1. Llamar get_kam_portfolio para entender su situación
2. Llamar flag_border_cases para identificar casos ambiguos
3. Llamar generate_kam_alert con toda la información recopilada

Procesa todos los KAMs de la lista."""

    messages = [{"role": "user", "content": prompt}]

    if verbose:
        print(f"\n🤖 Agente iniciado — {len(kams)} KAMs con alertas")
        print("=" * 55)

    max_iterations = len(kams) * 6
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            if verbose:
                print(f"\n✅ Agente terminó — {len(notifications)} notificaciones generadas")
            break

        if response.stop_reason == "tool_use":
            tool_results = []

            for block in response.content:
                if block.type != "tool_use":
                    continue

                if verbose:
                    print(f"  → {block.name}({block.input.get('kam_name', '')})")

                result = execute_tool(block.name, block.input, df, client, notifications)

                if verbose and block.name == "generate_kam_alert":
                    print(f"  ✅ Notificación lista: {block.input['kam_name']}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

            messages.append({"role": "user", "content": tool_results})

    return notifications


def run_alerter(df: pd.DataFrame, api_key: str, verbose: bool = True) -> dict:
    """
    Entry point del Agente 2.

    Uso:
        from agent.alerter import run_alerter
        notifications = run_alerter(df_pred, api_key="sk-ant-...")
        print(notifications["María López"])
    """
    return run_agent(df, api_key=api_key, verbose=verbose)


def generate_kam_alert_llm(
    kam_name: str,
    resumen_criticos: str,
    resumen_riesgo: str,
    casos_frontera: str,
    api_key: str
) -> str:
    """Wrapper para el dashboard — acepta api_key en lugar de client."""
    import anthropic as _anthropic
    client = _anthropic.Anthropic(api_key=api_key)
    return generate_kam_alert(
        kam_name=kam_name,
        resumen_criticos=resumen_criticos,
        resumen_riesgo=resumen_riesgo,
        casos_frontera=casos_frontera,
        client=client
    )