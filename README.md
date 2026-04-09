Hola!


Los KAMs de Rappi gestionan portfolios de 300 a 600 restaurantes cada uno.
El monitoreo actual es completamente reactivo: el KAM se entera cuando el daño ya ocurrió. Este sistema cambia eso
Sistema de alertas tempranas para Key Account Managers. 
Detecta restaurantes en riesgo antes de que los problemas impacten el rating, las ventas o la retención del usuario.

# Pipeline


DatasetExcel
     │
     ▼

     
AGENTE 1 — Clasificación ML (agent/features.py + agent/model.py)


  • Feature engineering:

  
    satisfaccion_index; Combina rating (60%) y NPS (40%). Correlación 0.887 — miden lo mismo
    cancel_vs_vertical: Z-score por vertical. Bebidas tiene 29% críticos vs 9% en Mercado
    ticket_vs_vertical: Ticket baja entre clases. Relativizado evita comparar verticales. No se esperan los mismos desmbolos entre verticales
    ordenes_vs_vertical: El cambio de volumen importa más que el absoluto.
    critico_rate_vertical: Tasa histórica de riesgo por vertical, contexto para el modelo.
  • Árbol de decisión → Crítico / En Riesgo / Estable

  
    Utilize las etiquetas propocionadas: En un EDA se descurbio que son linealmenmte separables
    Se utilizo un arbol para saber que variables del data set son las que lllevan a cada etiqueta.
    El arbol descubrio:
      ¿Cancelaciones > 10.1%?
    NO → ESTABLE (56% del portfolio)
    SÍ → ¿Cancelaciones > 20.4%?
            NO → ¿Volumen cayó vs su vertical?
                    SÍ → CRÍTICO
                    NO → EN RIESGO
            SÍ → ¿Ticket bajo vs su vertical?
                    SÍ → CRÍTICO
                    NO → EN RIESGO
  
  • Probabilidad por clase → identifica casos en frontera
     │
     ▼

     
AGENTE 2 — Notificaciones (agent/alerter.py)


  • Tool: get_kam_portfolio    → filtra restaurantes por KAM
  • Tool: flag_border_cases    → identifica casos ambiguos
  • Tool: generate_kam_alert   → Claude genera notificación accionable
     │
     ▼

     
Dashboard Streamlit (dashboard.py)


  • Dropdown por KAM
  • To-do list ordenada por severidad con checkboxes
  • Métricas por nivel y por vertical

Instalacion local
  # Clonar el repo
git clone https://github.com/Howl-coder/kam-agent.git
cd kam-agent

# Instalar dependencias
pip install -r requirements.txt

# Configurar API key
export ANTHROPIC_API_KEY="sk-ant-..."   # Linux/Mac


$env:ANTHROPIC_API_KEY="sk-ant-..."     # Windows PowerShell

# Correr el dashboard
streamlit run dashboard.py
