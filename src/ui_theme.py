"""Visual layer for the FraudLens dashboard: CSS, colours and Plotly defaults.

Separate from app.py so presentation can change without touching analysis.
Nothing here reads data or computes anything.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------- palette
TURQ = "#19B8B0"
TURQ_DK = "#087F8C"
TURQ_LT = "#A7E8E3"
NAVY = "#142B3B"
BG = "#F5F8FA"
CARD = "#FFFFFF"
INK = "#1F2937"
MUTED = "#64748B"
LINE = "#E2E8F0"

RED = "#DC2626"
AMBER = "#D97706"
GREEN = "#059669"

# Fraud and legitimate appear in this order everywhere, so the pairing is
# learned once: turquoise is fraud, slate is legitimate.
FRAUD = TURQ
LEGIT = "#94A3B8"

# Sequential turquoise ramp for ordered categories (deciles, hours).
RAMP = ["#CFF3F0", "#A7E8E3", "#6FD6CE", "#3FC4BA", "#19B8B0",
        "#13A098", "#0E8880", "#087F8C", "#06636E", "#044853"]

CATEGORICAL = [TURQ, NAVY, "#5EC9C2", "#3D5A73", TURQ_DK, "#8FA8BC",
               "#0E8880", MUTED]


def install_plotly_template():
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif",
                  size=13, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=CATEGORICAL,
        # Top margin leaves room for the title AND the horizontal legend
        # beneath it; at t=46 the two collide on any chart with a legend.
        margin=dict(l=10, r=10, t=78, b=10),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left",
                   y=0.97, yanchor="top"),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE,
                        font=dict(size=12, color=INK)),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED, size=12),
                   title=dict(font=dict(color=MUTED, size=12))),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
                   tickfont=dict(color=MUTED, size=12),
                   title=dict(font=dict(color=MUTED, size=12))),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left",
                    x=0, font=dict(size=12, color=MUTED), title=None),
    )
    pio.templates["fraudlens"] = t
    pio.templates.default = "fraudlens"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

html, body, [class*="st-"], .stMarkdown {{
    font-family:'Inter','Segoe UI',system-ui,sans-serif;
}}
.stApp {{ background:{BG}; }}
#MainMenu, footer, header {{ visibility:hidden; }}
/* Hide only the sidebar collapse control. Icon fonts are NOT hidden globally:
   Streamlit uses Material Symbols ligatures for expander arrows, and hiding
   the class leaves the raw ligature text visible. The font is imported above
   so the glyphs render. */
[data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarHeader"] {{ display:none !important; }}
.block-container {{ padding:1.5rem 2.2rem 3rem 2.2rem; max-width:1520px; }}

/* ------------------------------------------------------------- sidebar */
section[data-testid="stSidebar"] {{ background:{NAVY}; border:none; }}
section[data-testid="stSidebar"] .block-container {{ padding-top:1.1rem; }}
.fl-brand {{
    color:#fff; font-weight:700; font-size:1.12rem; letter-spacing:.2px;
    padding:.2rem .3rem 1.1rem .3rem;
    border-bottom:1px solid rgba(255,255,255,.10); margin-bottom:1rem;
}}
.fl-brand .dot {{ color:{TURQ}; margin-right:.45rem; }}
.fl-brand small {{
    display:block; font-weight:400; font-size:.7rem; color:#8FA8BC;
    letter-spacing:.3px; margin-top:.25rem;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    display:flex; align-items:center; padding:.55rem .75rem; margin-bottom:.15rem;
    border-radius:8px; cursor:pointer; font-size:.875rem; font-weight:500;
    color:#B6C6D4; transition:all .13s ease; border-left:3px solid transparent;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background:rgba(255,255,255,.055); color:#fff;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background:rgba(25,184,176,.14); color:#fff; font-weight:600;
    border-left:3px solid {TURQ};
}}
/* Hide the radio marker: the highlighted pill carries the selected state.
   Streamlit nests it as label > div > div:first-child (the <input> sits in a
   visually-clipped <span> before that wrapper). */
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child {{
    display:none !important;
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div {{
    gap:0 !important; padding-left:0 !important;
}}
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p {{
    margin:0; font-size:.875rem;
}}
.fl-side-foot {{ font-size:.68rem; color:#6E8296; line-height:1.65; }}

/* --------------------------------------------------------------- cards */
.fl-card {{
    background:{CARD}; border:1px solid {LINE}; border-radius:10px;
    padding:1.1rem 1.25rem; margin-bottom:.9rem;
    box-shadow:0 1px 2px rgba(16,24,40,.04);
}}
.fl-kpi {{
    background:{CARD}; border:1px solid {LINE}; border-radius:10px;
    padding:.95rem 1.1rem; height:100%;
    border-top:3px solid {TURQ}; box-shadow:0 1px 2px rgba(16,24,40,.04);
}}
.fl-kpi.neutral {{ border-top-color:{MUTED}; }}
.fl-kpi.warn    {{ border-top-color:{AMBER}; }}
.fl-kpi .lab {{
    font-size:.7rem; text-transform:uppercase; letter-spacing:.7px;
    color:{MUTED}; font-weight:600;
}}
.fl-kpi .val {{ font-size:1.7rem; font-weight:700; color:{INK}; line-height:1.2; margin:.3rem 0 .12rem 0; }}
.fl-kpi .sub {{ font-size:.75rem; color:{MUTED}; }}

.fl-h  {{ font-size:1.35rem; font-weight:700; color:{INK}; margin:.1rem 0 .12rem 0; }}
.fl-sub{{ font-size:.85rem; color:{MUTED}; margin-bottom:1.1rem; }}
.fl-sec{{
    font-size:.98rem; font-weight:700; color:{INK};
    margin:1.45rem 0 .55rem 0; padding-bottom:.35rem;
    border-bottom:1px solid {LINE};
}}

/* ------------------------------------------------------------- insight */
.fl-insight {{
    background:#F0FBFA; border:1px solid #CFF0ED; border-left:3px solid {TURQ};
    border-radius:8px; padding:.85rem 1.05rem; margin-bottom:.9rem;
    font-size:.92rem; color:{INK}; line-height:1.55;
}}
.fl-caveat {{
    background:#FFFBEB; border:1px solid #FDE9C8; border-left:3px solid {AMBER};
    border-radius:8px; padding:.85rem 1.05rem; margin-bottom:.9rem;
    font-size:.88rem; color:#78350F; line-height:1.6;
}}
.fl-note {{
    background:#F8FAFC; border:1px solid {LINE}; border-radius:8px;
    padding:.7rem .95rem; font-size:.8rem; color:{MUTED}; line-height:1.6;
}}

/* -------------------------------------------------------------- badges */
.badge {{ padding:.2rem .55rem; border-radius:5px; font-size:.72rem; font-weight:600; }}
.badge-hi {{ background:#FEE2E2; color:{RED}; }}
.badge-ot {{ background:#FEF3C7; color:{AMBER}; }}
.badge-ok {{ background:#D1FAE5; color:{GREEN}; }}
.badge-tq {{ background:#E0F7F5; color:{TURQ_DK}; }}
.badge-nu {{ background:#F1F5F9; color:{MUTED}; }}

/* -------------------------------------------------------------- tables */
.stDataFrame {{ border:1px solid {LINE}; border-radius:8px; }}
div[data-testid="stMetricValue"] {{ font-size:1.4rem; font-weight:700; color:{INK}; }}
div[data-testid="stMetricLabel"] {{ color:{MUTED}; font-weight:500; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.35rem; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{
    height:38px; padding:0 1rem; font-size:.875rem; font-weight:500; color:{MUTED};
}}
.stTabs [aria-selected="true"] {{ color:{TURQ_DK} !important; font-weight:650; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{TURQ}; }}
div[data-testid="stExpander"] details {{
    border:1px solid {LINE}; border-radius:8px; background:{CARD};
}}
div[data-testid="stExpander"] summary {{
    font-size:.85rem; font-weight:600; color:{TURQ_DK}; padding:.6rem .9rem;
}}
/* Streamlit draws the expander chevron as a Material Symbols ligature. If that
   font has not loaded the raw word shows as text, so the glyph is hidden and a
   CSS triangle drawn instead. The affordance is kept, the failure mode is not. */
div[data-testid="stExpander"] summary span[data-testid="stIconMaterial"] {{
    font-size:0 !important; width:0 !important; overflow:hidden !important;
}}
div[data-testid="stExpander"] summary::before {{
    content:""; display:inline-block; margin-right:.55rem;
    border-left:5px solid {TURQ}; border-top:4px solid transparent;
    border-bottom:4px solid transparent; transition:transform .15s ease;
}}
div[data-testid="stExpander"] details[open] summary::before {{
    transform:rotate(90deg);
}}
div[data-testid="stExpander"] summary:hover {{ color:{TURQ}; }}
.stButton>button, .stDownloadButton>button {{
    background:{TURQ}; color:#fff; border:none; border-radius:7px;
    font-weight:600; font-size:.85rem;
}}
.stButton>button:hover {{ background:{TURQ_DK}; color:#fff; }}




/* Only the file-picker button. Never style the uploaded-file delete button.
   Streamlit's upload icon can render as the literal word "upload" when the
   Material Symbols font is unavailable; cover that native label with one
   reliably centered visual label on this button only. */
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button[kind="secondary"] {{
    position: relative !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-width: 160px !important;
    min-height: 52px !important;
    padding: .65rem 1.8rem !important;
    background-color: {TURQ} !important;
    border: none !important;
    border-radius: 999px !important;
    cursor: pointer !important;
}}

/* Hide ONLY the original inner content of the browse button. */
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] > *,
[data-testid="stFileUploaderDropzone"] button[kind="secondary"] > * {{
    visibility: hidden !important;
}}

/* Add one visual label, centered independently of icon-font loading. */
[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::after,
[data-testid="stFileUploaderDropzone"] button[kind="secondary"]::after {{
    content: "Upload";
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    color: #FFFFFF;
    font: 600 .9rem 'Inter', 'Segoe UI', sans-serif;
    pointer-events: none;
}}

[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stFileUploaderDropzone"] button[kind="secondary"]:hover {{
    background-color: {TURQ_DK} !important;
}}

</style>
"""


def install():
    st.markdown(CSS, unsafe_allow_html=True)
    install_plotly_template()


def kpi(label, value, sub="", tone="turq"):
    cls = {"turq": "", "neutral": " neutral", "warn": " warn"}[tone]
    st.markdown(
        f'<div class="fl-kpi{cls}"><div class="lab">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def heading(title, sub=""):
    st.markdown(f'<div class="fl-h">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="fl-sub">{sub}</div>', unsafe_allow_html=True)


def section(t):
    st.markdown(f'<div class="fl-sec">{t}</div>', unsafe_allow_html=True)


def insight(t):
    st.markdown(f'<div class="fl-insight">{t}</div>', unsafe_allow_html=True)


def caveat(t):
    st.markdown(f'<div class="fl-caveat">{t}</div>', unsafe_allow_html=True)


def note(t):
    st.markdown(f'<div class="fl-note">{t}</div>', unsafe_allow_html=True)
