import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ──────────────────────────────
# Protection functions
# ──────────────────────────────
def thermal_trip_time(I, I_th, tau, A2=0, K=1.0, I2=0):
    I_eq = np.sqrt(I*2 + K * I2*2)
    if I_eq <= I_th:
        return np.inf
    try:
        t = -tau * np.log(1 - (I_th/I_eq)**2 * (1 - A2))
        return max(t, 0.01)
    except ValueError:
        return np.inf

def idmt_trip_time(I, I_pickup, tms, curve="NI"):
    M = I / I_pickup
    if M <= 1:
        return np.inf
    if curve == "NI":
        return tms * 0.14 / (M**0.02 - 1)
    elif curve == "VI":
        return tms * 13.5 / (M - 1)
    elif curve == "EI":
        return tms * 80 / (M**2 - 1)
    return np.inf

def motor_start_curve(I_f, I_lr, t_acc, V_pct=100):
    I_lr_adj = I_lr * V_pct / 100
    times = np.linspace(0, t_acc, 200)
    slip = np.maximum(0.01, 1 - times / t_acc)
    currents = I_f + (I_lr_adj - I_f) * slip
    return times, currents

# ──────────────────────────────
# Streamlit UI
# ──────────────────────────────
st.title("Induction Motor Protection Curves")

st.sidebar.header("Motor Parameters")
kw = st.sidebar.number_input("Motor rating (kW)", 1, 5000, 500)
V = st.sidebar.number_input("Voltage (V)", 100, 15000, 3300)
I_f = st.sidebar.number_input("Full load current (A)", 1.0, 2000.0, 100.0)

st.sidebar.header("Motor Starting")
I_lr_mult = st.sidebar.slider("Locked rotor current (×FLC)", 3.0, 10.0, 6.0)
t_acc = st.sidebar.slider("Acceleration time (s)", 1, 60, 10)
V_start = st.sidebar.slider("Start voltage (% of rated)", 50, 100, 100)

st.sidebar.header("Thermal Model")
Ith_mult = st.sidebar.slider("Thermal pickup (×FLC)", 1.0, 8.0, 1.2)
tau = st.sidebar.slider("Heating time constant τ (s)", 10, 600, 120)
A2_hot = st.sidebar.slider("Hot condition (A2)", 0.0, 1.0, 0.5)
K_nps = st.sidebar.slider("NPS weighting factor K", 0.0, 5.0, 2.0)
I2_unbal_pct = st.sidebar.slider("Negative-sequence unbalance (%FLC)", 0.0, 50.0, 10.0)

st.sidebar.header("Overcurrent Protection")
I_inst_mult = st.sidebar.slider("Instantaneous OC (×FLC)", 1.0, 20.0, 8.0)
I_dt_mult = st.sidebar.slider("Definite-time OC (×FLC)", 1.0, 10.0, 2.0)
t_dt = st.sidebar.slider("Definite-time delay (s)", 0.1, 10.0, 1.0)

st.sidebar.header("IDMT OC Settings")
I_pickup_mult = st.sidebar.slider("IDMT pickup (×FLC)", 1.0, 5.0, 1.2)
tms = st.sidebar.slider("TMS", 0.05, 1.0, 0.1)
curve_type = st.sidebar.selectbox("Curve type", ["NI", "VI", "EI"])

st.sidebar.header("Earth Fault (EF) Protection")
I_ef_mult = st.sidebar.slider("EF pickup (×FLC)", 0.05, 1.0, 0.2)
t_ef = st.sidebar.slider("EF delay (s)", 0.05, 10.0, 0.5)

st.sidebar.header("Negative Phase Sequence (NPS)")
I2_pickup_pct = st.sidebar.slider("NPS pickup (%FLC)", 1.0, 50.0, 10.0)
t_nps = st.sidebar.slider("NPS delay (s)", 0.05, 10.0, 0.5)

st.sidebar.header("Locked Rotor Protection")
LR_prot_mult = st.sidebar.slider("Locked Rotor pickup (×FLC)", 3.0, 10.0, 6.0)
LR_time = st.sidebar.slider("Locked Rotor max time (s)", 1.0, 60.0, 10.0)

# ──────────────────────────────
# Calculations
# ──────────────────────────────
I_lr = I_lr_mult * I_f
I_th = Ith_mult * I_f
I2 = I2_unbal_pct / 100 * I_f
I_pickup = I_pickup_mult * I_f
I_ef = I_ef_mult * I_f
I2_pickup = I2_pickup_pct / 100 * I_f
I_LR_prot = LR_prot_mult * I_f

currents = np.logspace(np.log10(I_f), np.log10(20 * I_f), 200)

thermal_cold = [thermal_trip_time(i, I_th, tau, 0.0, K_nps, I2) for i in currents]
thermal_hot  = [thermal_trip_time(i, I_th, tau, A2_hot, K_nps, I2) for i in currents]
t_start, I_start = motor_start_curve(I_f, I_lr, t_acc, V_start)
idmt_times = [idmt_trip_time(i, I_pickup, tms, curve=curve_type) for i in currents]

# ──────────────────────────────
# Variable visibility control
# ──────────────────────────────
ALL_SERIES = [
    "Thermal limit (Cold)",
    "Thermal limit (Hot)",
    "Motor start",
    f"IDMT ({curve_type})",
    "Instantaneous OC (vertical)",
    "Definite-time OC (vertical)",
    "Definite-time OC (horizontal)",
    "Earth Fault (vertical)",
    "Earth Fault (horizontal)",
    "NPS (vertical)",
    "NPS (horizontal)",
    "Locked Rotor Pickup (vertical)",
    "Locked Rotor Max Time (horizontal)",
]

st.subheader("Display options")
colA, colB = st.columns([3,1])
with colA:
    selected = st.multiselect(
        "Select variables to show on the chart",
        options=ALL_SERIES,
        default=[
            "Thermal limit (Cold)",
            "Thermal limit (Hot)",
            "Motor start",
            f"IDMT ({curve_type})",
            "Instantaneous OC (vertical)",
            "Definite-time OC (horizontal)",
        ],
    )
with colB:
    sel_all = st.checkbox("Select all", value=False)
    if sel_all:
        selected = ALL_SERIES[:]
    sel_none = st.checkbox("Select none", value=False)
    if sel_none:
        selected = []

# ──────────────────────────────
# Plotting
# ──────────────────────────────
def _finite_mask(y):
    a = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(a) & (a > 0)  # log-axis safe
    return ok

fig = go.Figure()

# Curves (add only if selected)
if "Thermal limit (Cold)" in selected:
    ok = _finite_mask(thermal_cold)
    fig.add_trace(go.Scatter(
        x=currents[ok], y=np.asarray(thermal_cold)[ok],
        mode="lines", name="Thermal limit (Cold)",
        line=dict(color="blue", width=2)
    ))

if "Thermal limit (Hot)" in selected:
    ok = _finite_mask(thermal_hot)
    fig.add_trace(go.Scatter(
        x=currents[ok], y=np.asarray(thermal_hot)[ok],
        mode="lines", name="Thermal limit (Hot)",
        line=dict(color="orange", width=2)
    ))

if "Motor start" in selected:
    okx = _finite_mask(I_start)
    oky = _finite_mask(t_start)
    ok  = okx & oky
    fig.add_trace(go.Scatter(
        x=np.asarray(I_start)[ok], y=np.asarray(t_start)[ok],
        mode="lines", name=f"Motor start ({V_start}%)",
        line=dict(color="black", width=2, dash="dash")
    ))

if f"IDMT ({curve_type})" in selected:
    ok = _finite_mask(idmt_times)
    fig.add_trace(go.Scatter(
        x=currents[ok], y=np.asarray(idmt_times)[ok],
        mode="lines", name=f"IDMT ({curve_type})",
        line=dict(color="purple", width=2, dash="dot")
    ))

# Vertical lines (as shapes so they span the plot & don't clutter legend)
def _vline(x, color):
    fig.add_shape(type="line", x0=x, x1=x, xref="x", yref="paper", y0=0, y1=1,
                  line=dict(color=color, width=2, dash="dash"))

def _hline(y, color):
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=y, y1=y, yref="y",
                  line=dict(color=color, width=2, dash="dash"))

if "Instantaneous OC (vertical)" in selected:        _vline(I_inst_mult * I_f, "red")
if "Definite-time OC (vertical)" in selected:        _vline(I_dt_mult * I_f, "green")
if "Definite-time OC (horizontal)" in selected:      _hline(t_dt, "green")
if "Earth Fault (vertical)" in selected:             _vline(I_ef, "magenta")
if "Earth Fault (horizontal)" in selected:           _hline(t_ef, "magenta")
if "NPS (vertical)" in selected:                     _vline(I2_pickup, "cyan")
if "NPS (horizontal)" in selected:                   _hline(t_nps, "cyan")
if "Locked Rotor Pickup (vertical)" in selected:     _vline(I_LR_prot, "brown")
if "Locked Rotor Max Time (horizontal)" in selected: _hline(LR_time, "brown")

# Axes, titles, ETAP-green grid (lighter, semi-transparent), NO legend on chart
ETAP_GREEN_RGBA_MAJOR = "rgba(0,166,81,0.4)"  # 40% opacity
ETAP_GREEN_RGBA_MINOR = "rgba(0,166,81,0.25)" # 25% opacity

fig.update_layout(
    title=f"Motor Protection Curves<br><sup>Motor: {kw} kW, {V} V</sup>",
    xaxis_title="Current (A)",
    yaxis_title="Time (s)",
    template="simple_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=60, b=40),
    showlegend=False,   # ✅ legend hidden on chart
)

# Axis labels/ticks in black
fig.update_xaxes(title_font=dict(color="black"), tickfont=dict(color="black"))
fig.update_yaxes(title_font=dict(color="black"), tickfont=dict(color="black"))

# Log scales + faint ETAP-green grid
fig.update_xaxes(
    type="log",
    showgrid=True, gridcolor=ETAP_GREEN_RGBA_MAJOR, gridwidth=0.4, griddash="dash",
    minor=dict(showgrid=True, gridcolor=ETAP_GREEN_RGBA_MINOR, gridwidth=0.25, griddash="dot")
)
fig.update_yaxes(
    type="log",
    showgrid=True, gridcolor=ETAP_GREEN_RGBA_MAJOR, gridwidth=0.4, griddash="dash",
    minor=dict(showgrid=True, gridcolor=ETAP_GREEN_RGBA_MINOR, gridwidth=0.25, griddash="dot")
)

# Make it really interactive (polished UX)
fig.update_layout(
    hovermode="x unified",  # single hover label across traces
    dragmode="zoom",        # default tool when you click-drag
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        # Navigation & zooming
        "scrollZoom": True,              # wheel zoom
        "doubleClick": "reset",          # double-click resets axes
        "responsive": True,

        # Modebar: show, clean, and useful tools only
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "autoScale2d", "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian",
            "lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "resetScale2d"
        ],
        "modeBarButtonsToAdd": [
            "zoom2d",          # box zoom
            "pan2d",           # pan
            "toImage",         # quick export
        ],

        # Export settings
        "toImageButtonOptions": {
            "format": "png",                 # png, svg, jpeg, webp
            "filename": "motor_protection_curves",
            "scale": 2,                      # 2x for crisper export
            "height": 600,
            "width": 1000,
        },
    },
)
# Legend panel BELOW the chart
# ──────────────────────────────
def swatch(color, style="solid"):
    # small HTML span as color swatch; style note appended in text
    return f'<span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:8px;border:1px solid #333"></span>'

legend_items = []

if "Thermal limit (Cold)" in selected:
    legend_items.append(swatch("blue") + "Thermal limit (Cold)")
if "Thermal limit (Hot)" in selected:
    legend_items.append(swatch("orange") + "Thermal limit (Hot)")
if "Motor start" in selected:
    legend_items.append(swatch("black") + f"Motor start ({V_start}%)  — dashed")
if f"IDMT ({curve_type})" in selected:
    legend_items.append(swatch("purple") + f"IDMT ({curve_type})  — dotted")

if "Instantaneous OC (vertical)" in selected:
    legend_items.append(swatch("red") + "Inst. OC (vertical)")
if "Definite-time OC (vertical)" in selected:
    legend_items.append(swatch("green") + "Definite-time OC (vertical)")
if "Definite-time OC (horizontal)" in selected:
    legend_items.append(swatch("green") + "Definite-time OC (horizontal)")
if "Earth Fault (vertical)" in selected:
    legend_items.append(swatch("magenta") + "Earth Fault (vertical)")
if "Earth Fault (horizontal)" in selected:
    legend_items.append(swatch("magenta") + "Earth Fault (horizontal)")
if "NPS (vertical)" in selected:
    legend_items.append(swatch("cyan") + "NPS (vertical)")
if "NPS (horizontal)" in selected:
    legend_items.append(swatch("cyan") + "NPS (horizontal)")
if "Locked Rotor Pickup (vertical)" in selected:
    legend_items.append(swatch("brown") + "Locked Rotor Pickup (vertical)")
if "Locked Rotor Max Time (horizontal)" in selected:
    legend_items.append(swatch("brown") + "Locked Rotor Max Time (horizontal)")

st.markdown("### Legend")
if legend_items:
    st.markdown(
        "<div style='line-height:1.8'>" + "<br>".join(legend_items) + "</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("No variables selected. Use the selector above to add curves/limits to the chart.")
