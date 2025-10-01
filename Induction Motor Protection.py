import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# ──────────────────────────────
# Protection functions
# ──────────────────────────────
def thermal_trip_time(I, I_th, tau, A2=0, K=1.0, I2=0):
    I_eq = np.sqrt(I**2 + K * I2**2)
    if I_eq <= I_th:
        return np.inf
    try:
        t = -tau * np.log(1 - (I_th/I_eq)**2 * (1 - A2))
        return max(t, 0.01)
    except ValueError:
        return np.inf

def top_trip_time(I, Ip, TDM):
    M = I / Ip
    if M <= 1:
        return np.inf
    return (87.4 * TDM) / (M**2 - 1)

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
st.title("Induction Motor Thermal Model and Protection Curves")

# NEW: Equipment Info (data entry tab)
with st.sidebar.expander("Equipment Info", expanded=True):
    equipment_name = st.text_input("Equipment name", value="Motor-1")
    equipment_tag  = st.text_input("Tag / ID (optional)", value="")

# Sidebar dropdown (expander) groups
with st.sidebar.expander("Motor Parameters", expanded=True):
    kw = st.number_input("Motor rating (kW)", 1, 5000, 500)
    V  = st.number_input("Voltage (V)", 100, 15000, 3300)
    I_f = st.number_input("Full load current (A)", 1.0, 2000.0, 100.0)

with st.sidebar.expander("Motor Starting", expanded=False):
    I_lr_mult = st.slider("Locked rotor current (×FLC)", 3.0, 10.0, 6.0)
    t_acc     = st.slider("Acceleration time (s)", 1, 60, 10)
    V_start   = st.slider("Start voltage (% of rated)", 50, 100, 100)

with st.sidebar.expander("Thermal Model", expanded=False):
    Ith_mult    = st.slider("Thermal pickup (×FLC)", 1.0, 8.0, 1.2)
    tau         = st.slider("Heating time constant τ (s)", 0, 3000, 120)
    A2_hot      = st.slider("Hot condition (A2)", 0.0, 1.0, 0.5)
    K_nps       = st.slider("NPS weighting factor K", 0.0, 15.0, 2.0)
    I2_unbal_pct= st.slider("Negative-sequence unbalance (%FLC)", 0.0, 50.0, 10.0)

with st.sidebar.expander("Overcurrent Protection", expanded=False):
    I_inst_mult = st.slider("Instantaneous OC (×FLC)", 1.0, 20.0, 8.0)
    I_dt_mult   = st.slider("Definite-time OC (×FLC)", 1.0, 10.0, 2.0)
    t_dt        = st.slider("Definite-time delay (s)", 0.1, 10.0, 1.0)

with st.sidebar.expander("IDMT OC Settings", expanded=False):
    I_pickup_mult = st.slider("IDMT pickup (×FLC)", 1.0, 5.0, 1.2)
    tms           = st.slider("TMS", 0.05, 10.0, 0.1)
    curve_type    = st.selectbox("Curve type", ["NI", "VI", "EI"])

with st.sidebar.expander("Earth Fault (EF) Protection", expanded=False):
    I_ef_mult = st.slider("EF pickup (×FLC)", 0.05, 1.0, 0.2)
    t_ef      = st.slider("EF delay (s)", 0.05, 10.0, 0.5)

with st.sidebar.expander("Negative Phase Sequence (NPS)", expanded=False):
    I2_pickup_pct = st.slider("NPS pickup (%FLC)", 1.0, 50.0, 10.0)
    t_nps         = st.slider("NPS delay (s)", 0.05, 10.0, 0.5)

with st.sidebar.expander("Locked Rotor Protection", expanded=False):
    LR_prot_mult = st.slider("Locked Rotor pickup (×FLC)", 3.0, 10.0, 6.0)
    LR_time      = st.slider("Locked Rotor max time (s)", 1.0, 60.0, 10.0)

with st.sidebar.expander("Thermal Overload (ANSI 49)", expanded=False):
    tdm = st.slider("Thermal Damage Multiplier (TDM)", 1.0, 15.0, 5.0, step=0.5)
    top_pickup_mult = st.slider("Thermal Overload Ip (×FLC)", 1.0, 10.0, 1.0, step=0.1)


# ──────────────────────────────
# ──────────────────────────────
# Calculations
# ──────────────────────────────
I_lr       = I_lr_mult * I_f
I_th       = Ith_mult * I_f
I2         = I2_unbal_pct / 100 * I_f
I_pickup   = I_pickup_mult * I_f
I_ef       = I_ef_mult * I_f
I2_pickup  = I2_pickup_pct / 100 * I_f
I_LR_prot  = LR_prot_mult * I_f
Ip_top     = top_pickup_mult * I_f   # <--- NEW

currents = np.logspace(np.log10(I_f), np.log10(20 * I_f), 200)

thermal_cold = [thermal_trip_time(i, I_th, tau, 0.0, K_nps, I2) for i in currents]
thermal_hot  = [thermal_trip_time(i, I_th, tau, A2_hot, K_nps, I2) for i in currents]
t_start, I_start = motor_start_curve(I_f, I_lr, t_acc, V_start)
idmt_times  = [idmt_trip_time(i, I_pickup, tms, curve=curve_type) for i in currents]

# --- Thermal Overload (ANSI 49) ---
def top_trip_time(I, Ip, TDM):
    M = I / Ip
    if M <= 1:
        return np.inf
    return (87.4 * TDM) / (M**2 - 1)

top_times = [max(top_trip_time(i, Ip_top, tdm), 1e-2) for i in currents]



# ──────────────────────────────
# Variable visibility control
# ──────────────────────────────
ALL_SERIES = [
    "Thermal limit (Cold)",
    "Thermal limit (Hot)",
    "Motor start",
    f"IDMT ({curve_type})",
    "Thermal Overload (ANSI 49)",              # ← NEW
    "Instantaneous OC (Pick-up)",
    "Definite-time OC (Pick-up)",
    "Definite-time OC (Time)",
    "Earth Fault (Pick-up)",
    "Earth Fault (Time)",
    "NPS (Pick-up)",
    "NPS (Time)",
    "Locked Rotor Pickup (Pick-up)",
    "Locked Rotor Max Time (Time)",
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
            "Thermal Overload (ANSI 49)",  # ← add here if you want it on by default
            "Instantaneous OC (Pick-up)",
            "Definite-time OC (Time)",
        ],
    )
with colB:
    sel_all = st.checkbox("Select all", value=False)
    if sel_all:
        selected = ALL_SERIES[:]
    sel_none = st.checkbox("Select none", value=False)
    if sel_none:
        selected = []
st.markdown("## Recommended Motor Protection setting")

with st.expander("Recommended Motor Protection setting", expanded=False):
    # Compute typical K from current LRC (×FLC)
    _lrc = float(I_lr_mult)  # your slider (×FLC)
    K_typical_val = 175 / (_lrc**2) if _lrc > 0 else None
    K_cons_val    = 230 / (_lrc**2) if _lrc > 0 else None

    rows = [
        {"Category":"Motor Parameters", "Parameter":"Motor Power, FLC, Voltage", "Recommended Setting/Typical Value":"As per motor datasheet"},
        {"Category":"Motor Parameters", "Parameter":"Locked Rotor Current (LRC) / Starting Current", "Recommended Setting/Typical Value":"5–7 × FLC"},

        {"Category":"Thermal Overload Protection", "Parameter":"Thermal Pickup (Ith)", "Recommended Setting/Typical Value":"FLC (or 1.05 × FLC for service factor)"},
        {"Category":"Thermal Overload Protection", "Parameter":"Heating Time Constant (τ)", "Recommended Setting/Typical Value":"25 minutes (typical)"},
        {"Category":"Thermal Overload Protection", "Parameter":"Cooling Time Constant", "Recommended Setting/Typical Value":"75 minutes (typical)"},
        {"Category":"Thermal Overload Protection", "Parameter":"A² Factor (Initial Heating State)", "Recommended Setting/Typical Value":"Cold: 0.0; Hot: 0.5"},
        {"Category":"Thermal Overload Protection", "Parameter":"Negative Sequence Heating (K Factor)",
         "Recommended Setting/Typical Value": (
             f"Typical: K = 175 / LRC² ≈ {K_typical_val:.2f}\n"
             f"Conservative: K = 230 / LRC² ≈ {K_cons_val:.2f}"
         )},
        {"Category":"Thermal Overload Protection", "Parameter":"IDMT Curve Type", "Recommended Setting/Typical Value":"NI / VI / EI (based on coordination) — Typical: NI"},
        {"Category":"Thermal Overload Protection", "Parameter":"Overload Curve Coordination", "Recommended Setting/Typical Value":"Must lie below motor thermal damage curve"},

        {"Category":"Overcurrent Protection", "Parameter":"Instantaneous OC", "Recommended Setting/Typical Value":"10 × FLC or 1.25 × Starting Current"},
        {"Category":"Overcurrent Protection", "Parameter":"Definite-Time OC", "Recommended Setting/Typical Value":"50–100 ms"},

        {"Category":"Earth Fault Protection", "Parameter":"Pickup (Residual CT)", "Recommended Setting/Typical Value":"0.2 × FLC"},
        {"Category":"Earth Fault Protection", "Parameter":"Pickup (CBCT)", "Recommended Setting/Typical Value":"0.1 × FLC"},
        {"Category":"Earth Fault Protection", "Parameter":"Time Delay", "Recommended Setting/Typical Value":"100 ms"},

        {"Category":"Locked Rotor Protection", "Parameter":"Pickup", "Recommended Setting/Typical Value":"> FLC, < Starting Current (refer OEM datasheet)"},
        {"Category":"Locked Rotor Protection", "Parameter":"Time Delay", "Recommended Setting/Typical Value":"> Start time @ 80% V, < Cold stall time"},

        {"Category":"Stall Protection", "Parameter":"Pickup", "Recommended Setting/Typical Value":"2.5–3 × FLC (refer OEM datasheet)"},
        {"Category":"Stall Protection", "Parameter":"Time Delay", "Recommended Setting/Typical Value":"3–5 s (typical)"},

        {"Category":"Negative Phase Sequence Protection", "Parameter":"Definite Time Element", "Recommended Setting/Typical Value":"30–50% of FLC"},
        {"Category":"Negative Phase Sequence Protection", "Parameter":"Definite Time", "Recommended Setting/Typical Value":"3–5 s"},
        {"Category":"Negative Phase Sequence Protection", "Parameter":"IDMT Element (IEC, ~1% V unbalance ≈ 7% I)", "Recommended Setting/Typical Value":"Pickup: 10–15% of FLC; TMS: 1"},
    ]

    df_rec = pd.DataFrame(rows)

    st.dataframe(df_rec, use_container_width=True, hide_index=True)

    csv_bytes = df_rec.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Recommended Settings (CSV)",
        csv_bytes,
        file_name="recommended_motor_protection_settings.csv",
        mime="text/csv",
        use_container_width=True,
    )
# ──────────────────────────────
# Plotting (Plotly)
# ──────────────────────────────
def _finite_mask(y):
    a = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(a) & (a > 0)  # log-axis safe
    return ok

fig = go.Figure()

# Curves — trace names include the equipment (for clarity in exports/legend)
equip_suffix = f" — {equipment_name}" + (f" ({equipment_tag})" if equipment_tag else "")

if "Thermal limit (Cold)" in selected:
    ok = _finite_mask(thermal_cold)
    fig.add_trace(go.Scatter(
        x=currents[ok], y=np.asarray(thermal_cold)[ok],
        mode="lines", name=f"Thermal limit (Cold){equip_suffix}",
        line=dict(color="darkblue", width=3)
    ))

if "Thermal limit (Hot)" in selected:
    ok = _finite_mask(thermal_hot)
    fig.add_trace(go.Scatter(
        x=currents[ok], y=np.asarray(thermal_hot)[ok],
        mode="lines", name=f"Thermal limit (Hot){equip_suffix}",
        line=dict(color="darkorange", width=3)
    ))

if "Motor start" in selected:
    ok = _finite_mask(t_start)
    fig.add_trace(go.Scatter(
        x=I_start, y=t_start,
        mode="lines", name=f"Motor start ({V_start}%){equip_suffix}",
        line=dict(color="black", width=3, dash="dash")
    ))

if f"IDMT ({curve_type})" in selected:
    ok = _finite_mask(idmt_times)
    fig.add_trace(go.Scatter(
        x=currents[ok], y=np.asarray(idmt_times)[ok],
        mode="lines", name=f"IDMT ({curve_type}){equip_suffix}",
        line=dict(color="purple", width=3, dash="dot")
    ))

# Vertical/horizontal lines (don’t clutter legend)
def _vline(x, color):
    fig.add_shape(type="line", x0=x, x1=x, xref="x", yref="paper", y0=0, y1=1,
                  line=dict(color=color, width=3, dash="dash"))

def _hline(y, color):
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", y0=y, y1=y, yref="y",
                  line=dict(color=color, width=3, dash="dash"))

if "Instantaneous OC (Pick-up)" in selected:        _vline(I_inst_mult * I_f, "darkred")
if "Definite-time OC (Pick-up)" in selected:        _vline(I_dt_mult * I_f, "darkgreen")
if "Definite-time OC (Time)" in selected:      _hline(t_dt, "darkgreen")
if "Earth Fault (Pick-up)" in selected:             _vline(I_ef, "darkmagenta")
if "Earth Fault (Time)" in selected:           _hline(t_ef, "darkmagenta")
if "NPS (Pick-up)" in selected:                     _vline(I2_pickup, "darkcyan")
if "NPS (Time)" in selected:                   _hline(t_nps, "darkcyan")
if "Locked Rotor Pickup (Pick-up)" in selected:     _vline(I_LR_prot, "saddlebrown")
if "Locked Rotor Max Time (Time)" in selected: _hline(LR_time, "saddlebrown")
if "Thermal Overload (ANSI 49)" in selected:
    ok = _finite_mask(top_times)
    fig.add_trace(go.Scatter(
        x=currents[ok], y=np.asarray(top_times)[ok],
        mode="lines",
        name=f"Thermal Overload (ANSI 49){equip_suffix}",
        line=dict(color="teal", width=3, dash="longdash")
    ))


# Title includes equipment
title_main = f"{equipment_name} — Protection Curves"
if equipment_tag:
    title_main += f" [{equipment_tag}]"

fig.update_layout(
    title=f"{title_main}<br><sup>Motor: {kw} kW, {V} V, FLC: {I_f:.1f} A</sup>",
    xaxis_title="Current (A)",
    yaxis_title="Time (s)",
    template="simple_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=70, b=40),
    showlegend=False,  # legend moved below
    font=dict(color="black", size=14),
)
fig.update_xaxes(type="log", showgrid=False, title_font=dict(color="black"), tickfont=dict(color="black"))
fig.update_yaxes(type="log", showgrid=False, title_font=dict(color="black"), tickfont=dict(color="black"))
fig.update_layout(hovermode="x unified", dragmode="zoom")

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "scrollZoom": True,
        "doubleClick": "reset",
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "autoScale2d", "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian",
            "lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "resetScale2d"
        ],
        "modeBarButtonsToAdd": ["zoom2d", "pan2d", "toImage"],
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"{equipment_name.replace(' ', '_').lower()}_protection_curves",
            "scale": 2,
            "height": 600,
            "width": 1000,
        },
    },
)

# Legend panel BELOW the chart (also shows equipment)
def swatch(color):
    return f'<span style="display:inline-block;width:12px;height:12px;background:{color};margin-right:8px;border:1px solid #333"></span>'

legend_items = []
if "Thermal limit (Cold)" in selected: legend_items.append(swatch("darkblue")   + f"Thermal limit (Cold) — {equipment_name}")
if "Thermal limit (Hot)"  in selected: legend_items.append(swatch("darkorange") + f"Thermal limit (Hot) — {equipment_name}")
if "Motor start"          in selected: legend_items.append(swatch("black")      + f"Motor start ({V_start}%) — {equipment_name}")
if f"IDMT ({curve_type})" in selected: legend_items.append(swatch("purple")     + f"IDMT ({curve_type}) — {equipment_name}")
if "Instantaneous OC (Pick-up)" in selected: legend_items.append(swatch("darkred")      + "Inst. OC (Pick-up)")
if "Definite-time OC (Pick-up)" in selected:   legend_items.append(swatch("darkgreen")  + "Definite-time OC (Pick-up)")
if "Definite-time OC (Time)" in selected: legend_items.append(swatch("darkgreen")  + "Definite-time OC (Time)")
if "Earth Fault (Pick-up)" in selected:        legend_items.append(swatch("darkmagenta")+ "Earth Fault (Pick-up)")
if "Earth Fault (Time)" in selected:      legend_items.append(swatch("darkmagenta")+ "Earth Fault (Time)")
if "NPS (Pick-up)" in selected:                legend_items.append(swatch("darkcyan")   + "NPS (Pick-up)")
if "NPS (Time)" in selected:              legend_items.append(swatch("darkcyan")   + "NPS (Time)")
if "Locked Rotor Pickup (Pick-up)" in selected:legend_items.append(swatch("saddlebrown")+ "Locked Rotor Pickup (Pick-up)")
if "Locked Rotor Max Time (Time)" in selected: legend_items.append(swatch("saddlebrown")+ "Locked Rotor Max Time (Time)")
if "Thermal Overload (ANSI 49)" in selected:
    legend_items.append(swatch("teal") + f"Thermal Overload (ANSI 49) — {equipment_name}")


st.markdown("### Legend")
if equipment_name or equipment_tag:
    st.caption(f"Protection curves for **{equipment_name}**" + (f" (Tag: {equipment_tag})" if equipment_tag else ""))

if legend_items:
    st.markdown("<div style='line-height:1.8'>" + "<br>".join(legend_items) + "</div>", unsafe_allow_html=True)
else:
    st.info("No variables selected. Use the selector above to add curves/limits to the chart.")

# Footer
st.markdown(
    "<hr style='margin-top:30px;margin-bottom:10px;border:1px solid #ccc'>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='text-align:center; color:gray; font-size:14px;'>"
    "Developed by <b>Arindam Chowdhury</b>, Electrical Engineer"
    "</div>",
    unsafe_allow_html=True
)
