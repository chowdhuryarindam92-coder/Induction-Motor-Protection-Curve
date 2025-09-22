import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
st.title("Indcution Motor Protection Curves")

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
thermal_hot = [thermal_trip_time(i, I_th, tau, A2_hot, K_nps, I2) for i in currents]
t_start, I_start = motor_start_curve(I_f, I_lr, t_acc, V_start)
idmt_times = [idmt_trip_time(i, I_pickup, tms, curve=curve_type) for i in currents]

# ──────────────────────────────
# Plotting
# ──────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

ax.loglog(currents, thermal_cold, label="Thermal limit (Cold)", color="b", linewidth=2)
ax.loglog(currents, thermal_hot, label="Thermal limit (Hot)", color="orange", linewidth=2)
ax.loglog(I_start, t_start, label=f"Motor start ({V_start}%)", linestyle="--", color="k")
ax.loglog(currents, idmt_times, label=f"IDMT ({curve_type})", linestyle=":", linewidth=2)

# Protection pickups
ax.axvline(I_inst_mult * I_f, color="r", linestyle="--", label="Inst. OC")
ax.axvline(I_dt_mult * I_f, color="g", linestyle="--")
ax.axhline(t_dt, color="g", linestyle="--", label="Definite-time OC")
ax.axvline(I_ef, color="m", linestyle="--")
ax.axhline(t_ef, color="m", linestyle="--", label="Earth Fault")
ax.axvline(I2_pickup, color="c", linestyle="--")
ax.axhline(t_nps, color="c", linestyle="--", label="NPS")
ax.axvline(I_LR_prot, color="brown", linestyle="-.", linewidth=2, label="Locked Rotor Pickup")
ax.axhline(LR_time, color="brown", linestyle=":", linewidth=2, label="Locked Rotor Max Time")

ax.set_xlabel("Current (A)")
ax.set_ylabel("Time (s)")
ax.set_title(f"Motor Protection Curves\nMotor: {kw} kW, {V} V")
ax.legend()
ax.grid(True, which="both")

st.pyplot(fig)
