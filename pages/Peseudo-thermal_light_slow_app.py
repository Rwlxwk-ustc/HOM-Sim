import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Pseudo‑thermal HOM Simulation")
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 物理计算核心 (赝热光模型)
# ---------------------------------------------------------
W_MIN, W_MAX = -100.0, 100.0
POINTS = 5000
omega = np.linspace(W_MIN, W_MAX, POINTS)
d_omega = (W_MAX - W_MIN) / POINTS
delta_w = 0.8 

def gaussian_comb(omega, N, Omega, omega_c, Delta_w, delta_w, intensity):
    E = np.zeros_like(omega, dtype=complex)
    indices = np.arange(N) - (N - 1) / 2.0
    for j in indices:
        wj = omega_c + j * Omega
        comb_tooth = np.exp(-(omega - wj)**2 / (4 * delta_w**2))
        weight = np.sqrt(intensity) * np.exp(-(wj - omega_c)**2 / (4 * Delta_w**2))
        E += weight * comb_tooth
    return E

def get_plot_data(tau, N, Om1, Om2, Dw1, Dw2, wc1, wc2, I1, I2):
    E1 = gaussian_comb(omega, N, Om1, wc1, Dw1, delta_w, I1)
    E2_base = gaussian_comb(omega, N, Om2, wc2, Dw2, delta_w, I2)
    P1 = np.sum(np.abs(E1)**2 * d_omega)
    P2 = np.sum(np.abs(E2_base)**2 * d_omega)
    tau_range = np.linspace(-2.0, 2.0, 500)
    product = np.conj(E1) * E2_base
    overlap_list = np.array([np.sum(product * np.exp(-1j * omega * t) * d_omega) for t in tau_range])
    M_tau_sq = np.abs(overlap_list)**2 / (P1 * P2)
    normalization_factor = 0.25 * (P1 + P2)**2
    Pc_curve = 0.25 * ((P1 + P2)**2 + P1**2 + P2**2) - 0.5 * (P1 * P2) * M_tau_sq
    Pc_final = Pc_curve / normalization_factor
    current_overlap = np.sum(product * np.exp(-1j * omega * tau) * d_omega)
    current_M = np.abs(current_overlap) / np.sqrt(P1 * P2)
    current_Pc = (0.25 * ((P1 + P2)**2 + P1**2 + P2**2) - 0.5 * (P1 * P2) * (current_M**2)) / normalization_factor
    return E1, E2_base, tau_range, Pc_final, current_Pc, current_M

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.title("多模 HOM 干涉模拟 (赝热光 / Pseudo‑thermal Light)")

st.sidebar.header("参数调节面板")

with st.sidebar.expander("光路1 参数", expanded=True):
    Om1 = st.slider('梳间距 Ω₁', 1.0, 20.0, 10.0, 0.5, key='om1')
    Dw1 = st.slider('包络带宽 Δω₁', 5.0, 50.0, 20.0, 1.0, key='dw1')
    wc1 = st.slider('包络中心 ω₁꜀', -10.0, 10.0, 0.0, 0.5, key='wc1')
    I1  = st.slider('强度 I₁', 0.1, 2.0, 1.0, 0.1, key='i1')

with st.sidebar.expander("光路2 参数", expanded=True):
    Om2 = st.slider('梳间距 Ω₂', 1.0, 20.0, 10.0, 0.5, key='om2')
    Dw2 = st.slider('包络带宽 Δω₂', 5.0, 50.0, 20.0, 1.0, key='dw2')
    wc2 = st.slider('包络中心 ω₂꜀', -10.0, 10.0, 0.0, 0.5, key='wc2')
    I2  = st.slider('强度 I₂', 0.1, 2.0, 1.0, 0.1, key='i2')

with st.sidebar.expander("全局参数", expanded=True):
    N   = st.slider('梳齿数 N', 1, 15, 5, 1, key='n')
    tau = st.slider('延迟 τ', -1.0, 1.0, 0.0, 0.02, key='tau')

E1, E2, t_r, Pc_f, c_Pc, c_M = get_plot_data(tau, N, Om1, Om2, Dw1, Dw2, wc1, wc2, I1, I2)

# 绘图
fig, (ax_spectra, ax_curve) = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(hspace=0.3)

ax_spectra.plot(omega, np.abs(E1)**2, label='Path 1 Spectrum', color='blue', alpha=0.6)
ax_spectra.plot(omega, np.abs(E2)**2, label='Path 2 Spectrum', color='red', linestyle='--')
ax_spectra.set_title("Frequency Domain Spectra")
ax_spectra.set_ylim(0, max(np.max(np.abs(E1)**2), np.max(np.abs(E2)**2)) * 1.2 or 1)
ax_spectra.legend()
ax_spectra.set_xlabel("Frequency (a.u.)")
ax_spectra.set_ylabel("Intensity")

ax_curve.plot(t_r, Pc_f, color='orange', lw=1.5, label='$g^{(2)}(\\tau)$')
ax_curve.plot([tau], [c_Pc], 'ro', markersize=6, label='Current $\\tau$')
ax_curve.set_title(f"Normalized Coincidence Probability $g^{(2)}(\\tau)$ [Pseudo‑thermal]  | Current |M| = {c_M:.4f}")
ax_curve.set_xlabel("Delay $\\tau$ (time units)")
ax_curve.set_ylabel("Coincidence Probability")
ax_curve.grid(True, alpha=0.3)
ax_curve.legend()

y_min, y_max = np.min(Pc_f), np.max(Pc_f)
padding = (y_max - y_min) * 0.2 if y_max != y_min else 0.1
ax_curve.set_ylim(y_min - padding, y_max + padding)

st.pyplot(fig)

st.info(f"当前延迟 τ = {tau:.3f} 时，符合概率 $g^{(2)}$ = {c_Pc:.4f}，模重叠度 |M| = {c_M:.4f}")
