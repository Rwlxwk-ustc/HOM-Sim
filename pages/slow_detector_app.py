import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Slow Detector HOM Simulation")
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 物理计算核心 (慢探测器)
# ---------------------------------------------------------
W_MIN, W_MAX = -100.0, 100.0
omega_axis = np.linspace(W_MIN, W_MAX, 5000)
delta_w = 0.8  # 固定子波包带宽

def get_slow_plot_data(tau, N, Omega, Dw1, Dw2, wc1, wc2, I1, I2, dw0):
    j_indices = np.arange(N) - (N - 1) / 2.0
    w1_j = dw0 / 2.0 + j_indices * Omega
    w2_j = -dw0 / 2.0 + j_indices * Omega
    n1_j = I1 * np.exp(-(w1_j - wc1)**2 / (2 * Dw1**2))
    n2_j = I2 * np.exp(-(w2_j - wc2)**2 / (2 * Dw2**2))
    N1_total = np.sum(n1_j)
    N2_total = np.sum(n2_j)
    
    E1_spec = np.zeros_like(omega_axis)
    E2_spec = np.zeros_like(omega_axis)
    for j in range(N):
        E1_spec += n1_j[j] * np.exp(-(omega_axis - w1_j[j])**2 / (2 * delta_w**2))
        E2_spec += n2_j[j] * np.exp(-(omega_axis - w2_j[j])**2 / (2 * delta_w**2))
    
    tau_range = np.linspace(-5.0, 5.0, 1000)
    overlap_sum = np.sum(n1_j * n2_j)
    mismatch_factor = np.exp(-dw0**2 / (2 * delta_w**2))
    interference = 0.5 * mismatch_factor * np.exp(-delta_w**2 * tau_range**2 / 2) * overlap_sum
    baseline = 0.25 * (N1_total + N2_total)**2
    Pc_curve = baseline - interference
    Pc_final = Pc_curve / baseline
    
    current_interference = 0.5 * mismatch_factor * np.exp(-delta_w**2 * tau**2 / 2) * overlap_sum
    current_Pc = (baseline - current_interference) / baseline
    return E1_spec, E2_spec, tau_range, Pc_final, current_Pc

# ---------------------------------------------------------
# Streamlit UI (侧边栏中文，图表英文)
# ---------------------------------------------------------
st.title("多模 HOM 干涉模拟 (慢探测器响应)")

st.sidebar.header("参数调节面板")

with st.sidebar.expander("显示范围调节", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        zoom_w = st.slider("光谱 X 轴范围 ±", 10.0, 100.0, 100.0, 1.0, format="%.1f")
    with col2:
        zoom_t = st.slider("时域 X 轴范围 ±", 0.5, 5.0, 5.0, 0.1, format="%.2f")

with st.sidebar.expander("核心时延参数", expanded=True):
    tau = st.slider("延迟 τ", -5.0, 5.0, 0.0, 0.01, format="%.3f")

with st.sidebar.expander("光路1 参数 (部分固定)", expanded=False):
    Dw1 = st.slider("带宽 Δω₁", 5.0, 50.0, 20.0, 1.0)
    # wc1 固定为 0, I1 固定为 1

with st.sidebar.expander("光路2 参数", expanded=False):
    I2 = st.slider("强度 I₂", 0.1, 2.0, 1.0, 0.05)
    Dw2 = st.slider("带宽 Δω₂", 5.0, 50.0, 20.0, 1.0)
    wc2 = st.slider("包络中心 ω₂꜀", -20.0, 20.0, 0.0, 0.5)

with st.sidebar.expander("梳齿与失配参数", expanded=False):
    N = st.slider("梳齿数 N", 1, 15, 7, 1)
    Omega = st.slider("间距 Ω", 1.0, 20.0, 10.0, 0.5)
    dw0 = st.slider("整体失配 Δω₀", -5.0, 5.0, 0.0, 0.1)

# 固定参数
wc1, I1 = 0.0, 1.0

E1_s, E2_s, t_r, Pc_f, c_Pc = get_slow_plot_data(
    tau, N, Omega, Dw1, Dw2, wc1, wc2, I1, I2, dw0
)

# 绘图
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(omega_axis, E1_s, label='Path 1 Spectrum', color='blue', alpha=0.6)
ax1.plot(omega_axis, E2_s, label='Path 2 Spectrum', color='red', linestyle='--')
ax1.set_title("Frequency Domain Spectra")
ax1.legend(loc='upper right')
ax1.set_xlim(-zoom_w, zoom_w)
max_E = max(np.max(E1_s), np.max(E2_s))
ax1.set_ylim(0, max_E * 1.2 if max_E > 0 else 1)
ax1.set_xlabel("Frequency (a.u.)")
ax1.set_ylabel("Intensity")

ax2 = plt.subplot(2, 1, 2)
ax2.plot(t_r, Pc_f, color='purple', lw=1.5, label='$g^{(2)}(\\tau)$')
ax2.plot(tau, c_Pc, 'ro', markersize=6, label='Current $\\tau$')
ax2.set_title("Normalized Coincidence Probability $g^{(2)}(\\tau)$ [Slow Detector]")
ax2.set_xlim(-zoom_t, zoom_t)
ax2.set_ylim(0.45, 1.1)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Delay $\\tau$ (time units)")
ax2.set_ylabel("Coincidence Probability")
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)

st.info(f"当前延迟 τ = {tau:.3f} 时，符合概率 $g^{(2)}$ = {c_Pc:.4f}")
