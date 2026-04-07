import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 页面设置（图表文字用英文，避免中文字体问题）
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Fast Detector HOM Simulation")
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 物理计算核心 (快探测器解析模型，包含线宽 dw1, dw2)
# ---------------------------------------------------------
W_MIN, W_MAX = -150.0, 150.0
omega_axis = np.linspace(W_MIN, W_MAX, 6000)

def get_fast_plot_data(tau, N, Om1, Om2, w0_1, w0_2, Dw1, Dw2, wc1, wc2, I1, I2, dw1, dw2):
    j_indices = np.arange(N) - (N - 1) / 2.0
    w1_j = w0_1 + j_indices * Om1
    w2_k = w0_2 + j_indices * Om2
    n1_j = I1 * np.exp(-(w1_j - wc1)**2 / (2 * Dw1**2))
    n2_k = I2 * np.exp(-(w2_k - wc2)**2 / (2 * Dw2**2))
    N1_total = np.sum(n1_j)
    N2_total = np.sum(n2_k)
    
    E1_spec = np.zeros_like(omega_axis)
    E2_spec = np.zeros_like(omega_axis)
    for i in range(N):
        E1_spec += n1_j[i] * np.exp(-(omega_axis - w1_j[i])**2 / (2 * dw1**2))
        E2_spec += n2_k[i] * np.exp(-(omega_axis - w2_k[i])**2 / (2 * dw2**2))
    
    tau_range = np.linspace(-6.0, 6.0, 5000)
    S1 = np.sum(n1_j[:, None] * np.exp(-1j * w1_j[:, None] * tau_range), axis=0)
    S2 = np.sum(n2_k[:, None] * np.exp( 1j * w2_k[:, None] * tau_range), axis=0)
    global_envelope = np.exp(-(dw1**2 + dw2**2) / 2.0 * tau_range**2)
    interference = 2 * np.real(S1 * S2) * global_envelope
    baseline = (N1_total + N2_total)**2
    Pc_final = 1.0 - interference / baseline
    
    c_S1 = np.sum(n1_j * np.exp(-1j * w1_j * tau))
    c_S2 = np.sum(n2_k * np.exp( 1j * w2_k * tau))
    c_envelope = np.exp(-(dw1**2 + dw2**2) / 2.0 * tau**2)
    c_interference = 2 * np.real(c_S1 * c_S2) * c_envelope
    current_Pc = 1.0 - c_interference / baseline
    return E1_spec, E2_spec, tau_range, Pc_final, current_Pc

# ---------------------------------------------------------
# 2. Streamlit UI (侧边栏控件保留中文，图表用英文)
# ---------------------------------------------------------
st.title("多模 HOM 干涉模拟 (快探测器响应)")

st.sidebar.header("参数调节面板")

with st.sidebar.expander("显示范围调节", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        zoom_w = st.slider("光谱 X 轴范围 ±", 10.0, 150.0, 150.0, 1.0, format="%.1f")
    with col2:
        zoom_t = st.slider("时域 X 轴范围 ±", 0.1, 6.0, 6.0, 0.1, format="%.2f")

with st.sidebar.expander("核心时延参数", expanded=True):
    tau = st.slider("延迟 τ", -6.0, 6.0, 0.0, 0.01, format="%.3f")

with st.sidebar.expander("光路1 参数", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        I1 = st.slider("强度 I₁", 0.1, 2.0, 1.0, 0.05)
        Om1 = st.slider("间距 Ω₁", 1.0, 30.0, 10.0, 0.5)
        w0_1 = st.slider("梳基频 ω₁₀", -80.0, 80.0, 0.0, 1.0)
        Dw1 = st.slider("带宽 Δω₁", 5.0, 80.0, 30.0, 1.0)
    with col2:
        wc1 = st.slider("包络中心 ω₁꜀", -80.0, 80.0, 0.0, 1.0)
        dw1 = st.slider("线宽 δω₁", 0.1, 5.0, 0.8, 0.05)

with st.sidebar.expander("光路2 参数", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        I2 = st.slider("强度 I₂", 0.1, 2.0, 1.0, 0.05)
        Om2 = st.slider("间距 Ω₂", 1.0, 30.0, 10.0, 0.5)
        w0_2 = st.slider("梳基频 ω₂₀", -80.0, 80.0, 0.0, 1.0)
        Dw2 = st.slider("带宽 Δω₂", 5.0, 80.0, 30.0, 1.0)
    with col2:
        wc2 = st.slider("包络中心 ω₂꜀", -80.0, 80.0, 0.0, 1.0)
        dw2 = st.slider("线宽 δω₂", 0.1, 5.0, 0.8, 0.05)

with st.sidebar.expander("梳齿结构参数", expanded=False):
    N = st.slider("梳齿数 N", 1, 31, 11, 1)

# 计算数据
E1_s, E2_s, t_r, Pc_f, c_Pc = get_fast_plot_data(
    tau, N, Om1, Om2, w0_1, w0_2, Dw1, Dw2, wc1, wc2, I1, I2, dw1, dw2
)

# ---------------------------------------------------------
# 3. 绘图（全部使用英文标签）
# ---------------------------------------------------------
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
ax2.plot(t_r, Pc_f, color='green', lw=1.2, label='$g^{(2)}(\\tau)$')
ax2.plot(tau, c_Pc, 'ro', markersize=6, label='Current $\\tau$')
ax2.set_title("Normalized Coincidence Probability $g^{(2)}(\\tau)$ [Fast Detector]")
ax2.set_xlim(-zoom_t, zoom_t)
ax2.set_ylim(0.0, 2.2)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Delay $\\tau$ (time units)")
ax2.set_ylabel("Coincidence Probability")
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)

st.info(f"当前延迟 τ = {tau:.3f} 时，符合概率 $g^{(2)}$ = {c_Pc:.4f}")
