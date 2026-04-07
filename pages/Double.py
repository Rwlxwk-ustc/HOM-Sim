import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 页面设置（图表文字用英文，避免中文字体问题）
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="Dual‑Frequency Beams HOM Simulation")
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 物理计算核心（双频光束时频解析模型）
# ---------------------------------------------------------
# 频域轴 (MHz)
F_MIN, F_MAX = -1500.0, 1500.0
f_axis = np.linspace(F_MIN, F_MAX, 10000)

# 时域轴 (us) – 高采样率以分辨高频拍频
TAU_MIN, TAU_MAX = -0.5, 0.5
tau_range = np.linspace(TAU_MIN, TAU_MAX, 20000)

def get_fast_plot_data(tau, f_center, f_spacing, shift_31, shift_42, I1, I2, dw1, dw2):
    # 光路1：两个频率对称分布
    f1 = f_center - f_spacing / 2.0
    f2 = f_center + f_spacing / 2.0
    f1_arr = np.array([f1, f2])

    # 光路2：频率3和4相对1和2的偏移
    f3 = f1 + shift_31
    f4 = f2 + shift_42
    f2_arr = np.array([f3, f4])

    # 光子数/强度分布（每路两个频率强度相同）
    n1_arr = np.array([I1, I1])
    n2_arr = np.array([I2, I2])

    N1_total = np.sum(n1_arr)
    N2_total = np.sum(n2_arr)

    # 频域光谱（高斯线宽）
    E1_spec = np.zeros_like(f_axis)
    E2_spec = np.zeros_like(f_axis)
    for i in range(2):
        E1_spec += n1_arr[i] * np.exp(-(f_axis - f1_arr[i])**2 / (2 * dw1**2))
        E2_spec += n2_arr[i] * np.exp(-(f_axis - f2_arr[i])**2 / (2 * dw2**2))

    # 时域相干计算（2π因子：f 单位 MHz，tau 单位 us）
    S1 = np.sum(n1_arr[:, None] * np.exp(-1j * 2 * np.pi * f1_arr[:, None] * tau_range), axis=0)
    S2 = np.sum(n2_arr[:, None] * np.exp( 1j * 2 * np.pi * f2_arr[:, None] * tau_range), axis=0)

    # 全局衰减包络（由线宽决定）
    global_envelope = np.exp(-2 * np.pi**2 * (dw1**2 + dw2**2) * tau_range**2)

    interference = 2 * np.real(S1 * S2) * global_envelope
    baseline = (N1_total + N2_total)**2
    Pc_final = 1.0 - interference / baseline

    # 当前 tau 点的 g(2)
    c_S1 = np.sum(n1_arr * np.exp(-1j * 2 * np.pi * f1_arr * tau))
    c_S2 = np.sum(n2_arr * np.exp( 1j * 2 * np.pi * f2_arr * tau))
    c_envelope = np.exp(-2 * np.pi**2 * (dw1**2 + dw2**2) * tau**2)
    c_interference = 2 * np.real(c_S1 * c_S2) * c_envelope
    current_Pc = 1.0 - c_interference / baseline

    return E1_spec, E2_spec, Pc_final, current_Pc

# ---------------------------------------------------------
# 2. Streamlit 界面（侧边栏中文，图表英文）
# ---------------------------------------------------------
st.title("多模 HOM 干涉模拟 (双频光束 / Dual‑Frequency Beams)")

st.sidebar.header("参数调节面板")

# ----- 显示范围调节 -----
with st.sidebar.expander("显示范围调节", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        zoom_f = st.slider("光谱 X 轴范围 ± (MHz)", 100.0, 1500.0, 1000.0, 10.0, format="%.1f")
    with col2:
        zoom_t = st.slider("时域 X 轴范围 ± (μs)", 0.01, 0.5, 0.2, 0.01, format="%.3f")

# ----- 核心时延参数 -----
with st.sidebar.expander("核心时延参数", expanded=True):
    tau = st.slider("延迟 τ (μs)", -0.5, 0.5, 0.0, 0.001, format="%.4f")

# ----- 光路1 参数 -----
with st.sidebar.expander("光路1 参数", expanded=False):
    f_center = st.slider("中心频率 (MHz)", -500.0, 500.0, 0.0, 10.0, format="%.1f")
    f_spacing = st.slider("模式间隔 (MHz)", 100.0, 2000.0, 800.0, 10.0, format="%.1f")
    I1 = st.slider("强度 I₁", 0.1, 2.0, 1.0, 0.05)
    dw1 = st.slider("线宽 δω₁ (MHz)", 0.1, 20.0, 2.0, 0.1, format="%.1f")

# ----- 光路2 参数（相对偏移）-----
with st.sidebar.expander("光路2 参数", expanded=False):
    shift_31 = st.slider("频差: 3 比 1 偏右 (MHz)", 0.0, 100.0, 10.0, 1.0, format="%.1f")
    shift_42 = st.slider("频差: 4 比 2 偏右 (MHz)", 100.0, 500.0, 200.0, 5.0, format="%.1f")
    I2 = st.slider("强度 I₂", 0.1, 2.0, 1.0, 0.05)
    dw2 = st.slider("线宽 δω₂ (MHz)", 0.1, 20.0, 2.0, 0.1, format="%.1f")

# ---------------------------------------------------------
# 3. 数据计算与绘图
# ---------------------------------------------------------
E1_s, E2_s, Pc_f, c_Pc = get_fast_plot_data(
    tau, f_center, f_spacing, shift_31, shift_42, I1, I2, dw1, dw2
)

# 创建图形
fig = plt.figure(figsize=(12, 8))

# 子图1：频谱
ax1 = plt.subplot(2, 1, 1)
ax1.plot(f_axis, E1_s, label='Path 1 (f1, f2)', color='blue', alpha=0.6)
ax1.plot(f_axis, E2_s, label='Path 2 (f3, f4)', color='red', linestyle='--')
ax1.set_title("Frequency Domain Spectra (MHz)")
ax1.legend(loc='upper right')
ax1.set_xlim(-zoom_f, zoom_f)
max_E = max(np.max(E1_s), np.max(E2_s))
ax1.set_ylim(0, max_E * 1.2 if max_E > 0 else 1)
ax1.set_xlabel("Frequency (MHz)")
ax1.set_ylabel("Intensity")

# 子图2：符合概率 g(2) 曲线
ax2 = plt.subplot(2, 1, 2)
ax2.plot(tau_range, Pc_f, color='green', lw=1.2, label='$g^{(2)}(\\tau)$')
ax2.plot(tau, c_Pc, 'ro', markersize=6, label='Current $\\tau$')
ax2.set_title("Normalized Coincidence Probability $g^{(2)}(\\tau)$ [Fast Detector]")
ax2.set_xlim(-zoom_t, zoom_t)
ax2.set_ylim(0.0, 2.2)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Delay $\\tau$ (μs)")
ax2.set_ylabel("Coincidence Probability")
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)

# 显示当前 tau 处的符合概率数值
st.info(f"当前延迟 τ = {tau:.4f} μs 时，符合概率 $g^{(2)}$ = {c_Pc:.4f}")