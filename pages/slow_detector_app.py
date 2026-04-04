import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 设置页面布局较宽，适合图表展示
st.set_page_config(layout="wide")

# (注意：部署在云端时，Linux 服务器可能没有本地中文字体，建议使用默认字体或在云端配置字体)
plt.rcParams['axes.unicode_minus'] = False 

W_MIN, W_MAX = -100.0, 100.0
omega_axis = np.linspace(W_MIN, W_MAX, 5000)
delta_w = 0.8  

# 核心物理计算逻辑保持不变
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
    
    Pc_final = (baseline - interference) / baseline  
    current_interference = 0.5 * mismatch_factor * np.exp(-delta_w**2 * tau**2 / 2) * overlap_sum
    current_Pc = (baseline - current_interference) / baseline
    
    return E1_spec, E2_spec, tau_range, Pc_final, current_Pc

st.title("多模 HOM 干涉模拟 (慢探测器模型)")

# 使用 Streamlit 的侧边栏或分列布局来放置滑块
col1, col2 = st.columns(2)
with col1:
    tau = st.slider('延迟 τ', -5.0, 5.0, 0.0, 0.1)
    N = st.slider('梳齿数 N', 1, 15, 7, 1)
    Omega = st.slider('间距 Ω', 1.0, 20.0, 10.0, 0.5)
    dw0 = st.slider('整体失配 Δω0', -5.0, 5.0, 0.0, 0.1)

with col2:
    Dw1 = st.slider('带宽 Δω1', 5.0, 50.0, 20.0, 1.0)
    Dw2 = st.slider('带宽 Δω2', 5.0, 50.0, 20.0, 1.0)
    wc2 = st.slider('中心 ω2', -20.0, 20.0, 0.0, 1.0)
    I2 = st.slider('强度 I2', 0.1, 2.0, 1.0, 0.1)

# 获取数据
E1_s, E2_s, t_r, Pc_f, c_Pc = get_slow_plot_data(tau, N, Omega, Dw1, Dw2, 0.0, wc2, 1.0, I2, dw0)

# 绘图
fig, (ax_spectra, ax_curve) = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(hspace=0.4)

ax_spectra.plot(omega_axis, E1_s, label='Path 1 Spectrum', color='blue', alpha=0.6)
ax_spectra.plot(omega_axis, E2_s, label='Path 2 Spectrum', color='red', linestyle='--')
ax_spectra.set_title("Frequency Domain Spectra")
ax_spectra.set_ylim(0, max(np.max(E1_s), np.max(E2_s)) * 1.2 or 1)
ax_spectra.legend()

ax_curve.plot(t_r, Pc_f, color='purple', lw=1.5)
ax_curve.plot([tau], [c_Pc], 'ro')
ax_curve.set_ylim(0.45, 1.1)
ax_curve.set_title("Normalized Coincidence Probability g(2)")
ax_curve.grid(True, alpha=0.3)

# 将 matplotlib 图表输出到网页
st.pyplot(fig)