import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 页面与字体设置（支持中文）
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(layout="wide", page_title="赝热光 HOM 干涉模拟器")
st.title("多模 HOM 干涉模拟 (赝热光 / Pseudo‑thermal Light)")

# ---------------------------------------------------------
# 1. 物理计算核心（与原代码相同）
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
# 2. 侧边栏：所有控制参数（与之前两个程序风格一致）
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 3. 数据计算与绘图
# ---------------------------------------------------------
E1, E2, t_r, Pc_f, c_Pc, c_M = get_plot_data(tau, N, Om1, Om2, Dw1, Dw2, wc1, wc2, I1, I2)

# 创建图形
fig, (ax_spectra, ax_curve) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.3)

# 频谱图
ax_spectra.plot(omega, np.abs(E1)**2, label='光路 1 光谱', color='blue', alpha=0.6)
ax_spectra.plot(omega, np.abs(E2)**2, label='光路 2 光谱', color='red', linestyle='--')
ax_spectra.set_title("频域光谱展示")
ax_spectra.set_ylim(0, max(np.max(np.abs(E1)**2), np.max(np.abs(E2)**2)) * 1.2 or 1)
ax_spectra.legend()
ax_spectra.set_xlabel("频率 (任意单位)")
ax_spectra.set_ylabel("强度")

# g(2) 曲线
ax_curve.plot(t_r, Pc_f, color='orange', lw=1.5, label='$g^{(2)}(\\tau)$')
ax_curve.plot([tau], [c_Pc], 'ro', markersize=8, label='当前 τ 位置')
ax_curve.set_title(f"归一化符合概率 $g^{(2)}(\\tau)$ [赝热光]  | 当前 |M| = {c_M:.4f}")
ax_curve.set_xlabel("延迟 τ (时间单位)")
ax_curve.set_ylabel("符合概率")
ax_curve.grid(True, alpha=0.3)
ax_curve.legend()

# 动态 y 轴范围（保留一定边距）
y_min, y_max = np.min(Pc_f), np.max(Pc_f)
padding = (y_max - y_min) * 0.2 if y_max != y_min else 0.1
ax_curve.set_ylim(y_min - padding, y_max + padding)

st.pyplot(fig)

# 显示当前 τ 处的符合概率数值
st.info(f"当前延迟 τ = {tau:.3f} 时，符合概率 $g^{(2)}$ = {c_Pc:.4f}，模重叠度 |M| = {c_M:.4f}")
