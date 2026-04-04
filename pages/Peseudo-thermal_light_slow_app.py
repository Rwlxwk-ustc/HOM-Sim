import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 页面与字体设置
# ---------------------------------------------------------
# 同样为了防止 Streamlit 云端渲染中文图表时出现方块，图表内文字使用英文。
plt.rcParams['axes.unicode_minus'] = False 

# ---------------------------------------------------------
# 1. 物理计算核心 (赝热光模型)
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
    
    # ==========================================
    # 核心公式：赝热光 (引入强度涨落)
    # Pc = 1/4 * [(P1+P2)^2 + P1^2 + P2^2] - 1/2 * P1*P2 * |M|^2
    # ==========================================
    normalization_factor = 0.25 * (P1 + P2)**2
    Pc_curve = 0.25 * ((P1 + P2)**2 + P1**2 + P2**2) - 0.5 * (P1 * P2) * M_tau_sq
    Pc_final = Pc_curve / normalization_factor  # 归一化为 g(2)
    
    current_overlap = np.sum(product * np.exp(-1j * omega * tau) * d_omega)
    current_M = np.abs(current_overlap) / np.sqrt(P1 * P2)
    current_Pc = (0.25 * ((P1 + P2)**2 + P1**2 + P2**2) - 0.5 * (P1 * P2) * (current_M**2)) / normalization_factor
    
    return E1, E2_base, tau_range, Pc_final, current_Pc, current_M

# ---------------------------------------------------------
# 2. Streamlit UI 界面
# ---------------------------------------------------------
st.title("MM-HOM Pseudo-thermal Light")

# 三列布局：左侧控制光路1，中间控制光路2和全局参数，右侧画图
col_ctrl1, col_ctrl2, col_plot = st.columns([1, 1, 3])

with col_ctrl1:
    st.markdown("#### 光路 1 参数")
    Om1 = st.slider('梳间距 Ω1', 1.0, 20.0, 10.0, 0.5, key='pt_om1')
    Dw1 = st.slider('包络带宽 Δω1', 5.0, 50.0, 20.0, 1.0, key='pt_dw1')
    wc1 = st.slider('包络中心 ω1,c', -10.0, 10.0, 0.0, 0.5, key='pt_wc1')
    I1  = st.slider('强度 I1', 0.1, 2.0, 1.0, 0.1, key='pt_i1')

with col_ctrl2:
    st.markdown("#### 光路 2 参数")
    Om2 = st.slider('梳间距 Ω2', 1.0, 20.0, 10.0, 0.5, key='pt_om2')
    Dw2 = st.slider('包络带宽 Δω2', 5.0, 50.0, 20.0, 1.0, key='pt_dw2')
    wc2 = st.slider('包络中心 ω2,c', -10.0, 10.0, 0.0, 0.5, key='pt_wc2')
    I2  = st.slider('强度 I2', 0.1, 2.0, 1.0, 0.1, key='pt_i2')
    
    st.markdown("#### 全局参数")
    N   = st.slider('梳齿数 N', 1, 15, 5, 1, key='pt_n')
    tau = st.slider('延迟 τ', -1.0, 1.0, 0.0, 0.02, key='pt_tau')

# ---------------------------------------------------------
# 3. 数据获取与绘图
# ---------------------------------------------------------
E1, E2, t_r, Pc_f, c_Pc, c_M = get_plot_data(tau, N, Om1, Om2, Dw1, Dw2, wc1, wc2, I1, I2)

with col_plot:
    fig, (ax_spectra, ax_curve) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3)

    # 绘制光谱
    ax_spectra.plot(omega, np.abs(E1)**2, label='Path 1 Spectrum', color='blue', alpha=0.6)
    ax_spectra.plot(omega, np.abs(E2)**2, label='Path 2 Spectrum', color='red', linestyle='--')
    ax_spectra.set_title("Optical Frequency Combs")
    ax_spectra.set_ylim(0, max(np.max(np.abs(E1)**2), np.max(np.abs(E2)**2)) * 1.2 or 1)
    ax_spectra.legend()

    # 绘制 g(2) 曲线
    ax_curve.plot(t_r, Pc_f, color='orange', lw=1.5)
    ax_curve.plot([tau], [c_Pc], 'ro', markersize=8)
    
    # 动态适应 y 轴范围，完美展示赝热光的基线漂移
    y_min, y_max = min(Pc_f), max(Pc_f)
    padding = (y_max - y_min) * 0.2 if y_max != y_min else 0.1
    ax_curve.set_ylim(y_min - padding, y_max + padding)
    
    # 在标题中实时显示 M 因子
    ax_curve.set_title(f"Normalized Coincidence Probability g(2) [Pseudo-thermal] | Current |M|: {c_M:.4f}")
    ax_curve.grid(True, alpha=0.3)

    st.pyplot(fig)
