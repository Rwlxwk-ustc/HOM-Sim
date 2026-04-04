import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 页面与字体设置
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="HOM Interference")

# 注意：Streamlit 云端服务器通常没有预装中文字体。
# 为了保证组会展示时图表上不出现乱码（方块），这里的图表标题和图例替换为了英文。
# 网页侧边栏/控件的中文不受影响，可以正常显示。
plt.rcParams['axes.unicode_minus'] = False 

# ---------------------------------------------------------
# 1. 物理计算核心 (快探测器解析模型)
# ---------------------------------------------------------
W_MIN, W_MAX = -150.0, 150.0
omega_axis = np.linspace(W_MIN, W_MAX, 6000)
delta_w = 0.8  

def get_fast_plot_data(tau, N, Om1, Om2, w0_1, w0_2, Dw1, Dw2, wc1, wc2, I1, I2):
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
        E1_spec += n1_j[i] * np.exp(-(omega_axis - w1_j[i])**2 / (2 * delta_w**2))
        E2_spec += n2_k[i] * np.exp(-(omega_axis - w2_k[i])**2 / (2 * delta_w**2))
    
    tau_range = np.linspace(-3.0, 3.0, 3000)
    
    S1 = np.sum(n1_j[:, None] * np.exp(-1j * w1_j[:, None] * tau_range), axis=0)
    S2 = np.sum(n2_k[:, None] * np.exp( 1j * w2_k[:, None] * tau_range), axis=0)
    
    interference = 2 * np.real(S1 * S2)
    baseline = (N1_total + N2_total)**2
    
    Pc_final = 1.0 - interference / baseline
    
    c_S1 = np.sum(n1_j * np.exp(-1j * w1_j * tau))
    c_S2 = np.sum(n2_k * np.exp( 1j * w2_k * tau))
    c_interference = 2 * np.real(c_S1 * c_S2)
    current_Pc = 1.0 - c_interference / baseline
    
    return E1_spec, E2_spec, tau_range, Pc_final, current_Pc

# ---------------------------------------------------------
# 2. Streamlit UI 界面
# ---------------------------------------------------------
st.title("多模 HOM 干涉模拟 (快探测器响应)")

# 使用 3 列布局：左列滑块，中列滑块，右列图表
col_controls_1, col_controls_2, col_plot = st.columns([1, 1, 3])

with col_controls_1:
    st.markdown("#### 光路 1 参数")
    w0_1 = st.slider('基频 ω1,0', -80.0, 80.0, 0.0, 1.0)
    Om1  = st.slider('梳间距 Ω1', 1.0, 30.0, 10.0, 0.5)
    wc1  = st.slider('包络中心 ω1,c', -80.0, 80.0, 0.0, 1.0)
    Dw1  = st.slider('包络带宽 Δω1', 5.0, 80.0, 30.0, 1.0)
    I1   = st.slider('强度 I1', 0.1, 2.0, 1.0, 0.1)

    st.markdown("#### 全局参数")
    N    = st.slider('梳齿数 N', 1, 21, 5, 1)
    tau  = st.slider('延迟 τ', -3.0, 3.0, 0.0, 0.05)

with col_controls_2:
    st.markdown("#### 光路 2 参数")
    w0_2 = st.slider('基频 ω2,0', -80.0, 80.0, 0.0, 1.0)
    Om2  = st.slider('梳间距 Ω2', 1.0, 30.0, 10.0, 0.5)
    wc2  = st.slider('包络中心 ω2,c', -80.0, 80.0, 0.0, 1.0)
    Dw2  = st.slider('包络带宽 Δω2', 5.0, 80.0, 30.0, 1.0)
    I2   = st.slider('强度 I2', 0.1, 2.0, 1.0, 0.1)

# 获取绘图数据
E1_s, E2_s, t_r, Pc_f, c_Pc = get_fast_plot_data(
    tau, N, Om1, Om2, w0_1, w0_2, Dw1, Dw2, wc1, wc2, I1, I2
)

# ---------------------------------------------------------
# 3. 绘图与渲染
# ---------------------------------------------------------
with col_plot:
    fig, (ax_spectra, ax_curve) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3)

    # 绘制光谱
    ax_spectra.plot(omega_axis, E1_s, label='Path 1 Spectrum', color='blue', alpha=0.6)
    ax_spectra.plot(omega_axis, E2_s, label='Path 2 Spectrum', color='red', linestyle='--')
    ax_spectra.set_title("Frequency Domain Spectra (Decoupled Comb & Envelope)")
    max_E = max(np.max(E1_s), np.max(E2_s))
    ax_spectra.set_ylim(0, max_E * 1.2 or 1)
    ax_spectra.legend(loc='upper right')

    # 绘制 g(2) 曲线
    ax_curve.plot(t_r, Pc_f, color='green', lw=1.2)
    ax_curve.plot([tau], [c_Pc], 'ro', markersize=8)
    ax_curve.set_ylim(0.0, 2.2) # 保持大范围展示拍频
    ax_curve.set_title("Normalized Coincidence Probability g(2) [Fast Detector]")
    ax_curve.grid(True, alpha=0.3)

    st.pyplot(fig)