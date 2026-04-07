import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 设置页面布局较宽，适合图表展示
st.set_page_config(page_title="快探测器 HOM 干涉", layout="wide")

# (注意：部署在云端时，Linux 服务器可能没有本地中文字体，建议使用默认字体或在云端配置字体)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

W_MIN, W_MAX = -150.0, 150.0
omega_axis = np.linspace(W_MIN, W_MAX, 6000)

# ---------------------------------------------------------
# 1. 物理计算核心 (缓存全曲线数据以加速 UI 响应)
# ---------------------------------------------------------
@st.cache_data
def compute_fast_physics(N, Om1, Om2, w0_1, w0_2, Dw1, Dw2, wc1, wc2, I1, I2, dw1, dw2):
    # 构建模式索引 j
    j_indices = np.arange(N) - (N - 1) / 2.0
    
    w1_j = w0_1 + j_indices * Om1
    w2_k = w0_2 + j_indices * Om2
    
    # 光子数包络分布
    n1_j = I1 * np.exp(-(w1_j - wc1)**2 / (2 * Dw1**2))
    n2_k = I2 * np.exp(-(w2_k - wc2)**2 / (2 * Dw2**2))
    
    N1_total = np.sum(n1_j)
    N2_total = np.sum(n2_k)
    
    # 频域光谱计算 (应用独立的 dw1 和 dw2)
    E1_spec = np.zeros_like(omega_axis)
    E2_spec = np.zeros_like(omega_axis)
    for i in range(N):
        E1_spec += n1_j[i] * np.exp(-(omega_axis - w1_j[i])**2 / (2 * dw1**2))
        E2_spec += n2_k[i] * np.exp(-(omega_axis - w2_k[i])**2 / (2 * dw2**2))
    
    # 时域范围
    tau_range = np.linspace(-3.0, 3.0, 3000)
    
    # 向量化计算求和项
    S1 = np.sum(n1_j[:, None] * np.exp(-1j * w1_j[:, None] * tau_range), axis=0)
    S2 = np.sum(n2_k[:, None] * np.exp( 1j * w2_k[:, None] * tau_range), axis=0)
    
    # 快探测器时域干涉的全局衰减包络
    global_envelope = np.exp(-(dw1**2 + dw2**2) / 2.0 * tau_range**2)
    
    interference = 2 * np.real(S1 * S2) * global_envelope
    baseline = (N1_total + N2_total)**2
    
    Pc_final = 1.0 - interference / baseline
    
    return E1_spec, E2_spec, tau_range, Pc_final, n1_j, w1_j, n2_k, w2_k, baseline

# 单独计算当前的红点，不使用缓存，保证滑动 τ 时绝对平滑
def get_current_point(tau, n1_j, w1_j, n2_k, w2_k, baseline, dw1, dw2):
    c_S1 = np.sum(n1_j * np.exp(-1j * w1_j * tau))
    c_S2 = np.sum(n2_k * np.exp( 1j * w2_k * tau))
    c_envelope = np.exp(-(dw1**2 + dw2**2) / 2.0 * tau**2)
    c_interference = 2 * np.real(c_S1 * c_S2) * c_envelope
    return 1.0 - c_interference / baseline

# ---------------------------------------------------------
# 2. UI 界面布局
# ---------------------------------------------------------
st.title("多模频率梳干涉模拟 (快探测器实时响应模型)")
st.markdown("该模型包含了独立的子波包线宽控制，能够展示拍频包络包裹下的超快时间干涉图样。")

col1, col2, col3 = st.columns([1, 1, 1.5]) # 左中放滑块，右侧放图表（如果你想上下布局，把图表放到底部即可）

with col1:
    st.subheader("⚙️ 整体与光路 1 参数")
    tau = st.slider('延迟 τ', -3.0, 3.0, 0.0, 0.01)
    N = st.slider('梳齿数 N', 1, 31, 15, 1)
    Om1 = st.slider('间距 Ω1', 1.0, 30.0, 2.0, 0.5)
    w0_1 = st.slider('梳基频 ω1,0', -80.0, 80.0, 15.0, 1.0)
    Dw1 = st.slider('带宽 Δω1', 2.0, 80.0, 5.0, 1.0)
    wc1 = st.slider('中心 ω1,c', -80.0, 80.0, 0.0, 1.0)
    I1 = st.slider('强度 I1', 0.1, 2.0, 1.0, 0.1)
    dw1 = st.slider('光路 1 线宽 δω1', 0.1, 5.0, 0.8, 0.1)

with col2:
    st.subheader("⚙️ 光路 2 参数")
    st.write(" ") # 占位对齐
    st.write(" ")
    Om2 = st.slider('间距 Ω2', 1.0, 30.0, 2.0, 0.5)
    w0_2 = st.slider('梳基频 ω2,0', -80.0, 80.0, -15.0, 1.0)
    Dw2 = st.slider('带宽 Δω2', 2.0, 80.0, 5.0, 1.0)
    wc2 = st.slider('中心 ω2,c', -80.0, 80.0, 0.0, 1.0)
    I2 = st.slider('强度 I2', 0.1, 2.0, 1.0, 0.1)
    dw2 = st.slider('光路 2 线宽 δω2', 0.1, 5.0, 0.8, 0.1)

# ---------------------------------------------------------
# 3. 计算与绘图
# ---------------------------------------------------------
# 获取全曲线数据
E1_s, E2_s, t_r, Pc_f, n1_j, w1_j, n2_k, w2_k, baseline = compute_fast_physics(
    N, Om1, Om2, w0_1, w0_2, Dw1, Dw2, wc1, wc2, I1, I2, dw1, dw2
)
# 获取当前延迟点的红点数据
c_Pc = get_current_point(tau, n1_j, w1_j, n2_k, w2_k, baseline, dw1, dw2)

with col3:
    fig, (ax_spectra, ax_curve) = plt.subplots(2, 1, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.3)

    # 绘制频域
    ax_spectra.plot(omega_axis, E1_s, label='Path 1 Spectrum', color='#1f77b4', alpha=0.8)
    ax_spectra.plot(omega_axis, E2_s, label='Path 2 Spectrum', color='#ff7f0e', linestyle='--', alpha=0.8)
    ax_spectra.set_title("Frequency Domain Spectra")
    ax_spectra.set_ylim(0, max(np.max(E1_s), np.max(E2_s)) * 1.2 or 1)
    ax_spectra.legend()
    ax_spectra.grid(True, alpha=0.2)

    # 绘制时域
    ax_curve.plot(t_r, Pc_f, color='#2ca02c', lw=1.2)
    ax_curve.plot([tau], [c_Pc], 'ro', markersize=6)
    ax_curve.set_ylim(0.0, 2.2) # 快探测器有拍频，上限会超过 1，设定为 2.2
    ax_curve.set_title("Normalized Coincidence Probability $g^{(2)}(\\tau)$ [Fast Detector]")
    ax_curve.grid(True, alpha=0.3)

    st.pyplot(fig)
