import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 0. 中文字体兼容性设置
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1. 物理计算核心 (慢探测器解析模型)
# ---------------------------------------------------------
W_MIN, W_MAX = -100.0, 100.0
omega_axis = np.linspace(W_MIN, W_MAX, 5000)
delta_w = 0.8  # 子波包带宽（固定）

def get_slow_plot_data(tau, N, Omega, Dw1, Dw2, wc1, wc2, I1, I2, dw0):
    # 构建模式索引
    j_indices = np.arange(N) - (N - 1) / 2.0

    # 频率梳中心频率 (引入整体失配 dw0)
    w1_j = dw0 / 2.0 + j_indices * Omega
    w2_j = -dw0 / 2.0 + j_indices * Omega

    # 根据泊松分布统计得到的平均光子数分布 (高斯包络)
    n1_j = I1 * np.exp(-(w1_j - wc1)**2 / (2 * Dw1**2))
    n2_j = I2 * np.exp(-(w2_j - wc2)**2 / (2 * Dw2**2))

    N1_total = np.sum(n1_j)
    N2_total = np.sum(n2_j)

    # 可视化频域光谱 (用于上方图表展示)
    E1_spec = np.zeros_like(omega_axis)
    E2_spec = np.zeros_like(omega_axis)
    for j in range(N):
        E1_spec += n1_j[j] * np.exp(-(omega_axis - w1_j[j])**2 / (2 * delta_w**2))
        E2_spec += n2_j[j] * np.exp(-(omega_axis - w2_j[j])**2 / (2 * delta_w**2))

    # 计算慢探测器 g(2) 曲线
    tau_range = np.linspace(-5.0, 5.0, 1000)

    # 解析公式干涉项: exp(-dw0^2 / 2 dw^2) * exp(-dw^2 tau^2 / 2) * sum(n1_j * n2_j)
    overlap_sum = np.sum(n1_j * n2_j)
    mismatch_factor = np.exp(-dw0**2 / (2 * delta_w**2))

    interference = 0.5 * mismatch_factor * np.exp(-delta_w**2 * tau_range**2 / 2) * overlap_sum
    baseline = 0.25 * (N1_total + N2_total)**2

    Pc_curve = baseline - interference
    Pc_final = Pc_curve / baseline  # 归一化

    # 计算当前 tau 点的 g(2)
    current_interference = 0.5 * mismatch_factor * np.exp(-delta_w**2 * tau**2 / 2) * overlap_sum
    current_Pc = (baseline - current_interference) / baseline

    return E1_spec, E2_spec, tau_range, Pc_final, current_Pc

# ---------------------------------------------------------
# 2. Streamlit 界面
# ---------------------------------------------------------
st.set_page_config(layout="wide", page_title="慢探测器符合概率模拟器")
st.title("慢探测器符合概率 $g^{(2)}(\\tau)$ 模拟器")

# ---------- 侧边栏：所有控制参数 ----------
st.sidebar.header("参数调节面板")

with st.sidebar.expander("显示范围调节", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        zoom_w = st.slider("光谱 X 轴范围 ±", 10.0, 100.0, 100.0, 1.0, format="%.1f")
    with col2:
        zoom_t = st.slider("时域 X 轴范围 ±", 0.5, 5.0, 5.0, 0.1, format="%.2f")

with st.sidebar.expander("核心时延参数", expanded=True):
    tau = st.slider("延迟 τ", -5.0, 5.0, 0.0, 0.01, format="%.3f")

with st.sidebar.expander("光路1 参数 (固定部分)", expanded=False):
    st.markdown("- 中心频率 ω₁꜀ = **0.0** (固定)")
    st.markdown("- 强度 I₁ = **1.0** (固定)")
    Dw1 = st.slider("带宽 Δω₁", 5.0, 50.0, 20.0, 1.0)

with st.sidebar.expander("光路2 参数", expanded=False):
    I2 = st.slider("强度 I₂", 0.1, 2.0, 1.0, 0.05)
    Dw2 = st.slider("带宽 Δω₂", 5.0, 50.0, 20.0, 1.0)
    wc2 = st.slider("包络中心 ω₂꜀", -20.0, 20.0, 0.0, 0.5)

with st.sidebar.expander("梳齿与失配参数", expanded=False):
    N = st.slider("梳齿数 N", 1, 15, 7, 1)
    Omega = st.slider("间距 Ω", 1.0, 20.0, 10.0, 0.5)
    dw0 = st.slider("整体失配 Δω₀", -5.0, 5.0, 0.0, 0.1)

# 固定参数（与原始UI一致）
wc1 = 0.0
I1 = 1.0

# ---------- 计算数据 ----------
E1_s, E2_s, t_r, Pc_f, c_Pc = get_slow_plot_data(
    tau, N, Omega, Dw1, Dw2, wc1, wc2, I1, I2, dw0
)

# ---------- 绘图 ----------
fig = plt.figure(figsize=(12, 8))

# 子图1：频谱
ax1 = plt.subplot(2, 1, 1)
ax1.plot(omega_axis, E1_s, label='光路 1 光谱 (平均光子数)', color='blue', alpha=0.6)
ax1.plot(omega_axis, E2_s, label='光路 2 光谱 (平均光子数)', color='red', linestyle='--')
ax1.set_title("频域光谱展示")
ax1.legend(loc='upper right')
ax1.set_xlim(-zoom_w, zoom_w)
max_E = max(np.max(E1_s), np.max(E2_s))
ax1.set_ylim(0, max_E * 1.2 if max_E > 0 else 1)
ax1.set_xlabel("频率 (任意单位)")
ax1.set_ylabel("强度")

# 子图2：符合概率曲线
ax2 = plt.subplot(2, 1, 2)
ax2.plot(t_r, Pc_f, color='purple', lw=1.5, label="$g^{(2)}(\\tau)$")
ax2.plot(tau, c_Pc, 'ro', markersize=6, label="当前 τ 位置")
ax2.set_title("归一化符合概率 $g^{(2)}(\\tau)$ [慢探测器]")
ax2.set_xlim(-zoom_t, zoom_t)
ax2.set_ylim(0.45, 1.1)
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("延迟 τ (时间单位)")
ax2.set_ylabel("符合概率")
ax2.legend(loc='upper right')

plt.tight_layout()
st.pyplot(fig)

# 显示当前 τ 处的符合概率数值
st.info(f"当前延迟 τ = {tau:.3f} 时，符合概率 $g^{(2)}$ = {c_Pc:.4f}")
