import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# 数据
energies = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200])
g_Ei = np.array([10.6, 16.0, 45.0, 83.5, 52.8, 19.9, 10.8, 8.25, 4.7])
errors = np.array([9.34, 17.9, 41.5, 85.5, 51.5, 21.5, 10.8, 6.29, 4.14])

# 任务1：拉格朗日多项式插值
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    p = np.zeros_like(x)
    for i in range(n):
        L = np.ones_like(x)
        for j in range(n):
            if i != j:
                L *= (x - x_points[j]) / (x_points[i] - x_points[j])
        p += y_points[i] * L
    return p

x_new = np.linspace(0, 200, 400)
y_lagrange = lagrange_interpolation(x_new, energies, g_Ei)

# 任务2：三次样条插值
cs = CubicSpline(energies, g_Ei)
y_cubic_spline = cs(x_new)

# 任务3：共振峰分析
def peak_analysis(y_values, x_values):
    max_value = np.max(y_values)
    max_index = np.argmax(y_values)
    peak_position = x_values[max_index]
    half_max = max_value / 2
    fwhm_left = np.max(np.where(y_values <= half_max)[0])
    fwhm_right = np.min(np.where(y_values <= half_max)[0]) + len(y_values)
    fwhm_right = x_values[fwhm_right]
    fwhm_left = x_values[fwhm_left]
    fwhm = fwhm_right - fwhm_left
    return peak_position, fwhm

# 插值结果比较
lagrange_peak, lagrange_fwhm = peak_analysis(y_lagrange, x_new)
spline_peak, spline_fwhm = peak_analysis(y_cubic_spline, x_new)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(energies, g_Ei, 'o', label='原始数据')
plt.plot(x_new, y_lagrange, '-', label='拉格朗日插值')
plt.plot(x_new, y_cubic_spline, '--', label='三次样条插值')
plt.axvline(lagrange_peak, color='r', linestyle='-', label=f'拉格朗日峰值: {lagrange_peak:.2f} MeV')
plt.axvline(spline_peak, color='g', linestyle='--', label=f'三次样条峰值: {spline_peak:.2f} MeV')
plt.xlabel('能量 (MeV)')
plt.ylabel('截面 g(Ei) (mb)')
plt.legend()
plt.title('插值结果比较')
plt.show()

# 输出共振峰分析结果
print(f"拉格朗日插值峰值位置: {lagrange_peak:.2f} MeV, FWHM: {lagrange_fwhm:.2f} MeV")
print(f"三次样条插值峰值位置: {spline_peak:.2f} MeV, FWHM: {spline_fwhm:.2f} MeV")

    plt.axvline(spline_peak, color='orange', linestyle=':', alpha=0.5)
    
    # 图表装饰
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Cross Section (mb)')
    plt.title('Neutron Resonance Scattering Cross Section Analysis')
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    plot_results()
