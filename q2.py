import numpy as np
import matplotlib.pyplot as plt

# 定義 f1 和 f2
def f1(t, u1, u2):
    return 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)

def f2(t, u1, u2):
    return -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)

# 精確解
def exact_u1(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

# Runge-Kutta 4th order for system
def runge_kutta_system(h, t_end):
    n = int(t_end / h)
    t = np.linspace(0, t_end, n+1)
    
    u1 = np.zeros(n+1)
    u2 = np.zeros(n+1)

    # 初始條件
    u1[0] = 4/3
    u2[0] = 2/3

    for i in range(n):
        k1 = h * f1(t[i], u1[i], u2[i])
        l1 = h * f2(t[i], u1[i], u2[i])
        
        k2 = h * f1(t[i] + h/2, u1[i] + k1/2, u2[i] + l1/2)
        l2 = h * f2(t[i] + h/2, u1[i] + k1/2, u2[i] + l1/2)
        
        k3 = h * f1(t[i] + h/2, u1[i] + k2/2, u2[i] + l2/2)
        l3 = h * f2(t[i] + h/2, u1[i] + k2/2, u2[i] + l2/2)
        
        k4 = h * f1(t[i] + h, u1[i] + k3, u2[i] + l3)
        l4 = h * f2(t[i] + h, u1[i] + k3, u2[i] + l3)

        u1[i+1] = u1[i] + (k1 + 2*k2 + 2*k3 + k4)/6
        u2[i+1] = u2[i] + (l1 + 2*l2 + 2*l3 + l4)/6

    return t, u1, u2

# 設定
t_end = 1.0  # 計算到t=1
h_values = [0.05, 0.1]

for h in h_values:
    print(f"\n=== Results for h = {h} ===\n")
    t, u1_approx, u2_approx = runge_kutta_system(h, t_end)
    
    u1_exact = exact_u1(t)
    u2_exact = exact_u2(t)
    
    # 顯示結果
    print(f"{'t':>6} {'u1 approx':>12} {'u1 exact':>12} {'u1 error':>12} {'u2 approx':>12} {'u2 exact':>12} {'u2 error':>12}")
    for i in range(len(t)):
        u1_error = abs(u1_approx[i] - u1_exact[i])
        u2_error = abs(u2_approx[i] - u2_exact[i])
        print(f"{t[i]:6.2f} {u1_approx[i]:12.6f} {u1_exact[i]:12.6f} {u1_error:12.2e} {u2_approx[i]:12.6f} {u2_exact[i]:12.6f} {u2_error:12.2e}")

    # 畫圖
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, u1_approx, 'o-', label="u1 approx")
    plt.plot(t, u1_exact, '-', label="u1 exact")
    plt.xlabel("t")
    plt.ylabel("u1")
    plt.title(f"u1 vs t (h = {h})")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(t, u2_approx, 'o-', label="u2 approx")
    plt.plot(t, u2_exact, '-', label="u2 exact")
    plt.xlabel("t")
    plt.ylabel("u2")
    plt.title(f"u2 vs t (h = {h})")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
