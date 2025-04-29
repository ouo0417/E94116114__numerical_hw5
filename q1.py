import numpy as np
import matplotlib.pyplot as plt

# 定義 f(t, y)
def f(t, y):
    return 1 + (y/t) + (y/t)**2

# 計算 df/dt 用來給Taylor用
def df(t, y):
    # 先手推計算f'：這裡展開後的結果
    #偏微分f對t
    term1 = (-y)/(t**2) + 2*(y/t)*(-y)/(t**2)
    #偏微分f對y
    term2 = (1/t) + (2*y)/(t**2)
    return term1 + term2 * f(t, y)

# 精確解
def exact_solution(t):
    return t * np.tan(np.log(t))

# 初始條件
a = 1
b = 2
h = 0.1
n = int((b - a) / h)

# 建立t陣列
t = np.linspace(a, b, n+1)

# 初始化w
w_euler = np.zeros(n+1)
w_taylor2 = np.zeros(n+1)

# 初值
w_euler[0] = 0
w_taylor2[0] = 0

# Euler's Method
for i in range(n):
    w_euler[i+1] = w_euler[i] + h * f(t[i], w_euler[i])

# Taylor's Method of order 2
for i in range(n):
    w_taylor2[i+1] = w_taylor2[i] + h * f(t[i], w_taylor2[i]) + (h**2)/2 * df(t[i], w_taylor2[i])

# 精確解
y_exact = exact_solution(t)

# 印出比較結果
print(f"{'t':>5} {'Euler':>10} {'Taylor 2':>12} {'Exact':>12} {'Euler Error':>15} {'Taylor Error':>15} {'Euler %Err':>12} {'Taylor %Err':>12}")
for i in range(n+1):
    euler_error = abs(w_euler[i] - y_exact[i])
    taylor_error = abs(w_taylor2[i] - y_exact[i])
    euler_percent = (euler_error / abs(y_exact[i])) * 100 if y_exact[i] != 0 else 0
    taylor_percent = (taylor_error / abs(y_exact[i])) * 100 if y_exact[i] != 0 else 0
    print(f"{t[i]:5.2f} {w_euler[i]:10.5f} {w_taylor2[i]:12.5f} {y_exact[i]:12.5f} {euler_error:15.5e} {taylor_error:15.5e} {euler_percent:12.2f} {taylor_percent:12.2f}")

