# BLIT 를 알기위한 사전 지식
    # 1. Sinc Function
    # 2. Band-limiting 에 sinc 가 왜 필요한지
    # 3. 복소수

""" Sinc function

sinc(x) = sin(πx) / (πx)

    : 이상적인 low pass filter 의 시간 응답 = sinc 
    : h(t) = sinc(2πfc * t)

"""


import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 1000)
sinc = np.sinc(x)  # numpy는 이미 sinc 제공

plt.figure(figsize=(10, 5))
plt.plot(x, sinc, linewidth=2)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='red', linestyle='--', alpha=0.5)
plt.title('Sinc Function')
plt.xlabel('x')
plt.ylabel('sinc(x)')
plt.grid(True, alpha=0.3)
plt.show()

"""

"""