import numpy as np
import matplotlib.pyplot as plt

N = 2000
t = np.linspace(-1, 1, N)

y = np.zeros_like(t)

M = 100
    # 이 값을 올릴수록 impulse 처럼 형태가 만들어짐 
    # 유한한 개수의 사인파를 더한 결과는 엄밀히 말하면 
    # 디지털 impulse 가 아니라 Dirichlet kernel 임 
    # 사인파 개수를 무한대로 보내면 이상적인 impulse(정확히는 주기적인 delta 열)

for k in range(-M, M + 1):
    y += np.cos(2 * np.pi * k * t)

plt.plot(t, y)
plt.grid(True)
plt.show()