import numpy as np
import matplotlib.pyplot as plt

# 최종 비교 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sample_rate = 1000
freq = 10
t = np.linspace(0, 0.5, sample_rate//2, endpoint=False)
period_samples = sample_rate // freq
impulse = np.zeros(sample_rate//2)
impulse[::period_samples] = 1

# 1. No Leak (leak = 1.0)
saw_no_leak = np.zeros_like(impulse)
acc = 0
for i in range(len(impulse)):
    acc = 1.0 * acc + impulse[i]
    saw_no_leak[i] = acc

axes[0, 0].stem(t[:100] * 1000, impulse[:100], basefmt=' ')
axes[0, 0].set_ylabel('Impulse', fontsize=9)
axes[0, 0].set_title('Input: Impulse Train', fontsize=10, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t * 1000, saw_no_leak, linewidth=2, color='red')
axes[0, 1].set_title('No Leak (leak=1.0) → Steps!', fontsize=10, fontweight='bold')
axes[0, 1].set_ylabel('Output', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].text(0.5, 0.5, 'Goes up\nNever down!',
               transform=axes[0, 1].transAxes,
               fontsize=10, color='red', ha='center',
               bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

# 2. With Leak (leak = 0.98)
saw_with_leak = np.zeros_like(impulse)
acc = 0
leak = 0.98
for i in range(len(impulse)):
    acc = leak * acc + impulse[i]
    saw_with_leak[i] = acc

saw_normalized = saw_with_leak / np.max(saw_with_leak)
saw_normalized = 2 * saw_normalized - 1

axes[1, 0].stem(t[:100] * 1000, impulse[:100], basefmt=' ')
axes[1, 0].set_xlabel('Time (ms)', fontsize=9)
axes[1, 0].set_ylabel('Impulse', fontsize=9)
axes[1, 0].set_title('Input: Impulse Train (same)', fontsize=10, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t * 1000, saw_normalized, linewidth=2, color='green')
axes[1, 1].set_xlabel('Time (ms)', fontsize=9)
axes[1, 1].set_ylabel('Output', fontsize=9)
axes[1, 1].set_title(f'With Leak (leak={leak}) → Sawtooth!', fontsize=10, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].text(0.5, 0.5, 'Goes up\nLeaks down!',
               transform=axes[1, 1].transAxes,
               fontsize=10, color='green', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.suptitle('Leak Makes Sawtooth from Impulses', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()