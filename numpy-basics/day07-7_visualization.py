import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

SAMPLE_RATE = 44100
DURATION = 1.0

def blit_impulse_train(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    nyquist = sample_rate / 2
    M = int(nyquist / freq)
    phase = (freq * t) % 1.0
    epsilon = 1e-10
    denominator = M * np.sin(np.pi * phase + epsilon)
    numerator = np.sin(np.pi * M * phase)
    blit = numerator / (denominator + epsilon)
    blit = blit / M
    return blit, t, M

def blit_to_sawtooth(blit_signal, sample_rate):
    leak = 0.999
    saw = np.zeros_like(blit_signal)
    accumulator = 0
    for i in range(len(blit_signal)):
        accumulator = leak * accumulator + blit_signal[i]
        saw[i] = accumulator
    saw = saw / np.max(np.abs(saw))
    saw = 2 * saw - 1
    return saw

def polyblep_residual(t, dt):
    if t < dt:
        t = t / dt
        return t + t - t * t - 1.0
    elif t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    else:
        return 0.0

def polyblep_sawtooth(freq, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    dt = freq / sample_rate
    phase = 0
    output = np.zeros_like(t)
    for i in range(len(t)):
        naive_saw = 2 * phase - 1
        correction = polyblep_residual(phase, dt)
        output[i] = naive_saw - correction
        phase += dt
        if phase >= 1.0:
            phase -= 1.0
    return output, t

# ============================================
# 제대로 된 시각화
# ============================================
freq = 440

# Generate signals
blit, t, M = blit_impulse_train(freq, DURATION, SAMPLE_RATE)
saw_blit = blit_to_sawtooth(blit, SAMPLE_RATE)
saw_polyblep, _ = polyblep_sawtooth(freq, DURATION, SAMPLE_RATE)

# Naive
phase = (freq * t) % 1.0
saw_naive = 2 * phase - 1

# ============================================
# 올바른 시각화: 주기별로 보기
# ============================================
fig, axes = plt.subplots(3, 3, figsize=(14, 9))

# 한 주기 계산
period = 1 / freq  # 초
samples_per_period = int(SAMPLE_RATE * period)  # 샘플 수

print(f"Frequency: {freq} Hz")
print(f"Period: {period*1000:.3f} ms")
print(f"Samples per period: {samples_per_period}")

# 3가지 확대 수준
zoom_levels = [
    (1, "1 Period"),
    (3, "3 Periods"), 
    (10, "10 Periods")
]

for col_idx, (num_periods, title) in enumerate(zoom_levels):
    plot_samples = samples_per_period * num_periods
    t_plot = t[:plot_samples] * 1000  # ms
    
    # Naive
    axes[0, col_idx].plot(t_plot, saw_naive[:plot_samples], 
                         linewidth=1.5, color='red', alpha=0.8)
    axes[0, col_idx].set_ylabel('Amplitude', fontsize=9)
    axes[0, col_idx].set_title(f'Naive - {title}', fontsize=10, fontweight='bold')
    axes[0, col_idx].grid(True, alpha=0.3)
    axes[0, col_idx].set_ylim(-1.5, 1.5)
    
    # BLIT
    axes[1, col_idx].plot(t_plot, saw_blit[:plot_samples], 
                         linewidth=1.5, color='blue', alpha=0.8)
    axes[1, col_idx].set_ylabel('Amplitude', fontsize=9)
    axes[1, col_idx].set_title(f'BLIT - {title}', fontsize=10, fontweight='bold')
    axes[1, col_idx].grid(True, alpha=0.3)
    axes[1, col_idx].set_ylim(-1.5, 1.5)
    
    # PolyBLEP
    axes[2, col_idx].plot(t_plot, saw_polyblep[:plot_samples], 
                         linewidth=1.5, color='green', alpha=0.8)
    axes[2, col_idx].set_xlabel('Time (ms)', fontsize=9)
    axes[2, col_idx].set_ylabel('Amplitude', fontsize=9)
    axes[2, col_idx].set_title(f'PolyBLEP - {title}', fontsize=10, fontweight='bold')
    axes[2, col_idx].grid(True, alpha=0.3)
    axes[2, col_idx].set_ylim(-1.5, 1.5)

plt.suptitle(f'Time Domain Comparison ({freq} Hz)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# 불연속점 확대 (가장 중요!)
# ============================================
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# 불연속점 근처만 (10 샘플 전후)
discontinuity_idx = samples_per_period
zoom_range = 20  # 전후 20 샘플
start = discontinuity_idx - zoom_range
end = discontinuity_idx + zoom_range

t_zoom = t[start:end] * 1000

# Naive
axes[0].plot(t_zoom, saw_naive[start:end], 'o-', 
            linewidth=2, markersize=4, color='red', label='Naive')
axes[0].set_ylabel('Amplitude', fontsize=10)
axes[0].set_title('Naive: Sharp Jump (Aliasing!)', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(t[discontinuity_idx]*1000, color='black', 
               linestyle='--', alpha=0.5, label='Discontinuity')

# BLIT
axes[1].plot(t_zoom, saw_blit[start:end], 'o-', 
            linewidth=2, markersize=4, color='blue', label='BLIT')
axes[1].set_ylabel('Amplitude', fontsize=10)
axes[1].set_title('BLIT: Smooth (Band-Limited)', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(t[discontinuity_idx]*1000, color='black', 
               linestyle='--', alpha=0.5)

# PolyBLEP
axes[2].plot(t_zoom, saw_polyblep[start:end], 'o-', 
            linewidth=2, markersize=4, color='green', label='PolyBLEP')
axes[2].set_xlabel('Time (ms)', fontsize=10)
axes[2].set_ylabel('Amplitude', fontsize=10)
axes[2].set_title('PolyBLEP: Smoothed Discontinuity', fontsize=11, fontweight='bold')
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3)
axes[2].axvline(t[discontinuity_idx]*1000, color='black', 
               linestyle='--', alpha=0.5)

plt.suptitle('Zoomed: Discontinuity Detail', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================
# 주파수 도메인 (로그 스케일로 제대로 보기)
# ============================================
N = len(saw_blit)
fft_blit = fft(saw_blit)
fft_poly = fft(saw_polyblep)
fft_naive = fft(saw_naive)

freqs = fftfreq(N, 1/SAMPLE_RATE)
positive_freqs = freqs[:N//2]
mag_blit = np.abs(fft_blit[:N//2])
mag_poly = np.abs(fft_poly[:N//2])
mag_naive = np.abs(fft_naive[:N//2])

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Naive
axes[0].plot(positive_freqs, mag_naive, linewidth=0.5, color='red', alpha=0.7)
axes[0].set_xlim(0, SAMPLE_RATE / 2)
axes[0].set_xlabel('Frequency (Hz)', fontsize=10)
axes[0].set_ylabel('Magnitude', fontsize=10)
axes[0].set_title('Naive (Lots of Aliasing)', fontsize=11, fontweight='bold')
axes[0].set_yscale('log')
axes[0].axvline(SAMPLE_RATE / 2, color='black', linestyle='--', 
               linewidth=2, label='Nyquist')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# BLIT
axes[1].plot(positive_freqs, mag_blit, linewidth=0.5, color='blue', alpha=0.7)
axes[1].set_xlim(0, SAMPLE_RATE / 2)
axes[1].set_xlabel('Frequency (Hz)', fontsize=10)
axes[1].set_ylabel('Magnitude', fontsize=10)
axes[1].set_title('BLIT (Clean!)', fontsize=11, fontweight='bold')
axes[1].set_yscale('log')
axes[1].axvline(SAMPLE_RATE / 2, color='black', linestyle='--', linewidth=2)
axes[1].grid(True, alpha=0.3)

# PolyBLEP
axes[2].plot(positive_freqs, mag_poly, linewidth=0.5, color='green', alpha=0.7)
axes[2].set_xlim(0, SAMPLE_RATE / 2)
axes[2].set_xlabel('Frequency (Hz)', fontsize=10)
axes[2].set_ylabel('Magnitude', fontsize=10)
axes[2].set_title('PolyBLEP (Clean!)', fontsize=11, fontweight='bold')
axes[2].set_yscale('log')
axes[2].axvline(SAMPLE_RATE / 2, color='black', linestyle='--', linewidth=2)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Frequency Spectrum Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()