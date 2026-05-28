"""omega, alpha, f 이해하기

1. 주파수를 "원을 도는 속도"로 보는 것
        ↓
2. 왜 sample_rate로 나눠서 정규화하냐
        ↓
3. sin/cos이 필터 계수에 왜 재료로 쓰이냐
        ↓
4. omega, alpha, f 이해


    - Cut off(Hz) -> omega(radian) 의 변환은 선형관계


"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import signal

SAMPLE_RATE = 44100

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
plt.subplots_adjust(bottom=0.25, hspace=0.4, wspace=0.35)

# --- Sliders ---
ax_cutoff = plt.axes([0.15, 0.13, 0.7, 0.03])
ax_q      = plt.axes([0.15, 0.08, 0.7, 0.03])
sl_cutoff = Slider(ax_cutoff, 'Cutoff (Hz)', 100, 10000, valinit=1000)
sl_q      = Slider(ax_q,      'Q',           0.1, 20,    valinit=1.0)

def compute(cutoff, Q):
    omega     = 2 * np.pi * cutoff / SAMPLE_RATE
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    alpha     = sin_omega / (2 * Q)
    f = 2 * np.sin(np.pi * cutoff / SAMPLE_RATE)
    q = 1 / Q
    return omega, sin_omega, cos_omega, alpha, f, q

def draw(cutoff, Q):
    omega, sin_omega, cos_omega, alpha, f, q = compute(cutoff, Q)

    for ax in axes.flat:
        ax.cla()

    # ── 1. omega on unit circle ──────────────────────────
    ax = axes[0, 0]
    theta = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='#b4b2a9', lw=1)
    ax.axhline(0, color='#b4b2a9', lw=0.5)
    ax.axvline(0, color='#b4b2a9', lw=0.5)
    px, py = np.cos(omega), np.sin(omega)
    ax.plot([0, px], [0, py], color='#888780', lw=1.5)
    ax.plot([0, px], [0, 0],  color='#378ADD', lw=3, label=f'cos={cos_omega:.3f}')
    ax.plot([px, px],[0, py], color='#378ADD', lw=1.5, ls='--')
    ax.plot([0, 0],  [0, py], color='#D85A30', lw=3, label=f'sin={sin_omega:.3f}')
    ax.plot([0, px], [py,py], color='#D85A30', lw=1.5, ls='--')
    ax.plot(px, py, 'o', color='#2C2C2A', ms=8, zorder=5)
    ax.set_xlim(-1.3, 1.3)   # fixed
    ax.set_ylim(-1.3, 1.3)   # fixed
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title(f'omega on unit circle\nomega = {omega:.4f} rad', fontsize=10)

    # ── 2. omega vs cutoff relationship ─────────────────
    ax = axes[0, 1]
    freqs = np.linspace(0, SAMPLE_RATE/2, 500)
    omegas = 2 * np.pi * freqs / SAMPLE_RATE
    ax.plot(freqs, omegas, color='#378ADD', lw=2)
    ax.axvline(cutoff, color='#D85A30', lw=1.5, ls='--', label=f'cutoff={cutoff:.0f}Hz')
    ax.axhline(omega,  color='#D85A30', lw=1.5, ls='--')
    ax.plot(cutoff, omega, 'o', color='#D85A30', ms=8, zorder=5)
    ax.set_xlabel('Cutoff (Hz)')
    ax.set_ylabel('omega (rad)')
    ax.set_title('cutoff to omega conversion', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, SAMPLE_RATE/2)  # fixed
    ax.set_ylim(0, np.pi)          # fixed

    # ── 3. alpha vs Q ────────────────────────────────────
    ax = axes[0, 2]
    q_range = np.linspace(0.1, 20, 300)
    alphas  = sin_omega / (2 * q_range)
    ax.plot(q_range, alphas, color='#7F77DD', lw=2)
    ax.axvline(Q,     color='#D85A30', lw=1.5, ls='--', label=f'Q={Q:.1f}')
    ax.axhline(alpha, color='#D85A30', lw=1.5, ls='--', label=f'alpha={alpha:.4f}')
    ax.plot(Q, alpha, 'o', color='#D85A30', ms=8, zorder=5)
    ax.set_xlabel('Q')
    ax.set_ylabel('alpha')
    ax.set_title(f'alpha = sin(omega) / (2*Q)\nalpha={alpha:.4f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)    # fixed
    ax.set_ylim(0, 0.6)   # fixed

    # ── 4. SVF f vs Biquad sin_omega ─────────────────────
    ax = axes[1, 0]
    freqs2     = np.linspace(0, SAMPLE_RATE/2, 500)
    f_vals     = 2 * np.sin(np.pi * freqs2 / SAMPLE_RATE)
    sin_vals   = np.sin(2 * np.pi * freqs2 / SAMPLE_RATE)
    omega_vals = 2 * np.pi * freqs2 / SAMPLE_RATE
    ax.plot(freqs2, omega_vals / np.pi, color='#b4b2a9', lw=1.5, ls='--', label='omega/pi (no correction)')
    ax.plot(freqs2, sin_vals,           color='#378ADD',  lw=2,   label='sin(omega) - Biquad')
    ax.plot(freqs2, f_vals,             color='#D85A30',  lw=2,   label='2*sin(pi*f/sr) - SVF f')
    ax.axvline(cutoff, color='#7F77DD', lw=1.5, ls='--')
    ax.set_xlabel('Cutoff (Hz)')
    ax.set_ylabel('value')
    ax.set_title('SVF f vs Biquad sin_omega\n(why sin is used at high freq)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, SAMPLE_RATE/2)  # fixed
    ax.set_ylim(0, 2.1)            # fixed

    # ── 5. Frequency response - Lowpass ──────────────────
    ax = axes[1, 1]
    b0 = (1 - cos_omega) / 2
    b1 = 1 - cos_omega
    b2 = (1 - cos_omega) / 2
    a0 = 1 + alpha
    a1 = -2 * cos_omega
    a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1/a0, a2/a0])
    w, h = signal.freqz(b, a, worN=4000, fs=SAMPLE_RATE)
    ax.plot(w, 20*np.log10(np.abs(h)+1e-10), color='#378ADD', lw=2)
    ax.axvline(cutoff, color='#D85A30', lw=1.5, ls='--', label=f'cutoff={cutoff:.0f}Hz')
    ax.axhline(-3, color='#888780', lw=1, ls='--', label='-3dB')
    ax.set_xscale('log')
    ax.set_xlim(20, SAMPLE_RATE/2)  # fixed
    ax.set_ylim(-60, 20)            # fixed
    ax.set_xlabel('Hz')
    ax.set_ylabel('dB')
    ax.set_title(f'Biquad Lowpass frequency response\nQ={Q:.1f}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # ── 6. Q to bandwidth ────────────────────────────────
    ax = axes[1, 2]
    q_range2 = np.linspace(0.1, 20, 300)
    bw = cutoff / q_range2
    ax.plot(q_range2, bw, color='#1D9E75', lw=2)
    ax.axvline(Q,        color='#D85A30', lw=1.5, ls='--', label=f'Q={Q:.1f}')
    ax.axhline(cutoff/Q, color='#D85A30', lw=1.5, ls='--', label=f'BW={cutoff/Q:.0f}Hz')
    ax.plot(Q, cutoff/Q, 'o', color='#D85A30', ms=8, zorder=5)
    ax.set_xlabel('Q')
    ax.set_ylabel('bandwidth (Hz)')
    ax.set_title(f'Q to bandwidth\nBW = cutoff/Q = {cutoff/Q:.0f}Hz', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)      # fixed
    ax.set_ylim(0, 10000)   # fixed

    fig.canvas.draw_idle()

def update(val):
    draw(sl_cutoff.val, sl_q.val)

sl_cutoff.on_changed(update)
sl_q.on_changed(update)
draw(1000, 1.0)
plt.suptitle('Understanding omega / alpha / f — move the sliders', fontsize=12, y=0.98)
plt.show()