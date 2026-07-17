"""
==============================================
DAY 20: Sampling Theory & Sample Rate Conversion
==============================================

MATHEMATICAL PREREQUISITES:
- Nyquist theorem (나이퀴스트 정리)
- Spectrum folding (스펙트럼 접힘)
- Interpolation/decimation concepts

KEY CONCEPT:
You can't represent frequencies above Nyquist!
This is a fundamental limit of digital audio.
(나이퀴스트 주파수 이상은 표현 불가)

Solution: Anti-aliasing filters, oversampling, proper resampling
Different sample rates need different conversion techniques

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_nyquist_theorem():
    # what is the Nyquist theorem?
    """
    [THEOREM] Nyquist-Shannon sampling theorem

    STATEMENT(정리)

    fs >= 2 * f_max

        - where fs = sampling frequency 
        - f_max = highest frequency in signal

    CD quality: 44.1kHz sample rate
        - can represent up to 22.05kHz (nyquist freq)
        - Humans hear up to ~ 20 kHz 
    HD audio: 96 kHz sample rate
        - can represent up to 48kHz 
        - way beyond human hearing

    **Aliasing

    Nyquist freq = 22.05kHz 일때,
    signal contains : 25kHz 를 포함하고 있다면
    what you hear: 44.1 - 25 = 19.1 kHz
    => 25kHz 의 신호가 19.1kHz 로 들리게 됨


    sin(2pi*(sample_rate - f)*t)
        => 이 주파수에서 들림

        ex. sample rate 48000, f=47000
            => 들리는 주파수는 1000Hz : aliasing freq
    
    """

def visualize_aliasing():
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
                             
    fs = 44100

    # case 1: safe frequency
    t = np.linspace(0, 0.001, 1000)
        # 10ms 까지를 1000개로 나눠라 -> 근데 너무 촘촘해보여서
        # 1ms (0.001) 로 바꿈
    f_safe = 5000 # 5000Hz 는 안전
    signal_safe = np.sin(2*np.pi*f_safe*t)

    ax = axes[0, 0]
    ax.plot(t*1000, signal_safe, 'b-', linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Safe: {f_safe} Hz signal')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.axis('off')
    ax.text(0.1, 0.5, f'Sampling at {fs} Hz\n→ Can represent {f_safe} Hz\n✓ NO ALIASING', 
           fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # case 2: danger zone (near nyquist)
    f_danger = 20000
    signal_danger = np.sin(2*np.pi*f_danger*t)

    ax = axes[1, 0]
    ax.plot(t*1000, signal_danger, 'orange', linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Danger: {f_danger} Hz signal')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.1, 0.5, f'Very close to Nyquist!\n→ Still representable\nBut risky', 
           fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # case 3: aliasing (Above nyquist)
    f_alias = 25000
    f_alias_perceived = fs - f_alias # 19.1kHz 에서 aliasing 발생

    signal_alias_true = np.sin(2*np.pi*f_alias*t)
    signal_alias_perceived = np.sin(2*np.pi*f_alias_perceived*t)

    ax = axes[2, 0]
    ax.plot(t*1000, signal_alias_true, 'b-', linewidth=1, alpha=0.5, label='True (25kHz)')
    ax.plot(t*1000, signal_alias_perceived, 'r-', linewidth=2, label=f'What we hear ({f_alias_perceived:.0f}Hz)')

    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'ALIASING: {f_alias} Hz → {f_alias_perceived:.0f} Hz')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


    ax = axes[2, 1]
    ax.axis('off')
    ax.text(0.1, 0.5, f'{f_alias} Hz FOLDS BACK\n→ Becomes {f_alias_perceived:.0f} Hz\n✗ UNWANTED ARTIFACT!', 
           fontsize=11, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

visualize_aliasing()