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

def understand_anti_aliaisng():
    # How to prevent aliasing

    """
    Signal contains frequencies > Nyquist
    They will alias into audible range (가청대역)
    Creates artifacts 

    [Solution]
    step1) Apply low-pass filter
        -cutoff : Slightly below Nyquist
        -Purpose : Remove frequencies > Nyquist
        (나이퀴스트 이상 주파수 제거)
    step2) Sample at fs
        - Now safe! No aliasing possible
    step3) Optionally, filter after
        - Reconstruction filter
        - Cleanup any artifacts

    [Audio Implications]

    Real ADCs have anti-aliasing filters
        => 보통 ADC 전단에서 anti-aliasing filter 적용됨
    Digital -> Analog converters have reconstruction filters
        (이거는 aliasing 을 제거하는 목적은 아님)
    This is why audio hardware is complex 

    [실제로]
    Analog signal -> Analog Low-pass filter -> ADC
    
    
    """

def understand_interpolation_and_decimation():
    # decimation : interpolation 의 반대

    """
    Changing sample rate 
    Upsampling and downsampling

    언제 해야하는가? (Sample rate conversion)

        1. 서로 다운 장비 연결 (ex. CD)
            => 그냥 연결하면 속도가 변하고, 음정이 변한다

        2. DSP 처리
            => Input -> Upsample(x4) -> Distortion -> Low-pass -> Downsample
            (Distortion 은 새로운 고조파를 만드는데, 그 고조파가 nyquist를 넘으면 aliasing 됨)
        3. distortion, saturation, compressor, synth 등의 플러그인은 내부에서
            2배, 4배 등의 업샘플링이 필요함
            => 비선형 처리에서는 새로운 고조파가 생기고, 이들이 원래 샘플레이트의 nyquist 를 넘으면
               aliasing 이 발생하기 때문에 

    
    [PROCESSES] Interpolation & Decimation

    1) DECIMATION (다운샘플링: 샘플 수 줄이기)
    Goal: Reduce sample rate (44.1 -> 22.05)

    Wrong way (danger of aliasing)
        1. Just drop every other sample
        2. Hope there's no high-frequency content
        3. Often creates artifacts

    Right way 
        1. Apply low-pass filter @ new Nyquist (22.05kHz 의 nyquist)
           Removes frequencies > 22.05kHz
        2. Then drop every other sample
        => safe, no aliasing 

    2) INTERPOLATION (업샘플링: 샘플 수 줄이기)
    Goal: Increase sample rate (44.1 -> 96kHz)

    Right way:
        1. Insert zeros between samples
        2. Apply low-pass filter @ old Nyquist
           Removes newly-created artifacts
        3. Scale by interplation factor

    => Must filter to prevent artifacts 
    (both require anti-aliasing/reconstruction filtering)

    
    **SUMMARY
    1) Downsampling:
        LPF -> Decimate
        (Prevent aliasing)
    2) Upsampling: imaging 발생(원래 스펙트럼의 복사본이 생김)
        Zero insertion -> LPF
        (remove imaging)

        ** Upsampling 에서 1 2 3 4 를 
                          1 0 2 0 3 0 4 0 이렇게 0을 끼워넣는 식으로 만드는데
                          image 가 생기는 이유
        근데 그거는 사실 원래신호에 1 0 1 0 1 0 1 0 을 곱한것과 같고, 이 패턴 자체가 새로운 주파수를 가지고 있다.
            (2배 업샘플링은 그냥 새로운 배열을 0끼워서 만드는것)
            => 이렇게 시간영역에서 갑자기 1 0 1 0 .. 으로 빠르게 변하면 푸리에 관점에서는 고주파수가 많이 생김
            그렇기 떄문에 그걸 LPF 해서 1
                                    \
                                      \
                                        2
                                          \
                                            \
                                              3 이렇게 각 샘플 사이의 중간값을 자연스럽게 만들어줌 
            => 0을 넣어서 샘플 위치를 만든뒤에, LPF 가 그 빈칸을 적절한 값으로 채우는것이 업샘플리의 핵심이다.
            (고주파를 없애는것 = 중간 샘플을 계산하는것)

            
    **고주파를 없애는 것 = 중간 샘플을 계산하는것 (오버샘플링 시 1 2 3 4 사이에 껴있는 0 들)

    고주파를 없애는 것은 필터(filter)이고...
    중간값을 계산하는것은 보간(interpolation)이라고 알고있는데..

    1 2 3 을 업샘플링하면 1 0 2 0 3 0 이 나오게 된다.
    => 우리는 아직 0 의 값을 모름

    우리가 원하는건! 1 1.5 2 2.5 3 이런거! 요기 0이었던것을 중간값으로 바꾸는것이다.

    근데 결국 1 0 2 이거는 빠르게 0을 갔다가 다시 2 로 움직인것이므로 불필요한 고주파가 생긴 상태이다.
    ... 이걸 시각적으로 이해하려면?

    """

def 


visualize_aliasing()