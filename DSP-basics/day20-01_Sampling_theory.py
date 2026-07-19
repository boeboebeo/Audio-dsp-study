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

def demonstrate_sample_rate_conversion():
    fs_old = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(fs_old*duration), endpoint=False)

    # mix of frequencies
    f1, f2 = 2000, 15000
    signal_orig = 0.5*np.sin(2*np.pi*f1*t) + 0.5*np.sin(2*np.pi*f2*t)

    # Resample to 48 kHz (Upsampling)
    fs_new = 48000
    ratio = fs_new / fs_old

    # proper resampling (uses anti-aliasing filter internally)
    signal_resampled = signal.resample(signal_orig, int(len(signal_orig)*ratio))
        #원래의 신호의 개수에다가 ratio 를 곱해서 리샘플링 시행

    # Visaulization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # plot 1: Original signal
    ax = axes[0, 0]
    ax.plot(t[:100], signal_orig[:100], linewidth=0.5, color='blue')
        # 왜 2000 까지만 찍는거지
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Original Signal\n({fs_old}Hz, {len(signal_orig)} samples)')
    ax.grid(True, alpha=0.3)

    # plot 2: Resampled signal
    ax = axes[0, 1]
    t_new = np.linspace(0, duration, int(fs_new*duration), endpoint=False)
    ax.plot(t_new[:100], signal_resampled[:100], linewidth=0.5, color='red')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Resampled Signal\n({fs_new}Hz, {(len(signal_resampled))} samples)')
    ax.grid(True, alpha=0.3)

    # plot 3: Spectrum comparison
    ax = axes[1, 0]
    X = np.fft.rfft(signal_orig)

    freq_orig = np.fft.rfftfreq(
        len(signal_orig),
        d = 1/fs_old
    )   #이렇게 하면 //2 로 나누지 않아도 됨 그냥 중복되지 않는곳까지만 계산해서 출력함
    mag_orig = np.abs(X)

    # ax.semilogy(freq_orig, mag_orig, linewidth=1, color='blue')
        # 한 축만 로그스케일로 그리는 그래프
        # ax.set_yscale('log') 와 같음 -> 근데 여기서는 이미 dB 로 log 써서 계산해놨으니 필요 없음
    
    ax.plot(freq_orig, mag_orig, linewidth=1, color='blue')
    ax.axvline(fs_old/2, color='r', linestyle='--', alpha=0.5, label=f'Nyquist {fs_old/2}Hz')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Original Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 25000)

    """
    ax = axes[1, 0]
    freqs_orig = np.fft.rfftfreq(len(signal_orig), d=1/fs_old)[:len(signal_orig)//2]
    mag_orig = np.abs(np.fft.fft(signal_orig))[:len(signal_orig)//2]
        => 원래 코드의 문제점:

    fft() 를 쓰게 되면 X = np.fft.fft(x)에는 
        0 1 2 3 4 3 2 1 이렇게 절반이 중복된다. 

        => np.fft.rfft(x) 는 'r'을 붙임으로써, 애초에 중복되는 뒤 절반을 계산하지 않음!
            0 1 2 3 4 까지만 계산하게 한다. 

        그래서 rfftfreq() 는 이미 0부터 nyquist 까지만 만들기 때문에 //2 를 할 필요 없음

    """

    # plot4: Resampled spectrum
    ax = axes[1, 1]
    freq_new = np.fft.rfftfreq(len(signal_resampled), d=1/fs_new)
    X_new = np.fft.rfft(signal_resampled)
    mag_new = np.abs(X_new)

    ax.plot(freq_new, mag_new, linewidth=1, color='red')
    ax.axvline(fs_new/2, color='r', linestyle='--', alpha=0.5, label=f'Nyquist {fs_new/2}Hz')
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Resampled Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 25000)
    
    # 그래프를 확인해보면 fs_old, fs_new 버전의 샘플레이트로 만든 신호의
    # 진폭이 미세하게 다른걸 확인할 수 있는데 이것은
    # => signal.resample() 이 새로운 샘플위치에서 다시 계산하는 함수이기 때문에 미세한 차이를 유발한다.
    # 실제로는 signal.resample_plot()를 더 많이 사용함


    plt.tight_layout()
    plt.show()
    
def practical_interpolation_methods():
    # 다양한 보간방법
    """[Interpolation Techniques]

    1) Nearest neighbor (최근접 이웃)

    2) Linear interpolation

    3) Cubic interpolation(3차)

    4) Sinc interpolation (Ideal)

    => Practical choice
        - Real-time : Linear or cubic
        - High quality : Sinc (windowed)
    
    """

# demonstrate_sample_rate_conversion()



# visualize_aliasing()