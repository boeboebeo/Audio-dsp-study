"""
==============================================
DAY 9: Filters (Biquad & State Variable Filters)
==============================================
Goal: 필터의 수학적 원리를 이해하고 다양한 필터 타입을 구현한다.

    -> 계수가 왜 그렇게 나오는지 추가로 더 공부해야함
        + 아직 위상과 그걸 radian 으로 보는것이 익숙하지 않은것같다. (260527)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 1.0

def biquad_filter(signal_input, filter_type, cutoff_freq, resonance, sample_rate):
    """
    Biquad Filter 

    Biquad = "Bi-quadratic" (2차 필터)

    Transfer function (전달함수)
    H(z) = (b0 + b1*z^(-1) + b2*z^(-2)) / (a0 + a1*z^(-1) + a2*z^(-2))

        차분방정식과 완전히 같은 내용임 (위 전달함수 H(z))
        z^(-1) : 1샘플 과거(Delay 1)
        z^(-2) : 2샘플 과거(Delay 2)
        => 차분방정식은 어떻게 계산하냐를 알려주지만, 전달 함수는 주파수별로 어떻게 반응하냐를 분석할 수 있게 함
        
        z = e^(jw) 를 대입하면 (w=각주파수) -> 각 주파수에서 필터가 신호를 얼마나 증폭/감쇠하는지 바로 계산 가능함
        H(z) → z = e^(jω) 대입 → H(e^(jω)) = 주파수 응답

        => H(z) 와 Y(n) 은 같은 존재임 (아이패드에 정리해둠)



    Difference equation (차분방정식)
    y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / a0

        지금 출력값 y[n]은, 지금/과거 입력값들과 과거 출력값들의 가중합이다 (b 계수 : 입력에 곱하는 가중치, a 계수 : 과거 출력에 곱하는 가중치. feedback)
        지금 출력 = 지금 입력 + 1샘플 전 입력 + 2샘플 전 입력 - 1샘플 전 출력 - 2샘플 전 출력
        (각각에 숫자(계수)를 곱해서 더함)

        Ex. b0=1, b1=1, 나머지가 다 0이면 y[n] = x[n] + x[n-1] 이게 전부
            => 지금 샘플이랑 1샘플 전 거 더해라! 
                : 저음 강조, 고음이 깎임 -> 고음은 샘플값이 빠르게 왔다갔다 하는데, 직전값이랑 더하면 서로 상쇄되어버림. 
                  근데 저음은 천천히 변하니까 더하면 오히려 커짐

        x[n]   지금 이 순간 들어온 입력
        x[n-1] 1샘플 전 입력
        x[n-2] 2샘플 전 입력
        y[n-1] 1샘플 전 출력 (자기 자신의 과거)
        y[n-2] 2샘플 전 출력
            b 계수들 : 입력 쪽 가중치(feedforward)
            a 계수들 : 출력 쪽 가중치(feedback. 자기 자신을 되먹임) => 위 수식은 그냥 레시피임

        => y[n-1], y[n-2] 항 : 출력이 다시 입력으로 들어오는 구조. 아날로그 필터에서의 커패시터/인덕터가 하던 일
        (아날로그에서 커패시터가 과거 전압을 기억하듯이, 디지털에서도 이전 샘플값을 메모리에 저장해서 같은 효과를 낸다)

        
    Parameters:
    - cutoff_freq: filter cutoff frequency (차단 주파수)
    - resonance: Q factor (공명, 0.5 ~ 20+)
      - Low Q: gentle slope (부드러운 기울기)
      - High Q: sharp peak (날카로운 피크), self-oscillation (자기 발진)
    
    Filter types:
    - lowpass (LPF): passes low frequencies (저역 통과)
    - highpass (HPF): passes high frequencies (고역 통과)
    - bandpass (BPF): passes middle band (대역 통과)
    - notch: rejects middle band (대역 차단)
   """
    
    # Normalize frequency (정규화된 주파수)
    omega = 2 * np.pi * (cutoff_freq / sample_rate)
        # 주파수를 각도로 변환하는것 
        # 2pi 안에서 전체에서 차지하는 비율이라고 보면 됨
        # Cutoff _Freq / sample_rate = 0 ~ 1 사이의 값으로 정규화를 하는것 -> * 2pi는 라디안 단위로 값을 변화시킴

        # 디지털 필터에서는 Hz 를 직접 모르고, 1샘플 안에서 몇 바퀴 도는가로 주파수를 이해함
        # ex. 2pi * 1000 / 44100 = 약 0.1425 라디안 => 한 샘플마다 원을 0.1425 라디안씩 돌음
        # 샘플레이트가 높을수록 omega 도 작아짐 (같은 주파수도 샘플이 많으면 한 샘플당 조금씩만 전진)
        # RBJ Audio EQ Cookbook 계수 공식

    sin_omega = np.sin(omega)
        # 그 각도의 세로위치
    cos_omega = np.cos(omega)
        # 그 각도의 가로위치

    # Q factor (공명 계수)
    Q = resonance
    alpha = sin_omega / (2 * Q)
        # b0, b1, b2, a1, a2 를 계산할때 거의 모든 식에 alpha가 들어감
        # 얼마나 넓은 주파수 범위에 필터가 작용하는지를 결정해야 함(필터가 작용하는 주파수 범위의 너비)
        # Q가 커지면 alpha가 작아짐 -> 필터 폭이 좁아짐
        # alpha 는 절반짜리 한쪽을 나타내는 값이기때문에 2 * Q 를 처리함
        # 주파수 위치에 따라서 필터 폭을 보정

        # 후... 이거는 pole/zero 까지 간다면 더 이해가 잘 될듯!!!! 그때 한번 다시 와 보자

    """ Self Oscillation

여기서 Q 값이 매우 높으면 alpha 값이 줄어들기 때문에 
        => y[n] = ... - a1·y[n-1] - a2·y[n-2]
        이 수식과 밑 lowpass filter 의 a0, a1, a2 의 계수가 각 
        a0 = 1 + alpha = 걍 1 (alpha 작음)
        a1 = 
        a2 = 1 - alpha = 거의 1 (alpha 작음) 
        되어서 거의 2샘플전 출력을 다시 그대로 더하는 뜻이 됨 -> 에너지가 빠져나가지 않고 계속 순환! 
        => self oscillation 

        Q가 낮다 => 공기 저항이 크다 : 빨리 멈춤
        Q가 높다 => 공기 저항이 없다 : 오래 울림 
        Q가 무한대 => 마찰이 아예 없음 : 자기 발진

** 아날로그에서는 열노이즈, 전원 노이즈가 존재하기 때문에 그걸 씨앗으로 feedback이 시작되어 발진 
    (모든 전자부품은 온도가 있으면 전자가 무작위로 진동 -> 저항, 트랜지스터, 전원 노이즈 등)

"""
       


    # Calculate coefficients based on filter type (필터종류에 따라서 b, a 계수를 다르게 셋팅)
    # a0, a1, a2 는 모두 동일함 -> LP, HP, BP, Notch 는 오직 b의 계수만 바꾸는 것
    # : a계수는 feedback 부분이기 때문에 항상 동일

    if filter_type == 'lowpass': # 느린 변화의 낮은 주파수는 유지, 빠른 변화의 높은 주파수는 통과
        b0 = (1 - cos_omega) / 2
        b1 = 1 - cos_omega #얘가 양수가 나오면 
        b2 = (1 - cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
        """ cos_omega = cutoff freq 에서의 cos 값

        - cutoff freq 가 낮으면 -> omega가 작음 -> cos_omega 가 1에 가까움
            : 1 - cos_omega 가 0에 가까움

        - cutoff freq 가 높으면 -> omega가 큼 -> cos_omega 가 -1에 가까움
            : 1 - cos_omega 가 2에 가까움

            => b1 이 b0, b2 의 딱 두배. 셋다 양수이기 -> 저음을 모으는 모양이 됨
        
        """
        
    elif filter_type == 'highpass':
        b0 = (1 + cos_omega) / 2
        b1 = -(1 + cos_omega)
        b2 = (1 + cos_omega) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
    elif filter_type == 'bandpass':
        b0 = alpha
        b1 = 0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha
        
    elif filter_type == 'notch':
        b0 = 1
        b1 = -2 * cos_omega  # COF 주파수 대역이 정확히 상쇄 -> 그 주파수만 제거
        b2 = 1
        a0 = 1 + alpha
        a1 = -2 * cos_omega
        a2 = 1 - alpha

    # Normalize coefficients (정규화)
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])
        
    # 원래의 차분방정식은 
    # a0·y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] - a1·y[n-1] - a2·y[n-2] 이런 형태인데,
    # 컴퓨터가 실제로 계산하려면 y[n] 혼자서 왼쪽에 있어야 하므로 양변을 a0 로 나눔
    # => y[n] = (b0/a0)·x[n] + (b1/a0)·x[n-1] + (b2/a0)·x[n-2]
    #    - (a1/a0)·y[n-1] - (a2/a0)·y[n-2]
        # 그걸 한줄씩 해놓은게, b = , a =  이라는 배열


    # Apply filter using difference equation
    filtered = signal.lfilter(b, a, signal_input)
        # lfilter : 정규화된 b, a 계수로 차분방정식을 신호 전체에 샘플 하나하나 적용하게 해줌
        # 직접 하면 for 문 돌려야 하는데 lfilter가 signal_input 의 모든 샘플에 대해 자동으로 반복해주게 함
    """ 
    아래와 같게 for 문 돌려서 한 샘플 한 샘플의 magnitude 계산해야 하는데, 
    signal.lfilter가 그 역할 대신함

    for n in range(len(signal)):
    y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    
    """

    return filtered, b, a

def plot_frequency_response(b, a, sample_rate, title):
    """
    Plot filter frequency response (주파수 응답)

    - Magnitude response: gain at each frequency (각 주파수의 이득)
    - Phase response: phase shift at each frequency (각 주파수의 위상 변화)    
    """
    w, h = signal.freqz(b, a, worN=8000, fs=sample_rate)

    fig, axes = plt.subplots(2, 1, figsize = (10, 8))

    # Magnitude response
    axes[0].plot(w, 20 * np.log10(abs(h)), linewidth=2, color='blue')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'{title} - Frequency Response')
    axes[0].set_xscale('log')
    axes[0].set_xlim(20, sample_rate / 2)
    axes[0].grid(True,alpha = 0.3, which='both')
    axes[0].axhline(-3, color='red', linestyle='--', alpha=0.5, label='-3dB (cutoff)')
    axes[0].legend()
    
    # Phase response
    angles = np.unwrap(np.angle(h))
    axes[1].plot(w, np.degrees(angles), linewidth=2, color='green')
    axes[1].set_ylabel('Phase (degrees)')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_xscale('log')
    axes[1].set_xlim(20, sample_rate / 2)
    axes[1].grid(True, alpha=0.3, which='both')
    
    return fig

def demonstrate_filter_types():
    # 다양한 필터 타입 시연

    # Generate test signal : sawtooth with multiple harmonics
    freq = 110
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)

    # Band-limited sawtooth (using additive synthesis)
    test_signal = np.zeros_like(t)
    nyquist = SAMPLE_RATE / 2
    max_harmonic = int(nyquist / freq)

    for n in range(1, min(max_harmonic, 100)):
        amplitude = (2 / np.pi) * ((-1) ** (n+1)) / n
        test_signal += amplitude * np.sin(2 * np.pi * n * freq * t)

    # Filter parameters
    cutoff = 880
    resonance = 2.0 # Moderate Q

    filter_types = ['lowpass', 'highpass', 'bandpass', 'notch']

    fig, axes = plt.subplots(len(filter_types), 2, figsize=(12, 8))

    for idx, ftype in enumerate(filter_types):
        # Apply filter
        filtered, b, a = biquad_filter(test_signal, ftype, cutoff, resonance, SAMPLE_RATE)

        # Time domain
        w, h = signal.freqz(b, a, worN=4000, fs=SAMPLE_RATE)
        axes[idx, 0].plot(w, 20 * np.log10(np.abs(h) + 1e-10), linewidth=2, color='blue')
        axes[idx, 0].axvline(cutoff, color='red', linestyle='--', alpha=0.7, label=f'Cutoff: {cutoff}Hz')
        axes[idx, 0].axhline(-3, color='orange', linestyle='--', alpha=0.5, label='-3dB')
        axes[idx, 0].set_ylim(-60, 10)
        axes[idx, 0].set_xscale('log')
        axes[idx, 0].set_xlim(20, SAMPLE_RATE / 2)
        axes[idx, 0].set_title(f'{ftype.upper()} - Frequency Response')
        axes[idx, 0].set_ylabel('Magnitude (dB)')
        axes[idx, 0].set_xlabel('Frequency (Hz)')
        axes[idx, 0].legend(fontsize=8)
        axes[idx, 0].grid(True, alpha=0.3, which='both')

        # Frequency domain
        N = len(filtered)
        fft_original = fft(test_signal)
        fft_filtered = fft(filtered)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        mag_original = np.abs(fft_original[:N//2]) * 2 / N
        mag_filtered = np.abs(fft_filtered[:N//2]) * 2 / N
        
        axes[idx, 1].plot(positive_freqs, mag_original, linewidth=1, 
                         color='gray', alpha=0.5, label='Original')
        axes[idx, 1].plot(positive_freqs, mag_filtered, linewidth=1.5, 
                         color='blue', label='Filtered')
        axes[idx, 1].axvline(cutoff, color='red', linestyle='--', 
                           alpha=0.7, label=f'Cutoff: {cutoff} Hz')
        axes[idx, 1].set_xlim(0, 3000)
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_title(f'{ftype.upper()} - Spectrum')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, alpha=0.3)
        # axes[idx, 1].set_yscale('log')
        # 위 set_yscale('log') 때문에 0에 가까운값을 -무한대로 보내버렸었음. 
        # 근데 스펙트럼에서 신호가 없는 주파수 구간은 거의 0이이여서 log(0) 이 아래로 날라가 버리면 이상하게 떠있는것 처럼 표현됨
    
    plt.tight_layout()
    plt.show()

def resonance_effect():
    """
    Resonance (Q factor) 효과 분석
    
    High Q:
    - Sharp peak at cutoff (차단 주파수에 날카로운 피크)
    - Can self-oscillate (자기 발진 가능)
    - "Acid" sound (애시드 사운드)
    """
    # White noise as test signal (모든 주파수 포함)
    noise = np.random.randn(int(SAMPLE_RATE * DURATION))
    
    cutoff = 1000  # 1 kHz
    Q_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, Q in enumerate(Q_values):
        filtered, b, a = biquad_filter(noise, 'lowpass', cutoff, Q, SAMPLE_RATE)
        
        # Spectrum
        N = len(filtered)
        fft_result = fft(filtered)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N
        
        axes[idx].plot(positive_freqs, magnitude, linewidth=1, color='blue')
        axes[idx].set_xlim(100, 5000)
        axes[idx].set_xlabel('Frequency (Hz)')
        axes[idx].set_ylabel('Magnitude')
        axes[idx].set_title(f'Resonance Q = {Q}')
        axes[idx].axvline(cutoff, color='red', linestyle='--', alpha=0.7)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xscale('log')
        
        # Calculate peak gain
        peak_idx = np.argmax(magnitude[positive_freqs > 100])
        peak_freq = positive_freqs[positive_freqs > 100][peak_idx]
        peak_gain = magnitude[positive_freqs > 100][peak_idx]
        
        axes[idx].text(0.6, 0.9, f'Peak: {peak_gain:.3f}\n@ {peak_freq:.0f} Hz',
                      transform=axes[idx].transAxes, fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def state_variable_filter(signal_input, cutoff_freq, resonance, sample_rate):
    """
    State Variable Filter (SVF) (상태 변수 필터)
        => 필터 내부 상태(state) 를 계속 업데이트 함
        - svf : 아날로그 integrator(적분) 회로 기반

    Biquad : 하나의 수식 -> b 계수를 바꾸면 LP/HP/BP/Notch 형태로 필터타입 바뀜
    SVF    : 하나의 구조 -> LP, HP, BP 세개가 동시에 나옴
        - lowpass = 0
        - bandpass = 0 
            : 이 두 변수가 SVF의 메모리! -> 현재 필터의 상태를 기억함 (매 샘플마다 이 값들이 업데이트 되면서 필터가 작동함)
    
    
    Architecture:
    - Produces LP, BP, HP outputs simultaneously (동시에 3개 출력)
    - Uses 2 integrators (2개 적분기 사용)
    - Better for modulation (변조에 더 좋음)
    
    Topology (위상 구조):
    Input → [+] → BP integrator → LP integrator → LP out
              ↑                         ↓
              └─────── feedback ────────┘
    
    Advantages:
    - Smooth parameter changes (부드러운 파라미터 변화)
    - Stable at high resonance (높은 공명에서도 안정적)
    - Multiple outputs (다중 출력)

    svf 는 매 샘플마다 :
        새 상태 = 현재 상태 + (f * 변화량) -> 이렇게 작동함

        f = 한 샘플마다 얼마나 전진하냐
    """

    # Calculate coefficients
    f = 2 * np.sin(np.pi * (cutoff_freq / sample_rate))
        # w = 2pi * (fc/fs) = radian 표현 방식
        # 컷오프를 높이면 f 가 커짐 => f가 커지면 고역대가 큰 진폭으로 출력 배열에 저장될 수 있음


    q = 1 / resonance
        # 레조넌스가 커지면 q 작아짐 
        # 레조넌스가 작아지면 q 커짐
        # q : 얼마나 오래 울릴까 
        # resonance 담당

    # Initialize state variables (상태 변수 초기화)
    lowpass = 0
    bandpass = 0 
        # 위와 같은 내부 상태가 매 샘플마다 조금씩 업데이트 됨

    #입력이랑 똑같은 길이의 0배열
    lp_out = np.zeros_like(signal_input)
    bp_out = np.zeros_like(signal_input)
    hp_out = np.zeros_like(signal_input)

    for i in range(len(signal_input)):
        # high-pass = input - lowpass - Q*bandpass
        highpass = signal_input[i] - lowpass - q * bandpass
            # q가 작아지면 (resonance가 커지면) q*bandpass 가 거의 0이 됨 => bandpass 를 빼는 양이 줄어들게 된다
            # => 그러면 에너지가 빠져나가지 않고 루프 안에서 계속 순환하다가 발진! 
            # self oscillation (bi-quad 보다 발진이 더 부드럽게 시작됨)

        # band-pass integrator (적분기)
        bandpass = bandpass + f * highpass

        # low-pass integrator 
        lowpass = lowpass + f * bandpass 

        # store outputs
        lp_out[i] = lowpass
        bp_out[i] = bandpass
        hp_out[i] = highpass 
            # i 번째 샘플자리에 계산 결과를 넣음

    """각각 저장되고 있는 샘플의 결과
    lp_out = [샘플0의 LP값, 샘플1의 LP값, 샘플2의 LP값, ...]
    bp_out = [샘플0의 BP값, 샘플1의 BP값, 샘플2의 BP값, ...]
    hp_out = [샘플0의 HP값, 샘플1의 HP값, 샘플2의 HP값, ...]

    **디지털 오디오에서의 거의 모든 처리가 결국 아래의 구조**

        입력 샘플 배열  →  뭔가 계산  →  출력 샘플 배열
    [0.5, 0.6, 0.7, ...]      [0.125, 0.306, 0.495, ...]    

        => 어떤 입력 샘플 배열에 무슨 계산 처리를 해서 그걸 출력 샘플배열로 저장함
        *샘플레이트가 44100번이라면 그걸 1초동안 44100번 반복하는 것임
            => 그 계산이 실시간으로 일어나야 하므로, 44100번의 계산이 1초안에 끝나야함
    """

    return lp_out, bp_out, hp_out

def compare_biquad_svf():
    """
    Biquad vs SVF 비교
    """
    # Test signal
    freq = 220
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    test_signal = np.zeros_like(t)
    nyquist = SAMPLE_RATE / 2
    max_harmonic = int(nyquist / freq)
    
    for n in range(1, min(max_harmonic, 80)):
        amplitude = 1 / n
        test_signal += amplitude * np.sin(2 * np.pi * n * freq * t)
    
    cutoff = 1000
    resonance = 5.0
    
    # Biquad
    biquad_lp, _, _ = biquad_filter(test_signal, 'lowpass', cutoff, resonance, SAMPLE_RATE)
    
    # SVF
    svf_lp, svf_bp, svf_hp = state_variable_filter(test_signal, cutoff, resonance, SAMPLE_RATE)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain comparison
    plot_samples = int(0.02 * SAMPLE_RATE)
    t_plot = t[:plot_samples] * 1000
    
    axes[0, 0].plot(t_plot, biquad_lp[:plot_samples], linewidth=1.5, color='blue', label='Biquad LP')
    axes[0, 0].plot(t_plot, svf_lp[:plot_samples], linewidth=1, color='red', 
                   linestyle='--', alpha=0.7, label='SVF LP')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Lowpass Output Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # SVF multiple outputs
    axes[0, 1].plot(t_plot, svf_lp[:plot_samples], linewidth=1.5, color='blue', label='LP', alpha=0.7)
    axes[0, 1].plot(t_plot, svf_bp[:plot_samples], linewidth=1.5, color='green', label='BP', alpha=0.7)
    axes[0, 1].plot(t_plot, svf_hp[:plot_samples], linewidth=1.5, color='red', label='HP', alpha=0.7)
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('SVF Simultaneous Outputs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Spectra
    N = len(test_signal)
    freqs = fftfreq(N, 1/SAMPLE_RATE)
    positive_freqs = freqs[:N//2]
    
    fft_biquad = fft(biquad_lp)
    fft_svf_lp = fft(svf_lp)
    fft_svf_bp = fft(svf_bp)
    fft_svf_hp = fft(svf_hp)
    
    mag_biquad = np.abs(fft_biquad[:N//2]) * 2 / N
    mag_svf_lp = np.abs(fft_svf_lp[:N//2]) * 2 / N
    mag_svf_bp = np.abs(fft_svf_bp[:N//2]) * 2 / N
    mag_svf_hp = np.abs(fft_svf_hp[:N//2]) * 2 / N
    
    axes[1, 0].plot(positive_freqs, mag_biquad, linewidth=1, color='blue', label='Biquad LP')
    axes[1, 0].plot(positive_freqs, mag_svf_lp, linewidth=1, color='red', 
                   linestyle='--', label='SVF LP')
    axes[1, 0].set_xlim(100, 5000)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Lowpass Spectrum Comparison')
    axes[1, 0].axvline(cutoff, color='black', linestyle=':', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].plot(positive_freqs, mag_svf_lp, linewidth=1, color='blue', label='LP')
    axes[1, 1].plot(positive_freqs, mag_svf_bp, linewidth=1, color='green', label='BP')
    axes[1, 1].plot(positive_freqs, mag_svf_hp, linewidth=1, color='red', label='HP')
    axes[1, 1].set_xlim(100, 5000)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].set_title('SVF All Outputs')
    axes[1, 1].axvline(cutoff, color='black', linestyle=':', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def filter_sweep():
    """
    Filter sweep (필터 스윕)
    
    Classic technique:
    - Sweep cutoff frequency over time (시간에 따라 차단 주파수 변화)
    - Creates "wah" effect (와우 효과)
    - Used in: disco, house, techno
    """
    #Rich harmonic source(sawtooth)
    freq = 55  #Low note for more harmonics
    t = np.linspace(0, 4.0, int(SAMPLE_RATE*4.0), endpoint=False)

    test_signal = np.zeros_like(t)
    nyquist = SAMPLE_RATE / 2
    max_harmonic = int(nyquist / freq)

    for n in range(1, min(max_harmonic, 200)):
        amplitude = 1 / n
        test_signal += amplitude * np.sin(2 * np.pi * n * freq * t)

    # sweep cutoff from 200 Hz to 4000Hz and back 
    sweep_period = 4.0
    cutoff_curve = 200 + 3800 * (0.5 + 0.5 * np.sin(2 * np.pi * t / sweep_period))

    # Apply time-varying filter
    resonance = 3.0
    filtered = np.zeros_like(test_signal)

    # Process in small chunks (작은 청크로 처리)
    chunk_size = 512
    num_chunks = len(test_signal) // chunk_size
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        
        # Use average cutoff for this chunk
        chunk_cutoff = np.mean(cutoff_curve[start:end])
        
        chunk_filtered, _, _ = biquad_filter(test_signal[start:end], 'lowpass', 
                                            chunk_cutoff, resonance, SAMPLE_RATE)
        filtered[start:end] = chunk_filtered
    
    # Spectrogram
    from scipy import signal as sp_signal
    f, t_spec, Sxx = sp_signal.spectrogram(filtered, SAMPLE_RATE, 
                                           nperseg=2048, noverlap=1536)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Cutoff frequency curve
    axes[0].plot(t, cutoff_curve, linewidth=2, color='blue')
    axes[0].set_ylabel('Cutoff Frequency (Hz)')
    axes[0].set_title('Filter Sweep: Cutoff Modulation')
    axes[0].grid(True, alpha=0.3)
    
    # Waveform
    axes[1].plot(t, filtered, linewidth=0.5, color='purple')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Filtered Waveform')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 4)
    
    # Spectrogram
    im = axes[2].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                           shading='gouraud', cmap='magma')
    axes[2].plot(t, cutoff_curve, 'w--', linewidth=2, alpha=0.7, label='Cutoff')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Spectrogram: Filter Sweep Effect')
    axes[2].set_ylim(0, 5000)
    axes[2].legend(loc='upper right')
    plt.colorbar(im, ax=axes[2], label='Power (dB)')
    
    plt.tight_layout()  
    plt.show()

def self_oscillation():
    """
    Self-oscillation (자기 발진)
    
    When Q → ∞:
    - Filter becomes oscillator (필터가 발진기가 됨)
    - Resonant frequency rings out (공명 주파수가 울림)
    - Used creatively in synthesis (신시사이즈에서 창의적으로 사용)
    """

    # Very quiet input(impulse)
    impulse = np.zeros(int(SAMPLE_RATE*2))
    impulse[1000] = 1.0 #single impulse 

        #디지털에서는 아날로그 장치에서의 열노이즈나 전원 노이즈 같은 씨앗이 없으므로 impulse 하나를 매우 짧게 입력함!

    cutoff = 440
    Q_values = [1.0, 5.0, 10.0, 50.0, 100.0]

    fig, axes = plt.subplots(len(Q_values), 2, figsize=(12, 8))

    for idx, Q in enumerate(Q_values):
        filtered, b, a = biquad_filter(impulse, 'lowpass', cutoff, Q, SAMPLE_RATE)

        # Time domain
        t = np.arange(len(filtered)) / SAMPLE_RATE
        plot_start = 900
        plot_end = plot_start + 2000
        
        axes[idx, 0].plot(t[plot_start:plot_end] * 1000, filtered[plot_start:plot_end], 
                         linewidth=1.5, color='blue')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].set_title(f'Q = {Q} - Impulse Response')
        axes[idx, 0].grid(True, alpha=0.3)
        
        if Q >= 10:
            axes[idx, 0].text(0.6, 0.8, 'Self-oscillating!\n(filter rings at cutoff)',
                            transform=axes[idx, 0].transAxes, fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        # Spectrum
        N = len(filtered)
        fft_result = fft(filtered)
        freqs = fftfreq(N, 1/SAMPLE_RATE)
        positive_freqs = freqs[:N//2]
        magnitude = np.abs(fft_result[:N//2]) * 2 / N
        
        axes[idx, 1].plot(positive_freqs, magnitude, linewidth=1, color='blue')
        axes[idx, 1].axvline(cutoff, color='red', linestyle='--', alpha=0.7)
        axes[idx, 1].set_xlim(100, 2000)
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Magnitude')
        axes[idx, 1].set_title(f'Q = {Q} - Spectrum')
        axes[idx, 1].grid(True, alpha=0.3)
        # axes[idx, 1].set_yscale('log')
    
    axes[-1, 0].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    plt.show()


# demonstrate_filter_types()
# resonance_effect()
# compare_biquad_svf()
# filter_sweep()
self_oscillation()


""" Svf 와 moog Ladder 

    -biquad , svf, moog ladder 모두 다 self oscillation 가능

1) SVF 
  : HP → (적분) → BP → (적분) → LP
  => 적분 2번이라서 2차 필터

  +SVF는 feedback 이 내부에서 바로 돌아옴
    - HP = x - q*BP - LP  
    -> 바로 자기 자신을 참조


2) Ladder 
  : 입력 → [적분] → [적분] → [적분] → [적분] → 출력
  => 적분 4번이라서 4차 필터
    (아날로그에서는 이 적분단계가 트랜지스터 + 커패시터 -> 무그는 이게 4개 직렬 연결된 것)

  +입력 → [1] → [2] → [3] → [4] → 출력
  ↑________________________________↓
              feedback

    -> Ladder 는 feedback 이 맨 끝 출력에서 맨 앞으로 돌아옴! 
       (4단계를 거쳐서 돌아오기 때문에 위상이 180도 뒤집혀서 돌아옴)


**아날로그 회로 이론**
       ↓
   Ladder 필터 (Moog, 1960년대)
   "트랜지스터로 적분을 4번"
       ↓
   SVF (1970년대)
   "Ladder를 단순화 + 세 출력 동시에"
       ↓
   Biquad (디지털 시대)
   "SVF를 수식으로 추상화"

**Self oscillation 가능여부**

    - feedback 이 있냐      -> FIR 필터는 원천 불가 (feedback 없음)
    - 에너지가 클리핑 되어있냐  -> 클리핑 되면 불가 (wasp filter 같은게 그럼)
    - 감쇠를 막아놨냐         -> 설계로 막으면 불가 

**Wasp -> CMOS 로직 칩으로 만든 필터

CMOS 칩 특성상 출력이 0 아니면 1로 클리핑돼버려. 아날로그처럼 부드럽게 증폭이 안 되고 바로 잘려나가거든.
그래서 Q를 아무리 올려도 에너지가 쌓이기 전에 파형이 잘려버려서 발진까지 못 가는 거야.

"""