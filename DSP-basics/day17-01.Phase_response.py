"""
==============================================
DAY 17: Phase Response Deep Dive
==============================================

MATHEMATICAL PREREQUISITES:
- Complex numbers (복소수): Form a + bi
- Derivatives (미분): Rate of change
- Unwrapping (풀기): Making discontinuous → continuous

KEY CONCEPT:
Magnitude tells you "how much" a filter affects volume
Phase tells you "when" a filter affects signals
Both matter for audio quality!
(크기: 얼마나 변하는가? 위상: 언제 변하는가?)

Human hearing is mostly magnitude-blind but PHASE-SENSITIVE for:
- Transients (순간적 변화)
- Stereo imaging (입체 음향)
- Punch and clarity (명확함)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft #inverse fast fourier transform

SAMPLE_RATE = 44100

def understand_why_phase_matters():
    """PHASE

    phase says : 얼마나 회전했는가!(각도)

    x(t) = sin(2πft + ϕ) <= 이 식에서의 ϕ가 위상을 가리킴(phase)

        ex. 1000Hz 의 파형은 한 주기가 1ms 
                : 90도는 0.25ms
            500Hz 의 파형은 한 주기가 2ms 
                : 90도는 0.5ms
            => 같은 주파수라도 90도 변화시 위상이 다름

        **필터는 주파수마다 다른 위상을 줌 -> 주파수마다 도착시간이 달라짐

        ex. kick drum 은 여러 주파수를 동시에 가지고 있는데, 
        필터가 각각의 주파수를 다른 ms 만큼의 딜레이를 준다면 주파수 마다 도착시간이 달라서
        원래 있던 트랜지언트가 퍼지게 됨 

        => linear phase : 모든 주파수를 같은 시간만큼 지연시킴
        => minimum phase : 주파수마다 다른 시간만큼 늦추지만, 그 지연을 최소화함 
    
    [THE PROBLEM] : waveform distorton
    => 각 주파수가 각각 다른 시간으로 도착하게 되면서 impulse 가 흐릿해질 수 있음

    **Linear vs minimum phase
    1) Linear phase = ALL frequencies delayed equally
        - Preserves waveform shape
        - Punch and clarity maintained
        (FIR) => symmetrical coefficients 

        - Definition : φ(ω) = - τ * ω
        - 위상이 주파수에 비례 (phase is proportional to frequency)
        => All frequencies delayed by SAME amount
            - 1000Hz delayed 10ms
            - 2000Hz delayed 10ms
            - 10000Hz delayed 10ms
            => Relative timing preserved

            : 파형형태 보존
            : No temporal distortion
            : Transients stay 'punchy'
            : Pre-ringing ! (Impulse ring starts Before arrival)


    
    2) Minimum phase = Different delays per frequency
        - waveform changes slightly
        - But no pre-ringing artifacts
        - Natural sounding
        (IIR) => all of them 

        - Definition : Phase varies with frequency (위상이 주파수 별로 다름)
            - 1000Hz delayed 5ms
            - 2000Hz delayed 7ms
            - 10000Hz delayed 10ms
            => Relative timing changes

            : No pre-ringing
            : Post-ringing only 
            : Transients slightly blurred


    [The TRADEOFF] 
    1) Linear phase 
    2) Minimum phase 

    """

    """PHASE REPRESENTATION (위상 표현방법)

    - Phase is an ANGLE (위상은 각도)
        => represented in radians
        - -π to +π (most common)  / 0 to 2π

        ex. 1000Hz 의 1 cycle 은 1ms
        - 90도 phase : 0.25ms delay
        - 180도 phase : 0.5ms delay 

    [THE PROBLEM] : Phase Wrapping

    when phase crosses ± π, it 'wraps around'
    π → -π (discontinuous jump!)

    : 실제로는 연속적이지만 표현상 끊어짐 보임

    [SOLUTION] : phase unwrapping
    'Unwrap' the phase to make it smooth
        - Remove the discontinous jumps
        - Result: Smooth line showing true phase change
        In code
            - phase_wrapped = np.anlge(H)
            - phase_unwrapped = np.unwrap(phase_wrapped)
    
    """

def visualize_linear_vs_minimum():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Create linear phase FIR (symmetric coefficients)
    # 여기서 계수 출력
    fir_linear = signal.firwin(101, 0.3) 
        #symmetric. fir window method
        #window method 로 FIR 필터 계수 만들어줘 (계수 101개 나옴)
        # y[n] = b0​x[n] + b1​x[n−1] + b2​x[n−2] + ... + b100​x[n−100] 이렇게 입력 101개나 참조해야함
        # 계수모양 - sinc, 좌우 대칭
        # fir_linear[0] 과 fir_linear[100] 이 양 끝으로 같고
        # fir_linear[50] 이 가장 큼 -> 가운데 = sinc 꼭대기
        # 대칭적인 딜레이! 
        # group delay = (101-1) / 2 = 50 샘플 만큼의 지연이 모든 주파수에서 일어남 

        #배열 한개 : 숫자 101개 들어있는 (numtaps : 101) : 계수 b
        #분모는 [1] : fir 은 극점이 없으므로 

    # Create minimum phase IIR (Butterworth)
    b_iir, a_iir = signal.butter(6, 0.3)
        #위 아래 둘다 0.3은 cutoff
        #0-1 까지 중 1은 nyquist 가 됨.
        # 그러면 44100의 22050 이 nyquist. 그리고 그 중 30%지점인 7.2kHz 대역이 여기서의 cutoff 

    # Impulse
    impulse = np.zeros(300)
    impulse[100] = 1 # impulse at sample 100
        # 1 의 (impulse) 앞뒤를 모두 보기 위해서 중간에 1 배치

    # Get response (시간에 따른 출력)
    # 여기서 impulse -> 차분방정식 -> 출력을 나타냄
    # 여기서 각 샘플의 response 출력 (여기서는 impulse response)
    ir_linear = signal.lfilter(fir_linear, [1], impulse)    #fir 은 계수 b 101개 (좌우 대칭인 계수 배열)
    ir_iir = signal.lfilter(b_iir, a_iir, impulse) 
        # iir 은 계수 7개 => iir 이 훨씬 적인 계수로 필터 만듦
        # iir 은 출력도 사용하기 때문에 b, a 계수 둘다  필요

    # Frequency response (주파수별 복소수 출력!!!!! - 주파수 응답)
    w, h_linear = signal.freqz(fir_linear, [1], worN=500)
    w, h_iir = signal.freqz(b_iir, a_iir, worN=500)
        # w - 주파수 배열(0 ~ 파이, 500개) 
        # h - 복소수 배열(500개, 그 주파수에서 필터가 뭘 하는지?) -> h는 복소수 (크기+위상정보)
        # 그 h를 가지고 -> 크기(np.abs(h)) : 그 주파수를 몇배로 통과시키는지
        #               위상(np.angle(h)) : 그 주파수를 얼마나 밀어내는지

    # Phase response 
    phase_linear = np.unwrap(np.angle(h_linear))
    phase_iir = np.unwrap(np.angle(h_iir))
        #np.unwrap : 위상(각도)이 +/- π 경계에서 뚝뚝 끊기는걸 매끄럽게 이어붙이는 함수 
        #np.angle(h) 가 각 주파수의 위상(각도)를 뽑아줌 => 이 함수는 각도를 -파이 ~ +파이 범위 안에서만 표현함
        #필터는 위상이 주파수가 올라가면서 계속 누적돼서 내려감. 그러다 -pi를 넘어가면 np.angle 이 각도는 한바퀴 돌면 같은 자리 라고 생각해서
        #+pi 로 확 점프시켜버림
        #angle(h)만 쓰면 위상이 +/- pi 경계에서 톱니처럼 끊기는걸 
        #np.unwrap 으로 보정하는것

    # Plot 1 : linear phase impulse
    ax = axes[0, 0]
    ax.stem(ir_linear, basefmt = ' ')
    ax.axvline(100, color='red', linestyle='--', alpha=0.5, label='Impulse location')
    ax.set_ylabel('Amplitude')
    ax.set_title('Linear Phase FIR\nImpulse Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.text(0.5, 0.9, 'Pre-ringing BEFORE!\n(rining before 100)', 
           transform=ax.transAxes, fontsize=8,
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Plot 2: Minimum phase impulse
    ax = axes[0, 1]
    ax.stem(ir_iir, basefmt=' ')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Impulse location')
    ax.set_ylabel('Amplitude')
    ax.set_title('Minimum Phase IIR\nImpulse Response')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.5, 0.9, 'Post-ringing AFTER!\n(ringing after 0)', 
           transform=ax.transAxes, fontsize=8,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 3: Comparison zoomed
    ax = axes[0, 2]
    # Shift FIR to same location
    ir_linear_shifted = np.roll(ir_linear, -50) #adjust for DF symmetry
        # 배열회전시킴 . 지금 ir_linear 의 출력은 100 에서 -> 150(제일 중심이)으로 밀려남(linear phase. group delay 때문에)
        # 근데 이렇게 np.roll(ir_linear, 50) 하게 된다면 150-250 으로 sinc 의 계수가 존재하게 된다
        # 근데 그런 ir_linear_shigted 를 :150 까지만 본다면 아무것도 안보이게 됨 
        # 그래서 np.roll(ir_linear, -50) 으로 수정!
    ax.plot(ir_linear_shifted[:150], linewidth=2, label='Linear phase', color='blue')
    ax.plot(ir_iir[:150], linewidth=2, linestyle='--', label='Minimum phase', color='red')
    ax.axvline(100, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Amplitude')
    ax.set_title('Comparison\n(notice pre vs post)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
        # iir, fir 의 post, pre-ringing 유무 비교! 

    # plot 4: Linear phase response
    ax = axes[1, 0]
    ax.plot(w/np.pi, 20*np.log10(np.abs(h_linear)+1e-10), linewidth=2, color='blue')
        # w 가 0부터 pi 까지 였으므로 0-1 까지의 범위로 x 축 나오게 됨
        # 20*np.log10 .. 은 h_linear 를 dB 로 변환한 것
    ax.set_ylabel('Magnitude(dB)')
    ax.set_title('Linear Phase\nMagnitude Response')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(-60, 5)

    # plot 5: Minimum phase response
    ax = axes[1, 1]
    ax.plot(w/np.pi, 20*np.log10(np.abs(h_iir)+1e-10), linewidth=2, color='red')
    ax.set_ylabel('Magnitude(dB)')
    ax.set_title('Minimum Phase\nMagnitude Response')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim(-60, 5)

    # plot 6: Phase comparison
    ax = axes[1, 2]
    ax.plot(w/np.pi, phase_linear, linewidth=2, label='Linear (straight)', color='blue')
    ax.plot(w/np.pi, phase_iir, linewidth=2, linestyle='--', label='Minimum(curved)', color='red')
    ax.set_ylabel('Phase(radians)')
    ax.set_xlabel('Normalized Frequency')
    ax.set_title('Phase comparison\n(linear=straight line)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    """ plot 4,5,6 비교

    1) plot 4 : FIR 의 계수가 훨씬 많기 떄문에 기울기가 더 가파름. 자유도 높음 

    2) plot 5 : IIR 은 6개의 pole 만 존재함. 비교적 완만하게 떨어짐
    => firwin(11)을 써서 계수 11개만 썼다면 fir filter 또한 완만해질 것.


    3) plot 6 : FIR 은 지금 (101-1)/2 이렇게 50 samples 의 지연이 발생한다.
    => 모든 주파수가 동일한 시간으로 지연되고 있는것. (주파수에 대해서 위상이 직선으로 변함)
    => IIR - butterworth는 pole 이 있음. pole 마다 회전하는 속도가 달라서 
    위상이 직선처럼 주파수 마다 동일하게 직선이 될수 없다.

    위에서 FIR 의 계수와 butter 의 차수를 바꾸면 기울기는 바뀌지만
    => 여전히 linear phase 는 phase response 가 직선의 형태로 만들어짐.

    => 그리고 phase response 가 측정되는것이 transition band 까지인 이유 또한
    그 이후는 magnitude 가 거의 0이기 때문에 그 위상이 실제 신호에는 거의 영향을 주지 않기 때문.
    
    """


    plt.tight_layout()
    plt.show()

visualize_linear_vs_minimum()