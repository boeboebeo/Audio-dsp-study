"""
==============================================
DAY 11: Z-Transform & Transfer Functions
==============================================

MATHEMATICAL PREREQUISITES (수학적 기초):
    - Complex numbers (복소수) : a + bi 형태
    - Power series (멱급수) : a0 + a1*x + a2*x^2 + ...
        - 중심 c 가 0인 경우가 maclaurin series 
        - 이 들의 응용이 Taylor series 
    - Basic algebra (기본대수)

KEY CONCEPT (핵심개념):
    - Z-transform is just a way to represent digital signals as equations
    (digital signal -> 방정식으로 표현)

EXAMPLE:
    Signal x[n] = [1, 2, 3] (3개의 샘플)
    ↓ Z-Transform으로 변환하면
    X(z) = 1*z^0 + 2*z^-1 + 3*z^-2 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_z_transform_basics():
    """
    from scratch

    ===what is Z-transform?===
    Z-transform converts a time-domain signal into a frequency-domain eqation.
    It's basically another way to write/analyze digital signals

    FORMULA:
    X(z) = x[0]*z^0 + x[1]*z^-1 + x[2]*z^-2...
         = x[0] + x[1]*z^-1 + x[2]*z^-2 ...

    z^-1 : 1 sample delay 
    z^-2 : 2 sample dalay

    EXAMPLE:
    if input is x[n] = [1, 2, 3]
    Then X(z) = 1 + 2*z^-1(1 sample delay) + 3*z^-2(2 sample delay)

    It's just notation (표기법)

    =======================================================

    H(z) = transfer function = Output(z) / Input(z)
                             = Y(z) / X(z)

    Example 
    input x[n] = [1, 0, 0, ....] -> impulse
    output y[n] = [0.5, 0.5, 0, 0, ....]

        => 그럼 이 필터는 y[n] = 0.5 * x[n] + 0.5 * x[n-1] 
            (FIR moving average filter. 평균필터)

        ==================================================

        **이걸로 어떻게 주파수 응답도 알수있음? 
        그럼 이거에 복소지수 입력을 넣으면? (z^-1 = e^(-iw))

        H(z) = 0.5 + 0.5*e^(-iw)  (z^-1 대신 e^(-iw) 대입)
        H(e^iw) = 0.5 + 0.5*e^(-iw)


        + 그럼 이제 주파수마다 계산해 볼 수 있음 

        1) w = 0 (DC) -> 진동 전혀 없음. 주파수 = 0Hz => DC
        ( 회전 안함 : 1 1 1 1 1 => DC (0Hz) )

        e^(-iw) => e^(-i0) = 1 
        H = 0.5 + 0.5*1 = 1
           => DC 는 100% 통과

           **오일러 e^(jθ) = cosθ + isinθ 여기서 
             cos0 + i *sin0 = 1 + 0 = 1 

        2) w = π 
            (매 샘플마다 180도 회전하는게 1 -> -1 -> 1 -> -1 ... 
            이게 내가 지금 가진 sample rate로 표현가능한 제일 높은 주파수 => nyquist freq)
            n=0 : e^i0 = 1
            n=1 : e^iπ = -1
            n=2 : e^i2π = 1
            n=3 : e^i3π = -1 
            ....

            **한 샘플마다 270도 회전하면 1 -> -i -> -1 -> +i 이렇게 회전
            => -π/2 로 회전하는것과 샘플 값이 완전히 같음 (디지털에서는 샘플 사이를 볼수 x)
            : π보다 큰 주파수는 새로운 주파수가 아니라, 이미 존재하는 더 낮은 주파수와 구별 할 수 없다 
                => Aliasing 

            ** 그러므로 디지털에서의 주파수는 0 ≤ ω ≤ π 여기만 보면 됨 

        e^(-iπ) = -1
        H = 0.5 + 0.5*-1 = 0  
            => 가장 높은 주파수는 완전히 제거 

            **오일러 e^(jθ) = cosθ + isinθ 여기서 
             cosπ + isinπ = -1 + 0 = -1 (단위원 왼쪽 끝)

           **만약 2/π 라면 
             cos2/π + isin2/π = 0 + i*1 = i (90도 위쪽)
           **만약 -2/π 라면
             e^(-i2/π) = -i (-90도 아래쪽)


            => 결국 H(z) 는 주파수 응답을 알려줌 


        ==================================================

    X(z) = 1
    Y(z) = 0.5 + 0.5*z^-1
    H(z) = Y(z)/X(z) = (0.5 + 0.5*z^-1) / 1 = 0.5 + 0.5*z^-1

    => H(z) tells us WHAT THE FILTER DOES to any input! 
    (H(z)는 필터가 입력신호에 뭘 하는지 알려줌)

    """

def explore_poles_and_zeros():
    """
    Understanding "Poles" and "Zeros"

    H(z) = Numerator(z) / Denominator(z)
         = (b0 + b1*z^-1 + ...) / (1 + a1*z^-1 + ...)

    ZEROS:
    Values of z where Numerator = 0 

    POLES:
    Values of z where Denominator = 0

    WHY CARE?
    - Poles determine STABILITY (안정성 결정)
    - Zeros determine FREQUENCY RESPONSE (주파수응답 결정)
    
    STABILITY RULE (안정성 규칙):
    System is stable IF ALL POLES are inside the unit circle
    (|pole| < 1 이어야 안정적)
    
    Unit circle = circle with radius(반지름) 1 centered at origin
    (원점을 중심으로 반지름이 1인 원)

    =======================================================

    1) CONCRETE EXAMPLE (구체적인 예)

    H(z) = 0.5 + 0.5*z^(-1) / (1 - 0.9*z^-1)
            -Numerator          -Denominator

    + Finding ZEROS (numerator = 0)

        0.5 + 0.5*z^(-1) = 0
        0.5*z^(-1) = 0.5
        z^(-1) = 1
        z = -1 (this is a ZERO)

    + Finding POLSE (denominator = 0)

        1 - 0.9*z^-1 = 0
        -0.9*z^-1 = -1
        z^-1 = 10/9
        z = 9/10 = 0.9 (this is a POLE)

        => pole 의 절대값 |0.9| < 1 이므로 this system is STABLE 

    if we feed(입력v) it any reasonable input
    the output won't explode to infinity
    it will decay(감쇠v) to silence eventually

    """

def create_and_visualize_simple_filter():
    """
    Create a simple filter and show its poles/zeros 

    This shows exactly where:
    b1 = [0.1]  ----  분자 계수 (상수항 하나)
    a1 = [1, -0.9] ----  분모 계수 (1(z^0), -0.9(z^-1))
    come from! (이 값들이 어디서 나오는지)

    ==================================================

    1) Decide on(평가v, 결정v) filter structure

    We want: H(z) = 0.1 / (1 - 0.9*z^-1) <= 1st-order filter(1차 필터)
        - b1 = 0.1
        - a1 = 1, 0.9

        In standard form(표준형):
        Numerator coefficients (b) = [0.1]
        Denominator coeffieicients (a) = [1, -0.9]
                + 1 : normalization 
                + -0.9

    """

    """
    우선 우리가 만들고자 하는 필터는 현재 입력 + 이전 출력의 90% 임

    y[n] = 0.1x[n] + 0.9y[n-1] <= IIR low-pass filter
    (아직까지 시간영역임)

        -> z-transform
        Y(z) = 0.1X(z) + 0.9z^(-1)Y(z)
        Y(z) - 0.9z^(-1)Y(z) = 0.1X(z)
        Y(z)(1 - 0.9z^(-1)) = 0.1X(z)
        Y(z)/X(z) = 0.1 / (1 - 0.9z^(-1)) 

    근데 H(z) = Y(z)/X(z) 따라서 H(z) = 0.1 / (1 - 0.9z^(-1)) 
        => 그냥 시간영역의 식을 다른 언어(z)로 번역한 것 

    컴퓨터는 분자 0.1 을       b = [0.1] 로 저장
    분모 (1 - 0.9z^(-1)) 을  a = [1, -0.9] 로 저장 


    + signal.tf2zpk는 위 배열 보고, 분모가 0되는 점(pole) / 분자가 0되는 점(zeros) 계산하는 함수
    + signal.freqz는 w=0 -> 0.01π -> 0.02π 를 각각넣고 H 계산 후에 π 까지 반복한다. 
        => 그리고 크기를 저장함 ! (frequency response 표현해줌)
    
    """

    # create it
    b = np.array([0.1])     # numerator [0.1]
    a = np.array([1, -0.9]) # denominator [1, -0.9]

    # Extract poles and zeros
    z, p, k = signal.tf2zpk(b, a) 
        # tf2zpk = Transfer Function(전달함수) to zero-pole-gain(k)
        # 전달함수에서 pole, zero 찾는 함수
        # 여기서는 분자가 상수 0.1 하나라서 zeros 가 없음

        # 분모 : 1 -0.9z^(-1) 는 출력이 자기자신을 얼마나 강하게 다시 사용하는가 
            # H(z) = 0.1 / 0 이라면 수학적으로 무한대로 발산함! => 출력이 엄청 커짐
            # resonance 생성됨 = pole (막대기? 극점)
            # -> feedback 의 흔적!
        # 분자 : 분자가 0이 된다면 출력이 0 배가 되기때문에 그 주파수는 완전히 사라짐 
            # H(z) = 0/무언가 = 0 
            # Zero -> 0 (영점)

        # ex. 1000Hz 를 올리고 싶다면 단위원의 1000Hz 방향에 근처에 pole 을 둠 (증폭)
        # ex. 1000Hz 를 없애고 싶다면 그 근처에 zero 를 둠 (감쇠)

    print(f"\nPoles : {p}")
    print(f"\nZeros : {z}")
    print(f"\nStability : All poles |z| <1? {np.all(np.abs(p) < 1)} ✓")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))

    # plot 1 : Pole-zero diagram
    ax = axes[0]

    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100) # 세타(θ). 0부터 2pi 까지를 100개로 나눠서 점을 찍은것 => 원 만들기 위함 
    unit_circle_x = np.cos(theta)
    unit_circle_y = np.sin(theta)
        # 위의 cos+sin 이 합쳐져서 
        # (1,0), (0.99, 0.06), ... (-1, 0) ... -> (1, 0) 의 원이 만들어 짐
    ax.plot(unit_circle_x, unit_circle_y, 'k-', linewidth=2, label = 'Unit circle')
    ax.fill(unit_circle_x, unit_circle_y, alpha = 0.1, color = 'gray')

    # plot poles (as X marks)
    ax.plot(np.real(p), np.imag(p), 'rx', markersize=15, markeredgewidth=3, label='Poles')

    # plot zeros (as 0 circles)
    ax.plot(np.real(z), np.imag(z), 'bo', markersize=10, markeredgewidth=2,
            fillstyle='none', label='Zeros')
    
    ax.axhline(0, color='k', linewidth=0.5)
        #axis horizontal line ( 수평선 ) -> x 축
        #실수축
    ax.axvline(0, color='k', linewidth=0.5)
        #axis vertical line ( 수직선 ) -> y 축
        #허수축 그리기 
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
        # 원 찌그러 지지 않게 가로세로 비율 1:1고정
        # 가로 세로 비율 설정. set aspect ratio
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Pole-Zero Diagram\nX=poles (must be inside circle)\n0=Zeros')
    ax.legend()
        # 범례만들기 . 각 색이나 선이 무얼 의미하는지 표현해놓은 표

    for pole in p:
        ax.annotate(f'pole={pole:.2f}', xy=(np.real(pole), np.imag(pole)),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    for zero in z:
        ax.annotate(f'zero={zero:.2f}', xy=(np.real(zero), np.imag(zero)),
                   xytext=(5, -15), textcoords='offset points', fontsize=9)

    # plot 2 : frequency response 
    ax = axes[1]
    w, h = signal.freqz(b, a, worN=500) # get frequency response
        # 주파수 응답 H(e^(iw)) 를 계산해줌 
        # 0부터 π까지 500개의 주파수에서의 필터의 응답을 계산 함
        # w 는 0, 0.006π, 0.012π, ... π 까지의 배열이 들어있음
    magnitude = np.abs(h)
        # 여기서 h 는 복소수 (ex. 0.8+0.2j 이런 값들)
        # 근데 우리는 크기만 보고 싶으므로 abs(h) 하면 1까지의 크기중 몇 배 통과시키는지가 나옴
        # 여기서의 np.abs(h) : 원점과의 거리
        # 만약 z=3+4j 라면 |z| = 루트(3^2 + 4^2) = 5 
        # 필터는 주파수마다 위상+크기를 동시에 바꾸기 때문 
    magnitude_db = 20*np.log10(magnitude + 1e-10) 
        # convert to dB
        # ex. x=0 DC 라면 여기서는 그냥 100% 통과
        # ex. x= 0.3 라면 nyquist freq 의 30% 쯔음 되는 주파수 => -1dB
        # ex. x = 0.8 라면 nyquist freq 의 80%정도 되는 주파수 이므로 많이 줄임

        # 따라서 H(z) = 0.1 / (1 - 0.9z^(-1)) 이 필터는 이전의 출력의 90%를 현재 입력에 더하고
        # 그러므로 급격하게 변하는 고주파는 평균화 되어 약해짐
        # 천천히 변하는 저주파는 거의 그대로 유지되므로 => low pass filter 

        # if, pole = 0.5 로 바꾸면 (지금은 0.9에서) 더 급격하게 떨어지게 됨
        # 😍! 여기서 1 pole = 6dB/oct
        # pole 의 위치 (ex. 0.8 + 0.3i) 는 필터의 성격을 결정함 -> 어디를 강조할지!
        # Pole dml 개수 는 필터의 차수(order)를 결정함! -> pole 이 많아질수록 감쇠 기울기가 더 가파름

    ax.plot(w/np.pi, magnitude_db, linewidth = 2.5, color='blue')
        # w/np.pi 로 나누는 이유는 π 까지의 값을 0-1 까지로 나누어서 표현하기 위함
        # 0 - π(rad/sample) 
        # 0 - 1(1은 nyquist frequency) => 0-1 로 정규화한것 !
        # ex. 0.5 는 0.5π = w 를 의미함
    ax.axhline(-3, color='r', linestyle='--', alpha=0.5, label='-3dB (cutoff)')
    ax.set_xlabel('Normalized Frequency (×π rad/sample)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Frequency Response\n(What this filter does to different frequencies)')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend()
    ax.set_ylim(-40, 5)

    plt.tight_layout()
    plt.show()

def demonstrate_stability():
    """
    show difference between STABLE and UNSTABLE systems
    """

    fig, axes = plt.subplots(2, 2, figsize = (12, 8))

    # case 1: STABLE ( pole inside circle )
    # pole = 0.8 (inside unit circle)
    b1 = np.array([0.1])
    a1 = np.array([1, -0.8]) # Pole at 0.8

    impulse = np.zeros(50)
    impulse[0] = 1
    y1 = signal.lfilter(b1, a1, impulse)
        # lfilter(b, a, x)는 계수 b, a로 정의된 차분방정식을 입력 x 에 한샘플씩 적용해서 출력을 내줌
        # impulse 는 0번째 인덱스만 1이고 뒤는 다 0인 배열인데, 그걸로 freq response 표현해냄

    z1, p1, _ = signal.tf2zpk(b1, a1)

    ax = axes[0, 0]
    circle_theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), 'k-', linewidth=2)
    ax.fill(np.cos(circle_theta), np.sin(circle_theta), alpha=0.1, color='gray')
    ax.plot(np.real(p1), np.imag(p1), 'rx', markersize=15, markeredgewidth=3)
    ax.axhline(0, color='k', linewidth=0.5) #k : 검정 색(black - 마지막 글자)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('STABLE: Pole at 0.8\n(inside circle) ✓')
    ax.text(0, -1.3, '|Pole| = 0.8 < 1', ha= 'center', fontsize=11, fontweight='bold')

    ax = axes[0, 1]
    ax.stem(y1, basefmt=' ')
        #여기서 y1으로 y값만 주어서 x 축은 자동으로 배열의 인덱스로 채움 -> 0 - 49까지의 배열
        #basefmt=' ' : 공백. 아무스타일도 안주어서 baseline 숨겨버림
    ax.set_ylabel('Amplitude')
    ax.set_title('Impulse Response\n(decays to zero)')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.8, 'Signal dies away\n= Safe to use ✓',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # case 2: UNSTABLE (pole outside circle)

    b2 = ([0.1])
    a2 = ([1, -1.1]) # pole at 1.1 -> unstable

    y2 = signal.lfilter(b2, a2, impulse)
        # 실제로는 폭발하지만 여기서는 clipping 해서 visualization 처리
    # y2 = np.clip(y2, -10, 10)

    z2, p2, _ = signal.tf2zpk(b2, a2)

    ax = axes[1, 0]
    ax.plot(np.cos(circle_theta), np.sin(circle_theta), 'k-', linewidth=2)
    ax.fill(np.cos(circle_theta), np.sin(circle_theta), alpha=0.1, color='gray')
    ax.plot(np.real(p2), np.imag(p2), 'rx', markersize=15, markeredgewidth=3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlim(-1.5, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('UNSTABLE : Pole at 1.1\n(outside circle) x')
    ax.text(0, -1.3, '|pole| = 1.1 > 1', ha='center', fontsize=11, fontweight='bold', color='red')

    ax = axes[1, 1]
    ax.stem(y2, basefmt= ' ')
    ax.set_ylabel('Amplitude')
    ax.set_title('Impulse Response\n(Explodes!)')
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.8, 'Signal explodes\n=DO NOT USE x',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    

    plt.tight_layout()
    plt.show()

    

create_and_visualize_simple_filter()
demonstrate_stability()

"""SUMMARY
Z-Transform: Converts time signal to equation
X(z) = x[0] + x[1]*z^-1 + x[2]*z^-2 + ...

Transfer Function: H(z) = Y(z)/X(z)

Poles & Zeros

    -Poles: where Denominator = 0
    -Zeros: where Numerator = 0

Stability Rule
    
    - All poles must satisfy |pole| < 1

b and a coefficients:

    - b = numerator coefficients
    - a = denominator coefficients

"""

""" 
filter 의 안정성이란?
: 입력이 유한하면 출력도 유한하다

 ex. y[n] = 0.1x[n] + 0.9y[n-1] 이 필터식에서 
 입력이 impulse 로 딱 한번만 들어오게 된다면, 점점 작아진다

 ex. y[n] = 0.1x[n] + 1.1y[n-1] 이 필터식에서는 pole 이 1.1 이기 떄문에 
 출력이 점점 커진다. 

    => 그래서 |Pole| < 1 이여야 하는것

** 절대값인 이유는 원점과의 거리를 구할때 루트(a^2 + b^2) 를 시행하기 때문 

=> 시스템이 시간이 지나면서 에너지가 줄어드는지, 유지되는지, 커지는지를 결정하는 기준이 됨

"""