"""
==============================================
DAY 15: IIR Filter Design Methods
==============================================

MATHEMATICAL PREREQUISITIES:
    - Laplace transform: continuous-time version of Z-transform
    - Complex s-plane (복소 s-평면): where analog filters live
    - Bilinear transformation (쌍선형 변환): Bridge between analog and digital

KEY CONCEPT:
Digital IIR filters are designed by:
    1. Designing an Analog filter 
    2. converting to DIGITAL using bilinear transformation
This is much easier than designing digital filters from scratch
(아날로그로 설계하고(s-domain), 디지털로 변환(z-domain하면 쉬움)

**아날로그로 설계 = 연속시간 필터를 s-domain에서 먼저 만든다는 뜻 
( z-domain. z=e^(iw)말고! => 그리고 그 다음에 z-domain으로 변환하는것 )


**아날로그로 설계 -> 디지털로 변환하는 이유?
(why convert from analog to digital?)

+Two ways to design IIR filters

    - WAY1: Design directly in digial domain
    => hard, lots of constraints, no clear

    - WAY2: Design in analog, convert to digital
    => easy, well-understood theory, clear procedures, standard method

    **대부분의 아날로그 시스템이 미분방정식으로 표현되는데, 그 미분방정식을 디지털로 옮기는것이 가장 자연스러움
    ex. RC curcuit

    **아날로그(s-domain) 필터이론 -> 100년 넘은 전자회로 필터 설계들
            - 원하는 스펙을 정해서(ex. 4th order, LPF, COF 1000Hz) 
              => s-domain에서 그 스펙에 만족하는 극점들을 특정패턴으로 배치 -> H(s)완성
              => 그 다음 이 H(s)를 디지털 H(z) 로 변환함
              아래의 signal.butter 이 함수가 변환해줌 

              b, a = signal.butter(4, 1000, fs=44100)   # 아날로그 Butterworth 설계 → 디지털 변환까지 한 번에

            (무그 같은 ladder 디지털 애뮬레이션도 결국 아날로그 회로 -> 디지털 변환 임)
        + Butterworth (maximally flat)
        + Chebyshev Type 1 (passband ripple)
        + Chebyshev Type 2 (stopband ripple)
        + Elliptic (ripple in both, sharpest)
         => 각자 pros and cons 가 분명함 (trade-off)

+Bridge : Bilinear transformation 
    - Formula 
        s = (2/Ts) * (1 - z^(-1)) / (1 + z^(-1))

            - 여기서 T = 샘플링 주기 (시간 간격)
            ex. sample rate = 48000Hz 면 T = 1/48000 s
            (샘플과 샘플 사이의 실제 시간)

        => converts s-plane (analog) to z-plane (digital)
        => Preserves stability (안정성 유지)
        => Causes frequency warping (주파수 왜곡)
        => Result: Analog design => Digital implementation 

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 4개의 IIR analog filter design 종류(pros and cons) 잘 알아야 함

def design_iir_filters():
    
    #step1: specifications
    fp = 5000 #passband edge(Hz)
    fs = 6000 #stopband edge(Hz)
    Ap = 0.5 #Max passband ripple (dB) . max allowed
    As = 60 #Min stopband attenuation (dB) . minimum required (최소 요구사항)
    sample_rate = 44100

    #step2: Calculate required filter order
    #different formulas for each type:

    #Normalized frequencies
    wp = fp / (sample_rate/2)
    ws = fs / (sample_rate/2)
        # 이 정규화 주파수는 nyquist=1 로 두고 계산함
        # remez (0.5) 와는 다름 => fir optimal 계수 계산하던 함수

    # Calculate orders
    N_butter, Wn_butter = signal.buttord(wp, ws, Ap, As)
    N_cheby1, Wn_cheby1 = signal.cheb1ord(wp, ws, Ap, As)
    N_cheby2, Wn_cheby2 = signal.cheb2ord(wp, ws, Ap, As)
    N_ellip, Wn_ellip = signal.ellipord(wp, ws, Ap, As)
    # Filter 가 몇 차(order)여야 하는지, 컷오프 위치를 어디로 해야할지 계산해주는 함수
        # order : 이 스펙을 만족하려면 몇차 함수가 필요한지? => N 
        # 자연주파수 : 실제 설계에 쓸 컷오프 주파수 (정규화값) => Wn
        # pass/stopband freq passband 최대 손실, stopband 최소 감쇠
        # 위 네개의 값 넣어주면 그걸 만족하는 최소 차수와 컷오프를 알려줌

    print(f"\nRequired orders:")
    print(f"  Butterworth: order {N_butter}")
    print(f"  Chebyshev I: order {N_cheby1}")
    print(f"  Chebyshev II: order {N_cheby2}")
    print(f"  Elliptic: order {N_ellip} ← Lowest! (가장 낮음)")

    print(f"\nSavings with Elliptic:")
    print(f"    vs Butterworth: {N_butter - N_ellip} fewer poles")
    print(f"    (same performance, {((1-N_ellip/N_butter)*100):.4f}% less CPU)")

    #Design all types
    b_butter, a_butter = signal.butter(N_butter, Wn_butter)
        # transfer fuction 의 분모, 분자 계수 계산해줌
    b_cheby1, a_cheby1 = signal.cheby1(N_cheby1, Ap, Wn_cheby1)
        # 얘는 추가로 Ap 넣음 (passband 최대로 허용되는 리플 dB)
    b_cheby2, a_cheby2 = signal.cheby2(N_cheby2, As, Wn_cheby2)
        # 얘는 추가로 As 넣음 (stopband 최소로 작아지는게 이정도는 되야 한다는 dB)
    b_ellip, a_ellip = signal.ellip(N_ellip, Ap, As, Wn_ellip)
        # 얘는 추가로 Ap, As 둘다 넣음

        # signal.butter(...) 내부 처리과정
            # 차수, cutoff 를 넣어주면
            # => 아날로그 butterworth prototype 생성되고
            # => H(s) 생성. s-plane 에서의 전달함수
            # => Bilinear transform 을 거쳐서 H(z) 만들어냄
            # => H(z)
            # => 분자(b), 분모(a) 계수 계산 후 출력해줌

            # scipy 에서 analog=True 면 변환하지 않고 H(s)자체를 반환함


    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    filters = [
        ('Butterworth (order 7)', b_butter, a_butter, axes[0,0]),
        ('Chebyshev I (order 5)', b_cheby1, a_cheby1, axes[0,1]),
        ('Chebyshev II (order 5)', b_cheby2, a_cheby2, axes[1,0]),
        ('Elliptic (order 4)', b_ellip, a_ellip, axes[1,1]),
    ]

    for name, b, a, ax in filters:
        #Frequency response
        w, h = signal.freqz(b, a, worN=500, fs=sample_rate)
            # freqz 가 출력하는 w 의 범위는 0-_까지 
        mag_db = 20 *np.log10(np.abs(h) + 1e-10)

        ax.plot(w, mag_db, linewidth=2.5, color='blue')
        ax.axvline(fp, color='g', linestyle='--', alpha=0.5, label='Passband edge')
        ax.axvline(fs, color='r', linestyle='--', alpha=0.5, label='Stopband edge')
        ax.axhline(-Ap, color='g', linestyle='--', alpha=0.5)
        ax.axhline(-As, color='r', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 15000)
        ax.set_ylim(-100, 5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()

def understand_bilinear_transformation():
    """Bilinear transformation: how we convert analog to digital
    

    **Maps s-plane (analog) → z-plane (digital)

    Formula: (2/Ts)( 1-z^(-1) / 1+z^(-1) )
          => (2/Ts)( z-1 / z+1)   <- 위의 식의 분자 분모에 z 곱한 후, 풀면 이런 식 나옴 
           
           **풀이
            : ( 1-z^(-1) / 1+z^(-1) ) 여기만 
              z(1-z^(-1) / z(1+z^(-1) 이 나오게 되는데 여기서 
              z*z^(-1) 은 지수법칙으로 z^(0) = 1 
              => 따라서 z-1 이렇게 나오게 됨
        
        where Ts = 1/sample rate (sampling period)

    - 아날로그 필터 : s-plane 에서의 poles/zeros 를 디자인함
    - 디지털 필터 : z-plane 에서의 poles/zeros 를 디자인함

    ex. 
    + Analog pole at s = -1000
    => use bilinear to find digital pole location
        => get digital filter coefficients 

    [Frequency warping] 

    one issue : bilinear warps frequencies
    !! 주파수 왜곡이 발생할 수 있음 (주파수가 그대로 가지 않음)

    - Laplace 에서는 주파수가 s = jΩ 에서 움직임 (Ω = 아날로그 각주파수 (rad/sec))
    - Digital 에서는 주파수가 z = e^(jω) 위에서 움직임 (ω = 디지털 각주파수 (rad/sample))
    
    이 둘을 Biliear transform 으로 연결하면

        Ω = 2/T * tan(ω/2) 라는 식이 나오는데 이게 Frequency warping 의 원인

        => 선형이었다면 일정하게 대응될텐데
        => tan() 가 직선이 아니라 곡선이기 때문에 디지털로 변환시 같은 간격으로 변환이 안됨
        (고주파에서 더 심하게 변형)

        tan() : 90도 부근에서 갑자기 무한대로 치솟음 
                => 그래서 저주파수에서는 거의 안 변하지만, 높은 주파수에서는 매우 변함

    ** 그래서 엔지니어들은 pre-warping 을 함
    => 어짜피 Bilinear trnasform 시 휘어짐으로 처음부터 반대로 조금 비틀어서 설계 

        Ω = 2/T * tan(ωc/2)

        => 원하는 디지털 cutoff -> 먼저 아날로그 cutoff 계산 -> 그걸 butterworth 설계에 사용
        => Bilinear trasform => 원하는 cutoff 제대로 나옴! 

        원하는 디지털 Cutoff
                │
                ▼
        Pre-warping
        (탄젠트로 보정)
                │
                ▼
        Butterworth 설계(H(s))
                │
                ▼
        Bilinear Transform
                │
                ▼
        H(z)
                │
                ▼
        원하는 Cutoff가 정확하게 나옴

    - 근데 signal.butter() 같은 함수는 내부에서 자동으로 주파수 보정해줌! 
    

    
    """

def compare_stability_poles():
    """
    show where poles/zeros are for different filter types verify stability
    [verify : 확인v]
    """

    # [STABILITY CHECK] Pole locations

    # Design filters
    fp, fs = 0.2, 0.25  # pass, stop band freq - Normalized
    Ap, As = 0.5, 60  # pass band ripple, stop band attenuation

    b_butter, a_butter = signal.butter(4, (fp+fs)/2) 
    b_cheby1, a_cheby1 = signal.cheby1(4, Ap, (fp+fs)/2)
    b_ellip, a_ellip = signal.ellip(4, Ap, As, (fp+fs)/2)

    # Extract poles
    _, p_butter, _ = signal.tf2zpk(b_butter, a_butter)
    _, p_cheby1, _ = signal.tf2zpk(b_cheby1, a_cheby1)
    _, p_ellip, _ = signal.tf2zpk(b_ellip, a_ellip)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    filters = [
        ('Butterworth', p_butter, axes[0]),
        ('Chebyshev I', p_cheby1, axes[1]),
        ('Elliptic', p_ellip, axes[2]),
    ]

    for name, poles, ax in filters:
        # Draw unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit circle')
        ax.fill(np.cos(theta), np.sin(theta), alpha=0.1, color='gray')
            # x = rcos(theta), y = rsin(theta)
            # 단위원 자체가 복소평면에서의 |z| = 1 인 복소평면의 점들의 집합임
            # 복소수는 z = x + iy 이니까 
            # z = cos(theta) + i sin(theta)
            # 그러면 오일러 공식에 의해 e^(i*theta) = cos(theta) + i sin(theta)이고
            # 단위원위의 모든 점은 (cos(theta), sin(theta))로 표현되는 동시에
            # 복소평면위에서는 (Real, Imag)의 좌표이기도 함
            # => 좌표자체는 완전히 동일 

        # Plot poles
        ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=12, markeredgewidth=2.5)
            # 실수축, 허수축 각각에 pole 체크

        # Check stability
        all_stable = np.all(np.abs(poles) < 1)
            #모두 1보다 작은지! (단위원 안에 있는지 체크)
        
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')

        status = "STABLE ✓" if all_stable else "UNSTABLE x"
        ax.set_title(f'{name}\n{status}')


        if all_stable:
            ax.set_facecolor('#f0f0f0')
        else:
            ax.set_facecolor('#ffcccc')

    plt.tight_layout()
    plt.show()




# design_iir_filters()
compare_stability_poles()
    

"""need to know

저항(R)과 커패시터(C)의 성질
        ↓
RC Low-pass 회로

① Differential Equation (왜 RC 회로가 미분방정식인가?)

↓

② Laplace Transform (미분이 왜 s가 되는가?)

↓

③ Transfer Function H(s)

↓

④ Pole in s-plane

↓

⑤ Stability 
    - 여기까지가 아날로그 필터를 공부한 것

↓

⑥ Butterworth Filter
    - passband 가 최대한 평평 
    - 아날로그에서 설계한 '이상적인 성능에 가까운 low-pass filter

↓

⑦ Bilinear Transform

↓

⑧ H(z)

↓

⑨ Difference Equation

↓

⑩ Biquad

↓

⑪ Direct Form I / II

↓

⑫ RBJ Cookbook EQ



**Summary**
RC 회로
      ↓
미분방정식
      ↓
Laplace Transform
      ↓
H(s)
      ↓
Pole
      ↓
Bilinear Transform
      ↓
H(z)
      ↓
Difference Equation
      ↓
C++ 코드


① Voltage(전압)란?
      ↓
② Current(전류)란?
      ↓
③ 저항(R)은 왜 V=IR인가?
      ↓
④ Capacitor는 왜 전하를 저장하는가?
      ↓
⑤ 왜 i = C dv/dt 인가?
      ↓
⑥ RC Low-pass
      ↓
⑦ IIR
"""




