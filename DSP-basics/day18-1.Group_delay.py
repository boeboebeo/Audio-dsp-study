"""
==============================================
DAY 18: Group Delay & All-pass Filters
==============================================

MATHEMATICAL PREREQUISITES:
- Derivatives (미분): Rate of change of phase
- All-pass transfer function structure
- Cascade systems (다단 연결)

KEY CONCEPT:
Group Delay = "Speed" at which signal travels through filter
Different frequencies travel at different speeds
= Temporal smearing of transients

All-pass filter = Only changes phase, NOT magnitude
Uses: Compensate for phase distortion, create effects
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_group_delay():
    """what is group delay? 
    
    Group delay 

        - Formula : τ_g(ω) = -dφ(ω) / dω
                    (Negative derivative of phase)

            +Meaning : How many samples does each frequency delay? 
            (각 주파수가 몇 샘플 지연되는가?)

    [THE PROBLEM] Temporal Smearing (시간적 번짐)

    Imagine sharp transient (like impulse)
        - Contains all frequencies
        - All frequencies arrive at DIFFERENT TIMES
        - Transient becomes Blurred!

    Audible as:
        - Less punch
        - muddy bass
        - unclear transients
        
    [THE IDEAL] Constant Group delay
    If all frequencies have same group delay?
        - all arrive at same time
        - transient shape preserved
        - punch and clarity maintained
        - when do you get constant group delay? : Linear phase filters
    
    """

def calculate_group_delay():
    # Visualize and Calculate 

    # Design filter
    # FIR (linear phase) - should have constant group delay
    fir_linear = signal.firwin(51, 0.3)
        # symmetric 

    # IIR (minimum phase) - variable group delay
    b_iir, a_iir = signal.butter(6, 0.3)
        # IIR 6th order 

    # Calculate group delay
    w_fir, gd_fir = signal.group_delay((fir_linear, [1]), w=np.linspace(0, np.pi, 500))
    w_iir, gd_iir = signal.group_delay((b_iir, a_iir), w=np.linspace(0, np.pi, 500))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # plot 1: FIR group delay (should be flat)
    ax = axes[0, 0]
    ax.plot(w_fir/np.pi, gd_fir, linewidth=2.5, color='blue')
    ax.set_ylabel('Group delay (samples)')
    ax.set_ylim(20, 30)
        # matplotlib 은 변화값이 극도로 작을때, 자동으로 offset notation 으로 표기하는데 (y축을)
        # ylim 없다면 1e-8 + 2.5e1 이렇게 처리함 
        # 2.5 e1 = 2.6 * 10^(1) = 25
        # 1e-8 = 10 * (-8) = 0.00000001 
        # 이렇다는건 실제 y = 0 인 눈금은 실제로는 25.00000000
        # 눈금 1 = 25.00000001 을 표기함
        # 매우 변동이 미세하기 때문에 이렇게 확대해서 보여줌
    ax.set_title('Linear phase FIR\nGroup delay')
    ax.grid(True, alpha=0.3)

    ax.text(0.5, 0.9, 'FLAT!\n(same delay with all frequencies)', 
           transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 2: IIR group delay (should vary!)
    ax = axes[0, 1]
    ax.plot(w_iir/np.pi, gd_iir, linewidth=2.5, color='red')
    ax.set_ylabel('Group delay (samples)')
    ax.set_title('Minimum phase iir\nGroup delay')
    ax.grid(True, alpha=0.3)

    ax.text(0.5, 0.9, 'VARIABLE!\n(different delay with frequencies)', 
           transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
        # Plot 3: Comparison
    ax = axes[1, 0]
    ax.plot(w_fir/np.pi, gd_fir, linewidth=2.5, label='FIR (constant)', color='blue')
    ax.plot(w_iir/np.pi, gd_iir, linewidth=2.5, linestyle='--', label='IIR (variable)', color='red')
    ax.set_ylabel('Group Delay (samples)')
    ax.set_xlabel('Normalized Frequency')
    ax.set_title('Comparison\n(FIR is flat, IIR varies)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = """
    GROUP DELAY SUMMARY
    ──────────────────
    
    What it measures:
    • Sample delay at each frequency
    • Calculated from phase derivative
    • τ_g(ω) = -dφ/dω
    
    Flat group delay :
    • All frequencies same delay
    • Transient preserved
    • Punchy sound
    • Linear phase filters have this!
    
    Variable group delay :
    • Different frequencies delayed differently
    • Transient blurred
    • Muddy sound
    • Minimum phase filters have this
    
    In practice:
    • Check GD for mastering filters
    • Constant GD = better transients
    • Especially important for drums
    """
    
    ax.text(0.05, 0.95, summary, fontsize=8.5, verticalalignment='top',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # group delay 의 정의
    # = 각 주파수 성분이 필터를 지나면서 시간이 얼마나 지연되는가 (샘플 또는 초 단위)
    # group delay 는 FIR, IIR 에 둘다 존재하며 모양이 다름 
    # group delay = τg​(ω) = D (상수, fir 에서는) 
    

    plt.tight_layout()
    plt.show()

def understand_allpass_filters():
    """All-pass filter : Only change phase( not magnitude )
    
    Transfer function(1st order):
        H(z) = (a + z^(-1) / (1 + a*z^(-1))
        where 0 < a < 1

        **이렇게 transfer fuction 이 배치되어 있어야 단위원 기준으로 거울처럼 배치됨
        ex. Pole = 0.6 이라면 Zero = 1/0.6 에 있다.
            => 이렇게 해야 magnitude 가 완벽하게 상쇄됨
            + Pole 은 크기를 키우려 하고
            + Zero 는 크기를 줄이려 하니까 => 둘이 정확히 상쇄해서 항상 magnitude 는 1이 된다. (변화 x)

        but, phase 는 바뀐다.
        크기는 pole 이 +3dB 만들때, zero 는 -3dB 처리하지만
        시간은 pole 이 조금 늦출때, zero 도 조금 늦춘다. => 위상은 계속 누적됨

    + Magnitude property
        |H(e^jω)| = 1 for ALL frequencies
        (모든 주파수에서 크기 = 1)

    + Phase property
        Angle H(e^jω) varies with frequency
        (위상은 주파수에 따라 변함)

    [what does it do?]

    Input signal -> All-pass -> Output signal

        - Phase compensation (위상보정)
        - Group delay equalization
        - Creating phase effects
        - Converting linear <-> minimum phase

    [pole-zero pattern]

    all-pass filters have special property

        - Pole at z = a
        - Zero at z = 1/a
        - Zero is Outside unit circle (영점이 단위원 밖에)

    This unusual structure
        - keeps magnitude flat (|H| = 1)
        - But changes phase 

    """

def create_allpass_filter():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Three all-pass filters with different 'a' values
    a_values = [0.3, 0.7, 0.9]

    for idx, a in enumerate(a_values):
        # All-pass transfer function
        b = np.array([a, 1]) # Numerator
        denom = np.array([1, a]) # Denominator (분모)
        
        # Freuqeuncy response
        w, h = signal.freqz(b, denom, worN=500)
        magnitude = np.abs(h)
        magnitude_db = 20*np.log10(magnitude + 1e-10)
        phase = np.unwrap(np.angle(h))

        # plot magnitude
        ax = axes[0, idx]
        ax.plot(w/np.pi, magnitude_db, linewidth=2.5, color='blue')
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel('Magnitude(dB)')
        ax.set_title(f'All pass (a= {a}\n Magnitude)')
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.8, f'FLAT! |H|=1 always', transform=ax.transAxes,
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # plot phase
        ax = axes[1, idx]
        ax.plot(w/np.pi, phase, linewidth=2.5, color='red')
        ax.set_ylabel('Phase(radians)')
        ax.set_title(f'All-pass (a={a}\n Phase)')
        ax.grid(True, alpha=0.3)

        # More annotations for a=0.9 (most dramatic)
        if a == 0.9:
            ax.text(0.5, 0.1, 'Steeper phase change', 
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.show()

    """
    All pass filter {idx+1}: a = {a}
    H(z) = {a} + z^(-1) / 1 + {a}*z^(-1)

        - All magnitudes are flat (|H| = 1)
        - Phase changes with frequency
        - Larger 'a' = stronger phase effect
        
    """

def cascade_allpass_for_arbitrary_phase():
    # 원하는 위상응답 만들기

    """
    Single all-pass : Limited phase change
    (all-pass 하나만으로는 위상을 변화시키는것에 한계가 있다)
        => 조절할 수 있는건 pole 의 위치(a) 하나 뿐이기 때문
    Cascade (chain) them : Arbitrary phase response
        => all pass filter 여러개를 직렬로 연결함

        - Formula
        : H_total(z) = H1(z) * H2(z) * H3(z) * ...

            => 직렬연결이므로 전달함수는 곱함
            (DSP 에서 직렬연결은 항상 전달함수의 곱이다.)
        
        Each Hi ia all-pass
            - Magnitudes multiply: 1*1*1* ...= 1
                => 근데 진폭은 각각 1이기 떄문에 곱하고곱해도 여전히 1나옴
            - Phase add: φ1 + φ2 + φ3 + ... 
                => 근데 복소수는 곱하면 위상은 더해짐
                    e^(jϕ1) * e^(jϕ2) = ej^(ϕ1+ϕ2)

    IIR filter has variable group delay 
        : 각각 group delay 가 주파수마다 다름
    => Design all-pass cascade to compensate
    => Result: all-pass filter 는 group delay 를 원하는 방향으로 조절할 수 있는 도구

    ex. butterworth 의딜레이가 100Hz 는 1 sample delay, 1000Hz 는 5 sample delay 라면
    => butterworth 뒤에 all pass 를 몇개 붙임 
     + 100hz +4 sample, 1000Hz 0 sample 이 되도록 만든다

    =>그럼 전체는 100Hz 5 sample, 1000Hz 5 sample 로 동일해짐

    **실제 사용 예시
        - 라우드스피커 크로스오버: 우퍼와 트위터의 시간정렬
        - 마이크 배열: 채널 간 시간 맞춤
        - 통신 시스템: 위상 왜곡 보정
        - 디지털 이퀄라이저: 크기 유지하면서 위상만 수정
        - Phaser: 여러개의 all-pass 를 직렬로 연결해서 여러개의 노치를 만듦
    
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Design 3 all-passes 
    a1, a2, a3 = 0.3, 0.6, 0.8

    # Individual responses
    w = np.linspace(0, np.pi, 500)

    b1, d1 = np.array([a1, 1]), np.array([1, a1])
        # 이게 뭘까
    b2, d2 = np.array([a2, 1]), np.array([1, a2])
    b3, d3 = np.array([a3, 1]), np.array([1, a3])

    _, h1 = signal.freqz(b1, d1, w)
    _, h2 = signal.freqz(b2, d2, w)
    _, h3 = signal.freqz(b3, d3, w)

    h_cascade = h1 * h2 * h3

    # plot 1: Magnitude ( all 1 )
    ax = axes[0]
    mag1_db = 20*np.log10(abs(h1) + 1e-10)
    mag2_db = 20*np.log10(abs(h2) + 1e-10)
    mag3_db = 20*np.log10(abs(h3) + 1e-10)
    mag_cascade_db = 20*np.log10(np.abs(h_cascade) + 1e-10)

    ax.plot(w/np.pi, mag1_db, linewidth=2.5, label='AP1', alpha=0.5)
    ax.plot(w/np.pi, mag2_db, linewidth=1, label='AP2', alpha=0.5)
    ax.plot(w/np.pi, mag3_db, linewidth=1, label='AP3', alpha=0.5)
    ax.plot(w/np.pi, mag_cascade_db, linewidth=1, label='Cascade', color='red')
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel('Magnitude(dB)')
    ax.set_title('Cascaded All-pass: Magnitude\n(all =1 , always)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # plot 2: Phase (adds together)
    ax = axes[1]
    phase1 = np.unwrap(np.angle(h1))
    phase2 = np.unwrap(np.angle(h2))
    phase3 = np.unwrap(np.angle(h3))
    phase_cascade = np.unwrap(np.angle(h_cascade))

    ax.plot(w/np.pi, phase1, linewidth=1, label='AP1', alpha=0.5)
    ax.plot(w/np.pi, phase2, linewidth=1, label='AP2', alpha=0.5)
    ax.plot(w/np.pi, phase3, linewidth=1, label='AP3', alpha=0.5)
    ax.plot(w/np.pi, phase_cascade, linewidth=2.5, label='Cascade', color='red')
    ax.set_ylabel('Phase (radians)')
    ax.set_title('Cascaded All-pass: Phase\n(phases add together!)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
            
cascade_allpass_for_arbitrary_phase()

"""all-pass filter 사용예시

1) phase correction (위상보정)
: multiple processing stages -> phase drift

    ex. comp -> eq -> saturation -> crossover -> limiter 
    이렇게 다양한 플러그인을 걸었을때 

    EQ도, 컴프안의 사이드 체인 필터도, crossover 도 위상을 바꾼다. 
    => 원래보다 주파수마다 위상이 많이 틀어짐

    => so, 마지막에 all-pass filter 를 하나 넣어서 틀어진 만큼 반대로 위상만 돌려줌
        ex. multiband comp의 low, mid, high 각각의 대역에 위상이 다다르게 갖게 되는데,
        그걸 다시 합칠때 all pass 를 써서 위상을 맞춤

2) Group delay compensation
: Butterworth EQ 를 만들었는데, 각 주파수의 group delay 는 주파수 별로 다르다.
=> 트랜지언트가 흐려질 수 있으므로 all-pass 를 설계해서 추가 delay 를 준다. 
    + 모든 주파수가 다 동일한 시간에 도착할 수 있게 원래 덜 지연있던거를 더 지연시켜버림

3) Stereo phase adjustment
 : 왼족오른쪽 채널중 오른쪽을 좀 늦춰서 스테레오 효과를 만들기도..


4) minimum <-> linear phase conversion
 **부족한 phase 를 all pass filter 로 채우면 

 minimum phase EQ + all-pass => phase 만 수정 

    => linear 에 가까운 응답! 가질 수 있다.


    !! 매우 복잡함..뭐 단순히 linear phase fir 을 All-pass 몇개만 붙여서
    그대로 만들 수 있는것은 아니다.



"""






# calculate_group_delay()
# create_allpass_filter()