"""
==============================================
DAY 16: Filter Topologies (Direct Form, TDF, Cascade)
==============================================

[topology : 위상구조 or 연결구조]

MATHEMATICAL PREREQUISITES:
- Difference equations (차분방정식): From Day 12
- Block diagrams (블록다이어그램): Visual representation of equations
- Numerical stability (수치 안정성): Precision in computation

**세 방식은 무한 정밀도 에선 완전히 같은 필터인데, 컴퓨터에서는 유한한 정밀도(float)를 가지고
계산하기 때문에 반올림 오차가 쌓이는 정도가 달라짐 


KEY CONCEPT:
Same mathematical filter can be implemented many different ways!
[implement : 구현v]
Each way has different:
- Stability properties (안정성)
- CPU efficiency (효율성)
- Numerical precision (정밀도)
- Memory requirements (메모리)

[Numerical stability : 수치적 안정성]

Choose the right topology for your situation!
(상황에 맞는 위상구조 선택)
: 같은 필터를 컴퓨터 안에서 어떻게 계산할지에 대한 차이
    (같은 수식을 계산하는 방법이 여러가지 있음(계산 순서, 구조에 따른 차이))
    => 컴퓨터 안에서의 메모리 사용량, 연산량, 반올림 오차 등에 따라서 
    => 어떤 지연 버퍼를 써서 계산하는지


**what is a 'topology'?

Topology = way to implement a filter (필터 구현 방식)

H(z) = b0 + b1 * z^(-1) / 1 + a1 * z^(-1) 의 구조를 다양한 방법으로 구현할 수 있음

    ex. 8th-order IIR filter 
    Direct form I: Unstable due to rounding
    (라운딩 오차 때문에 불안정)
    Cascade of 2nd order sections : Very stable

Difference matters for
    - real-time audio (라이브처리)     
    - Fixed-point arithemetic (임베디드)
    - Long-duration processing (긴 신호 처리)


"""

def understand_direct_form():
    """
    Direct Form I and II
    (most straightforward but can be unstable) 
     - 가장 직관적이지만 불안정할 수 있음

    1) Direct form I
        : 입력 저장 x[n] -> x[n-1] -> x[n-2] .. 그리고 Multiply by b -> feedforward(입력들)
        : 출력 저장 y[n] -> y[n-1] -> y[n-2] .. 그리고 Multiply by a -> feedback(출력들)
        즉, 입력용 delay line, 출력용 delay line 이렇게 두개를 만든다 
        => 딜레이 두줄필요, difference equation 그대로 구현함
        => 입력 지연버퍼, 출력 지연버퍼를 따로 둠 (그 만큼 Delay 저장소 많이 필요)

    +structure 

        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2]
                - a1*y[n-1] - a2*y[n-2]

    +How it works?
        1. Input x[n] goes through feedforward (b coefficients)
        2. Output y[n] is fed back through feedback (a coefficients)
        3. both combined for final output 

    +Pros and cons
        - Pros : simple to understand, straightforward code (직관적인 코드)
        - Cons : can accumulate rounding errors, unstable for high-order filters
                , large intermediate values possible (큰 값이 나와서 오버플로우 위험)

    2) Direct form II
    : 입력 delay 와 출력 delay를 합칠 수 있지 않을까?
    => DF I에 비해서 딜레이 라인 절반으로 줄어듦. (두 지연버퍼를 공유하도록 함 
                                -> 지연 저장소 절반으로 줄어들어 메모리 효율적)
    but, feedback 안에서 숫자가 커졌다 작아졌다 하면서 반올림 오차가 생길 수 있다.

    **결국 입력 delay 와 출력 delay 는 한 샘플 저장 이라는 똑같은 일을 하고 있음
    (메모리 통일!)
    => DF-1 에서는 delay 안에 입력, 출력을 저장. 
    => DF-2 에서는 delay 안에 입력, 출력이 아닌 중간 계산값(state)를 저장한다.
    (feedback 과 feedforward 가 합쳐진 값이기 때문)

    근데 컴퓨터는 18.700000000을 사실은 18.69999976처럼 저장한다.
    => 그 다음 샘플 오차가 또 feedback 되고 그게 반복된다면 필터 자체는 안정하지만
       컴퓨터 계산은 불안정해질 수 있다 (수치적 안정성)

       (이걸 TDF-II 에서는 출력을 먼저 만들고 -> 그 다음에 상태(state)를 업데이트함)

    + Rearranged Direct form I
    + Reverses order : IIR first, then FIR 

    +structure 
        ( w[n] : intermediate state - 중간상태 ) -> 딜레이 하나로 합친 결과 (큰 중간상태가 저장되는 경우 많음)
        w[n] = x[n] - a1*x[n-1] - a2*w[n-2]
        y[n] = b0*w[n] + b1*w[n-1] + b2*x[n-2]

    +Pros and cons
        - Pros: Uses half the memory of DF-I (메모리 절약 중요)
        - Cons: Still prone to rounding errors, not much beetter than DF-I

    
    """

def understand_transposed_direct_form():
    """
    Transposed Direct Form II
    : best for audio (실무에서 가장 많이 쓰임)
    DF2 를 signal flow graph 로 뒤집으면 
    => 반올림 오차가 훨씬 작음
    
    Transpose = flip the block diagram (블록 다이어그램을 뒤집음)

    +Formula

        y[n] = b0*x[n] + s1[n-1]
        s1[n] = b1*x[n] - a1*y[n] + s2[n-2]
        s2[n] = b2*n[n] - a2*y[n]

        where s1, s2 are internal states

    => 수학적으로는 DF-II와 같지만, 구조가 다름

    +Pros and cons
        - Pros: standard in professional audio (ableton, logic .. 에서 선호함)
                , best numerical properties, minimal rounding error builup
                , least likely to overflow
        - Cons: slightly more complex to understand
                , not as obvious as direct form 

    +WHY IS TDF-II BETTER?
        - Intermediate values stay smaller (중간값이 작아서 오버플로우 방지)
        - Rounding errors don't accumulate (반올림 오차가 누적이 안됨)
        - MUltiplication order optimal (곱셈 순서가 최적)
    
    """

def understand_cascade_sos():
    """
    Cascade of second-order sections (SOS)
    The SAFEST way for high-order filters
    (높은 차수 필터의 가장 안전한 방법)
    : ex. butterworth 8차 는 Pole 이 8개 있기때문에 차분 방정식 생성시 매우 길어짐
    => 이걸 다 계산하면 오차가 매우 많이 생김 -> 그래서 이 8차를 2+2+2+2차로 나눔 
    (2차가 제일 안정적)
        ex. signal.butter(..., output="sos) 
            => 이게 cascade_sos 사용하는것 


    High-order filters (8th, 10th order, etc..) 
    Single direct form -> UNSTABLE (수치적으로 불안정)
    Rounding errors accumulate
    Output can explode or become garbage
    

    [The solution]
    Split into cascade of 2nd-order sections (2차 섹션으로 분해)

        ex. 8th-order filter
        -> split into 4 biquads (2nd-order filters)
        -> Chain them: input -> biquad1 -> biquad2 -> ... -> output
    
    +Formula
        H(z) = H1(z) * H2(z) * H3(z) * H4(z)
            each Hi is 2nd order     

        **극점을 네개로 쪼개는 moog ladder - 24dB/oct low pass filter 
            => 구조가 여기에 적용됨
            : 1차 네개 이어붙임

    +WHY is this safe?
        : each section is 2nd-order (작은 수치라서 오차적음)
        : Poles are locally grouped (극점이 묶여있음 = 안정성 높음)
        : Proven industrial standard (산업 표준 = 신뢰도 높음)

    +Pros and cons
        - Pros : Very stable, even for high order
                , No special numerical tricks needed
                , Professional standard
                , Easy to implement with scipy
        - Cons : Slight overhead (small)
                , slightly more code
    """

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compare_topologies_visually():
    """
    Compare the different topologies on same filter 
    """

    # Design a high-order filter
    fp, fs = 0.2, 0.25
    Ap, As = 0.5, 60

    b, a = signal.butter(6, (fp+fs)/2) # 6차 order
        # (fp+fs)/2 로 중간갑을 그냥 cutoff 로 잡는것 

    # get SOS form
    # sos = signal.butter(6, (fp+fs)/2), signal.tf2sos(b, a)
        # sos = cascade 2nd-order sections 
        # 6차 필터면 biquad 가 3개로 나눠짐 
        # signal.tf2sos(b, a) 
        #   : 이미 있는 b, a(전달함수 계수)를 biquad 직렬 (sos) 형식으로 바꿔주는 함수
    """
 sos = [[b0,b1,b2, a0,a1,a2],  ← 1번째 biquad (6개 숫자)
       [b0,b1,b2, a0,a1,a2],   ← 2번째 biquad
       [b0,b1,b2, a0,a1,a2]]   ← 3번째 biquad
        """
    sos = signal.butter(6, (fp+fs)/2, output='sos')
        # sos 는 위처럼 두개씩 나눈 biquad 필터의 각 계수를 알려줌 

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # test signal: impulse
    impulse = np.zeros(300)
    impulse[0] = 1

    # Method 1: Direct form 
    y_df = signal.lfilter(b, a, impulse)

    # Method 2: Cascade SOS
    y_sos = signal.sosfilt(sos, impulse)
        # signal.sosfilt(전달함수 계수, input 소스)
        # 의 출력은 output array

    # Plot 1: Impulse response = Direct form
    ax = axes[0, 0]
    ax.stem(y_df[:100], basefmt=' ')
    ax.set_ylabel('Amplitude')
    ax.set_title('Direct form I \n(can be unstable)')
    ax.grid(True, alpha=0.3)

    # Plot 2: Impulse response = SOS
    ax = axes[0, 1]
    ax.stem(y_sos[:100], basefmt=' ')
    ax.set_ylabel('Amplitude')
    ax.set_title('Cascade SOS\n(very stable)')
    ax.grid(True, alpha=0.3)

    # Plot 3: Frequency response comparison
    ax = axes[1, 0]
    w, h_df = signal.freqz(b, a, worN=500)
    w, h_sos = signal.sosfreqz(sos, worN=500)

    mag_df = 20 * np.log10(np.abs(h_df) + 1e-10)
    mag_sos = 20 * np.log10(np.abs(h_sos) + 1e-10)

    ax.plot(w/np.pi, mag_df, linewidth=2, label='Direct form', color='blue')
    ax.plot(w/np.pi, mag_sos, linewidth=2, linestyle='--', label='SOS', color='red', alpha=0.7)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Frequency Response\n(Should be identicals)')
        # [identical : 동일한]
    ax.legend()
    ax.grid(True, alpha=0.3, which = 'both')

    # Plot 4: Pole locations
    ax = axes[1, 1]
    ax.axis('off')

    pole_text = """
    TOPOLOGY COMPARISON
    ───────────────────
    
    Direct Form I:
    • Simple code
    • Can be unstable
    • Rounding errors accumulate
    • For low order only
    
    Transposed DF-II:
    • Better than DF-I
    • Good stability
    • Less memory
    • Professional choice
    
    Cascade SOS:
    • Best stability ✓
    • High order safe
    • Industry standard
    • Recommended!
    
    Recommendation:
    Use SOS for safety!
"""
    ax.text(0.05, 0.95, pole_text, fontsize=9, verticalalignment='top',
            family='monospace', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    

    plt.tight_layout()
    plt.show()
    


compare_topologies_visually()


"""Real time audio 에서는 

zi = signal.sosfilt_zi(sos) 
y_chunk, zi = signal.sosfilt(sos, x_chunk, zi=zi)
    # sosfilt(..., zi=zi)가 출력과 함께 갱신된 zi 를 반환하고,
    그걸 다음 chunk 호출에 다시 넣으면 필터가 연속 신호처럼 동작해서 경계 클릭 안생김
    => 실시간 오디오에서는 필수다.

'zi' carries state between calls 
(zi 가 호출들 사이에 상태를 나른다.)

**실시간에서는 오디오를 '조각'으로 처리함
=> 전체신호를 한번에 처리하는것 아님 (실시간으로 처리해야하기 때문에)
ex. 512 샘플씩 버퍼로 주고받음. 필터를 chunk 단위로 반복호출

=> 여기서 'zi' 는 초기상태=initial cconditions. 
: 필터의 내부상태(과거 값등)을 담아서 다음 chunk 로 넘겨주는 상자임

"""

"""Transpose Theorem
: 블록 다이어그램의 모든 화살표 방향을 뒤집고, 입력과 출력을 바꾸면 같은 전달함수를 얻는다.

즉, DF-II를 거꾸로 그리면 TDF-II 가 됨

"""