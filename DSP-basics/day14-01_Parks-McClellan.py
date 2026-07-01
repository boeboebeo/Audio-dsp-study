"""
==============================================
DAY 14: Parks-McClellan Algorithm (Optimal FIR)
==============================================

MATHEMATICAL PREREQUISITES:
    - Optimization (최적화): Finding best solution among many
    - Chebyshev Polynomials (다항식): Special math functions
    - Iterative algorithms (반복 알고리즘): Refine solution step by step

KEY CONCEPT:
Window method makes "good" filters but not OPTIMAL.
Parks-McClellan makes OPTIMAL filters for given specifications.
(PM 은 주어진 조건에서의 최적의 필터)

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_optimal_filter_design():
    """
    what does 'optimal' mean?
    => why would we want optimal (최적이란?)

    **Parks-McClellan Algorithm (Optimal FIR)

    1) Window
        - You can't control ripple amount precisely
        - You can't specify exact specifications
        - Might use more taps than necessary
        - Wastes CPU

    2) Parks-McClellan solves this :
        - Meet EXACT specifications
        - Use minimum number of taps
        - get BEST possible response
        - Most efficient implementation

        => Best possible route given constraints[제약]
        (Constraints: Allowed ripple amount, transition width
         Objective[목적]: minimize filter length)

    [analogy : 유사, 유추] like finding best route

    **why call it 'REMEZ'
        - algorithm based on 'Remez Exchange Theorem
        - Mathematical principle by Russian mathematician Remez
        - Shows that optimal filter has EQUIRIPPLE response
        => 모든 영역에서 리플이 동일하다 

    """

def understand_equiripple():
    """
    what is equiripple and why is it optimal?
    (등리플)

    - window method 의 response:
        + 부드럽지만 불균형
    
    - Parks-McClellan response:
        + Passband: RIPPLES equally (±ε)
        + Transition: Sharp boundary
        + Stopband: RIPPLES equally (±δ)
        => 모두 동일한 크기로 울림
        => 통과 대역이득이 정확히 1 이 아니라  1 ± ε(엡실론=>passband ripple 의 크기)
        => 차단 대역이 정확히 0이 아니라 ± δ 사이에서 출렁임= (델타 = stopband ripple 의 크기)
        : 모든 리플이 똑같은 높이로 균등함 
    
    **why is this 'optimal'?
        - Think of error budget : 100dB

        Window method
            + Passband : 2 dB ripple (여기서 여유 낭비)
            + Stopband : 98 dB ripple

        Equiripple
            + Passband : 50 dB ripple
            + stopband : 50 dB ripple 
            => 모든 영역에서 균형잡힌 사용 
            => 더 적은 탭으로 더 나은 사양 만들 수 있음 
            (Better specs with fewer taps)

    """

def compare_window_vs_parks_mcclellan():
    """Show concrete comparison: same cutoff, see which is better"""

    # design with window
    normalized_cutoff = 5000 / (44100/2)
    fir_window = signal.firwin(101, normalized_cutoff, window='hamming')
        #101 = tap number

    # design with maclellan
    # scipy.signal.remez degigns optimal FIR 
    # parameter : order, bands, desired values, weights

    numtaps = 51 # will try 51 taps
    bands = [0, 0.1134, 0.1247, 0.5]
        # pass band "0-0.22", 이 사이게 transition band, stop band "0.25-0.5"
        # 위 밴드 값은 내가 계산해서 넣거나 따로 계산해주는 함수 있음
        # 주파수 대역을 쌍으로 나눠서 정의 함 (앞, 뒤 각각 첫번째 ,두번째 밴드)
        # signal.remez 함수 사용할거면은 마지막 값이 1 이 아니고 0.5 로 끝나야 함 
        # remez 함수는 주파수 스케일이 0-1 이 아니라, 0-0.5(nyquist)로 사용함
        
    desired = [1, 0] # gain 1 in passband / gain 0 in stop

    fir_optimal = signal.remez(numtaps, bands, desired, weight=[1, 1])
    # remez exchange algorithm 을 사용해서 제일 최적화된 계수를 계산해주는 함수
    # 더 적은 탭으로 같은 성능을 만들 수 있음
    # weight[1, 10] : Stopband ripple 을 10배 더 신경써서 줄여줘라
    # => 대신 passband ripple 이 좀 더 커짐 => equiripple 의 균형을 옮기는 것 

    # calculated to meet exact specifications 
    # => 정확한 사양을 만족하도록 계산됨

    # Comparison
    # window-101 taps , optimal- 51 taps

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # get frequency responses
    w = np.linspace(0, 1, 100)
    w_rad = w * np.pi

    _, h_window = signal.freqz(fir_window, [1], w_rad)
    _, h_optimal = signal.freqz(fir_optimal, [1], w_rad)

    mag_window = 20*np.log10(abs(h_window) + 1e-10) 
    mag_optimal = 20*np.log10(abs(h_optimal) + 1e-10)

    freq_hz = w * 22050 #convert to Hz

    #  plot 1: Full magnitude response
    ax = axes[0, 0]
    ax.plot(freq_hz, mag_window, linewidth=2, label='Window (Hamming)', color='blue')
    ax.plot(freq_hz, mag_optimal, linewidth=2, label='Parks-mcClellan', color='red')
    ax.axhline(-60, color='gray', linestyle='--', alpha=0.5, label='60dB target')
    ax.axvline(5000, color='gray', linestyle='--', alpha=0.5)
        # cut off 
    ax.set_xlim(0, 15000)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequecy (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Full response Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # plot 2: Passband zoom
    ax = axes[0, 1]
    passband_idx = freq_hz < 6000
    ax.plot(freq_hz[passband_idx], mag_window[passband_idx], linewidth=2.5,
            label='window', color='blue')
    ax.plot(freq_hz[passband_idx], mag_optimal[passband_idx], linewidth=2.5,
            label='optimal', color='red')
    ax.axvline(5000, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('Magnitude(dB)')
    ax.set_ylim(-5, 5)
    ax.set_title('Passband: Notice equal ripple in PM')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # plot 3: stopband zoom
    ax = axes[1, 0]
    stopband_idx = freq_hz > 5000
    ax.plot(freq_hz[stopband_idx], mag_window[stopband_idx], linewidth=2.5,
            label='window', color='blue')
    ax.plot(freq_hz[stopband_idx], mag_optimal[stopband_idx], linewidth=2.5,
            label='optimal', color='red')
    ax.set_xlim(5000, 15000)
    ax.set_ylim(-100, 5)
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('Magnitude(dB)')
    ax.set_title('Stopband: PM is sharper')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # plot 4: summary comparison
    ax = axes[1, 1]
    ax.axis('off')

    comparison_text = """
    WINDOW vs PARKS-McCLELLAN
    ─────────────────────────
    
    Window Method:
    • Simpler to use
    • Less control (덜 정밀함)
    • More taps needed
    • Smooth but wasteful
    
    Parks-McClellan:
    • Optimal for specs (최적)
    • Precise control
    • Fewer taps
    • Equiripple (균형잡힘)
    
    When to use which:
    • Quick design? Window
    • Professional? Parks-McClellan
    • Space/CPU critical? Parks-McClellan (Positive)
    • Educational? Window (simpler)
    
    Result:
    PM gets same performance
    with FEWER taps!
"""
    ax.text(0.05, 0.95, comparison_text, fontsize=9, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round',facecolor='lightyellow', alpha=0.5))
    
    # PM : 내가 원하는 스펙(pass, transition, stop band ripple)을 말해주면, 
    #   그걸 만족하는 최적의 계수를 수학적으로 찾아줌
    # 탭 수 = 계산량 = 메모리 인데, FIR 필터는 매 샘플마다 계수 개수만큼 곱셈-덧셈을 함
    # PM 더 적은 탭수로 같은 사양을 만들어 낼 수 있으니 CPU 덜 씀
    # CPU critical, space critical(메모리 덜 씀) 한 상황에서 유리함
    # 임베디드, 실시간 오디오에서는 PM 이 더 유리하다

    """
    - firwin : window
    - remez : parks-mcclellan
    - firls : least-squares
    - frequency sampling : firwin2 

    => 각각의 목적에 따라서 FIR 설계법이 달라지고, 사용되는 함수또한 달라짐
    (근데 탭을 하나라도 아껴야 할 경우에는 무조건 PM(remez) 사용)
    => 오디오 에서는 애초에 FIR 보다는 IIR 을 더 많이 씀
    
    """



    plt.tight_layout()
    plt.show()
        

compare_window_vs_parks_mcclellan()