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

    **why audio engineers care
    1) Linear phase = ALL frequencies delayed equally
        - Preserves waveform shape
        - Punch and clarity maintained
        (FIR)

    2) Minimum phase = Different delays per frequency
        - waveform changes slightly
        - But no pre-ringing artifacts
        - Natural sounding
        (IIR)

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