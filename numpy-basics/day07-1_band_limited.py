"""
==============================================
DAY 7: Band-Limited Synthesis (BLIT & PolyBLEP)
==============================================
Goal: 실시간으로 사용 가능한 band-limited waveform 생성 기법을 학습한다.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 1.0

def blit_impulse_train(freq, duration, sample_rate):
    """
    BLIT : Band-limited Impulse Train (대역 제한 임펄스 열)
    """