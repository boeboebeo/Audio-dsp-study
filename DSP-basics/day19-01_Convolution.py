"""
==============================================
DAY 19: Convolution & FFT-based Convolution
==============================================

MATHEMATICAL PREREQUISITES:
- Multiplication (곱하기): Basic math
- Summation (합하기): Adding many numbers
- Time-domain vs Frequency-domain (시간/주파수 영역)

KEY CONCEPT:
Convolution = most fundamental operation in DSP
    => 사실 그냥 지금까지 공부한 filter 가 사실은 Convolution 을 하는 기계임
    + DSP에서의 가장 기본적인 연산
It's how filters work (multiply-add operation)

Two ways to do convolution:
1. Direct time-domain: Simple but SLOW O(N²)
2. FFT frequency-domain: Complex but FAST O(N log N)

Choice depends on signal length!

    ex. FIR 의 

    b = [0.2, 0.6, 0.2]라면
    실제로 CPU 는 y[n] = 0.2x[n] + 0.6x[n-1] + 0.2x[n-2] : 이 차분방정식을 계산함
    
        => 이걸 convolution 이라고 부름!



"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft

SAMPLE_RATE = 44100

def understand_convolution_fundamentals():
    """what is convolution? (합성곱)

    y[n] = Σ x[k] * h[n-k]  (for k = 0 to N)

        - 주파수 별로 input * h (필터 impulse response) 곱한것을 다 합치는것

    h[n] = impulse response
    x[n] = input signal
    y[n] = output signal

        **Process
        1) Flip h[n] backwards -> h[-n]
        2) Slide it across x[n]
        3) Multiply overlappint parts
        4) sum the products
        5) Repeat for each position
        => Filtered signal y[n]  

    
    Every filter is a convolution!
    FIR: y[n] = b0*x[n] + b1*x[n-1] + ...
        (h[0] = b0, h[1] = b1.. 이렇게 둘은 완전히 같은 식)
    IIR: y[n] = Implemented via difference equation
        (IIR 은 출력도 들어가기에 겉보기에는 convolution 이 아니지만 IIR 에도 impulse를 넣으면
        출력이 h[0], h[1] .. 이렇게 나옴 -> 이게 IIR 의 Impulse response)

    
    ** y[n] = x[n] * h[n]
        : Impulse response 와 Convolution 한 결과
        + 근데 IIR 은 피드백때문에 무한한 impulse response 를 가지고 있기 때문에
          convolution 으로 계산하면 무한한 h를 계산해야하므로 불가능함
          => 그래서 그걸 차분방정식으로 효율적으로 구한것 
          **수학적으로 -> convoluton -> 컴퓨터 구현 -> difference equation
    
                    모든 LTI 필터
                      │
        ┌─────────────┴─────────────┐
        │                           │
      FIR                         IIR
        │                           │
Impulse Response              Impulse Response
유한함                         무한함
        │                           │
Convolution                  Convolution
        │                           │
직접 계산 가능              직접 계산하면 너무 김
        │                           │
Difference Equation      Difference Equation
(Convolution과 동일)     (Convolution을 효율적으로 구현)
    

**근데 IIR 은 무한한 Convolution 을 어떻게 계산할까
: 수학적으로는 출력을 계속 재사용하는것 처럼 보이지만, 전개해보면
-> 과거입력 -> impulse response 계수 -> 모두 더하기
: 즉, 무한한 FIR 와 완전 같은 형태가 된다.

예를 들어

    y[n] = x[n] + 0.5y[n-1] 이라는 식에서 

    y[n-1]를 펼쳐보면 x[n] + 0.5(x[n-1] + 0.5y[n-2]) 이 됨
    + 전개하면 : = x[n] + 0.5x[n-1] + 0.25y[n-2]

    y[n-2]를 펼치면 x[n] + 0.5x[n-1] + 0.25(x[n-2] + 0.5y[n-3]).. 이렇게 펼치다보면

    아까 convolution 식이랑 같아짐 
    
    y[n] = 1x[n] + 0.5x[n-1] + 0.25x[n-2]+⋯

    => 즉, impulse response 는 1, 0.5, 0.25, ... (이것과의 convolution 이 된것)

즉, IIR 은 convolution 과 다른 방식의 필터가 아니라, convolution 을 매우 효울적으로 구현한 방법임

    
    """

def demonstrate_direct_convolution():
    # step by step

    x = np.array([1, 2, 3]) #input signal (3 samples)
    h = np.array([0.5, 0.3]) # filter (2 samples). impulse response h[n]

    """Direct convolution by hand
    
    y[n] = x[n]*h[0] + x[n-1]*h[1]

        + Manual calculation

        n=0:
        y[0] = x[0]*h[0] + x[-1]*h[1]
             = 1*0.5 + 0*0.3 = 0.5

        n=1:
        y[1] = x[1]*h[0] + x[0]*h[1]
             = 2*0.5 + 1*0.3 = 1.3

        n=2:
        y[2] = x[2]*h[0] + x[1]*h[1]
             = 3*0.5 + 2*0.3 = 2.1

        n=3:
        y[3] = x[3]*h[0] + x[2]*h[1]
             = 0 *0.5 + 3*0.3 = 0.9

        y = 0.5 1.3 2.1 0.9 
    """

    y_manual = np.array([0.5, 1.3, 2.1, 0.9])

    # using numpy   
    y_numpy = np.convolve(x, h, mode='full')
    print(f"Numpy result: y[n] = {y_numpy}")
    print(f"Match? {np.allclose(y_manual, y_numpy)}")
        # 다 match 되는지 확인하는 함수

    """Observations

    - input : 3 samples
    - filter : 2 samples
    - output : 4 samples (3 + 2 - 1 )
        => output grows in size! 

    """

def visualize_convolution_process():
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # create signals
    x = np.array([1, 2, 3, 1, 0.5]) # Input
    h = np.array([0.5, 0.3, 0.1]) # Filter (impuluse response)

    # compute convolution
    y = np.convolve(x, h, mode='full') 

    # plot 1: Input signal
    ax = axes[0]
    ax.stem(range(len(x)), x, basefmt=' ')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Input signal x[n]\n({len(x)} samples)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3.5)

    # plot 2: filter (impulse response)
    ax = axes[1]
    ax.stem(range(len(h)), h, basefmt=' ')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Filter Impulse response h[n]\n({len(h)} samples)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.6)

    # plot 3: Output (convolve)
    ax = axes[2]
    ax.stem(range(len(y)), y, basefmt=' ')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Output signal y[n] = x[n] * h[n]\n({len(y)} samples = {len(x)} + {len(h)} - 1)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)

    plt.tight_layout()
    plt.show()


visualize_convolution_process()

# demonstrate_direct_convolution()





