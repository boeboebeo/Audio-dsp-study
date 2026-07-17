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

        **Process (기하학적 정의)
        1) Flip h[n] backwards -> h[-n]
        2) Slide it across x[n]
        3) Multiply overlappint parts
        4) sum the products
        5) Repeat for each position
        => Filtered signal y[n]  

            1) 예를 들어 h [ 1, 2 ] 라면 그걸 [2, 1] 로 뒤집음
            2) x: 1 2 3 4 
               h: 2 1
               => 여기 x=1, 2 에 h가 겹치니까 겹친 부분 곱하기 1*2 + 2*1

               x: 1 2 3 4 
                 h: 2 1
                => 이때는 2*2 + 3*1 

                3)이렇게 h 를 옆으로 옮겨가며 곱하고 다 더함

            **h를 뒤집는 이유:
            이게 차분방정식에서는 애초에 x[n]. x[n-1]. x[n-2] 이렇게 거꾸로 참조되고 있기 때문.

            y[n] = sum(h[k] * x[n-k]) 의 x[n-k] 부분때문에 convolution에서는 뒤집음

            -> 이걸 프로그래밍으로 쓰면(차분방정식에서는)

                y[n] = b0 * x[n] + b1 * x[n-1] + ...

        
    
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

def understand_computational_cost():
    # 합성곱 속도가 중요한 이유 & 계산량 차이
    """
    1) Direct convolution complexity

    For signal length N and filter length M:
        Operations = N * M multiplications + additions
        Complexity = O (N * M)

        example: 1-second audio at 44.1kHz, 1000-tap filter
        N = 44100 samples
        M = 1000 taps
        Operations = 44100 * 1000 = 44,100,000 (4천만번의 곱하기)

    Time @ 1GHz CPU: ~44ms (실시간 처리는 43ms 안에 처리해야함)
    => Borderline! : Might not fit in real-time


    2) FFT-Based convolution cost

    FFT of signal : N log N
    FFT of filter : M log M
    Multiply : N
    IFFT : N log N
    => Total : O (N log N)

        example: same 1-second audio, 1000-tap filter

        N = 44100 samples
        Operations = 44100 * log2(44100) * 3
                   = 44100 * 16 * 3 = 2,116,800 (약 210만번의 연산)

    Time @ 1GHz CPU: ~2ms (44배 빠름)

    **
    [Speedup comparison]

    print("Filter length    | Direct ops   | FFT ops    | Speedup")
    print("─────────────────┼──────────────┼────────────┼─────────")
    print("100 taps         | 4.4M         | 710K       | 6x"      )
    print("1000 taps        | 44M          | 2.1M       | 21x"     )
    print("5000 taps        | 220M         | 13M        | 17x"     )
    print("─────────────────┴──────────────┴────────────┴─────────")
    
    
    """

def explain_fft_convoluation():
    #How FFT based convolution works. why is it faster?
    """
    [THEORY] FFT-Based Convolution

    "Convolution Theorem (합성곱 정리)"

    Time domain: y[n] = x[n] * h[n]
                 output = input * filter (impulse response)
    
    Frequency domain: Y(ω) = X(ω) x H(ω)
                            => MULTIPLICATION (not convolution)

    => Meaning : Convolution in time = Multiplication in frequency
    (time domain에서의 합성곱은 = freq domain 에서의 곱셈이다.)
    시간에서 합성곱 = 주파수에서 곱하기

    
    [STRATEGY]
    1. Convert to frequency domain (FFT)
        x[n] -> X(ω)
        h[n] -> H(ω) 필터도 FFT 해서 H(w)로 바꿈
    2. Multiply (fast!)
        Y(ω) = X(ω) * H(ω) <- just multiply
    3. Convert back to time domain (IFFT)
        Y(ω) -> y[n] 

    => Result
    : Same output as direct convolution
    but, O(N log N) instead of O(N^2)

    why fast?
    FFT: Clever algorithm reduces N^2 to N log N
        - Direct multiply: N^2 multiplications
        - FFT multiply: N multiplications (way less)
            **주파수 영역에서의 Bin 끼리의 곱셈은 N 번, FFT , IFFT 또한 자체도 계산량이 있음
        - The FFT itself is clever too (divide & conquer)

    """

def demonstrate_fft_convolution():
    # Actually use FFT convolution

    #[IMPLEMENTATION] FFT convolution in Practice
    # Signal
    x = np.random.randn(1000) #1000sample input 
        # 평균 0, 표준편차 1인, Gaussian Noise 1000개 생성
        # 랜덤한 1000개의 값 있는 배열 출력
    h = signal.firwin(101, 0.3) #101 taps (Impulse response, FIR 계수)

    # Method 1: Direct convolution
    y_direct = np.convolve(x, h, mode='full')

    # Method 2: FFT convolution
    N = len(x) + len(h) - 1  # Linear convolution 을 얻기 위한FFT size.
        #입력길이 + 필터 길이 - 1
    X = fft(x, N) # Zero-padded FFT of input    
        # 원래 x 의 길이는 1000인데, FFT 는 1100으로 하니까 뒤에 부족한 100개는 0으로 자동으로 붙임
        # = Zero padding
    H = fft(h, N) # Zero-padded FFT of filter
    Y = X * H   # multiply in frequency
    y_fft = np.real(ifft(Y))[:len(y_direct)] #Back to time domain
        # convolution 길이는 입력+필터-1 이기때문에 입력의 길이보다 늘어남. 그래서 거기까지만 씀
        # np.real : IFFT 의 결과는 복소수이다.
        # (컴퓨터는 부동소수점 오차가 있기떄문에)
        # 원래는 허수부가 0이여야 하지만 컴퓨터에서는 오차가 생김 ex. 3.5+1.2e-15j
        # 그래서 np.real() : 허수부부분을 버린다. => 3.5만 가지고 옴

    # Verify they match
    error = np.max(np.abs(y_direct - y_fft))
        # 오차
        # 오차가 왜 생기지?
        # : 수학에서 cos pi = -1 이지만, 수학에서는 -0.999999999999처럼 나온다.

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # plot 1: Input 
    ax = axes[0]
    ax.plot(x[:200], linewidth=1, color='blue')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Input Signal\n({len(x)} samples)')
    ax.grid(True, alpha=0.3)

    # plot2: Filter
    ax = axes[1]
    ax.plot(h, linewidth=1.5, color='green')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Filter\n({len(x)} samples)')
    ax.grid(True, alpha =0.3)

    # plot3: Output comparison
    ax = axes[2]
    ax.plot(y_direct[:300], linewidth=1.5, label='Direct', color='red')
    ax.plot(y_fft[:300], linewidth=1.5, linestyle='--', label='FFT', color='blue')
    ax.set_ylabel('Amplitude')
    ax.set_title('Output: Direct vs FFT\n(should match)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def understand_overlap_add():
    # Overlap-add method for long signals
    # How to process very long audio files
    """[ADVANCED] Overlap-add method

    [Problem] 
    Direct FFT convolution requires the entire signal
    to perform one large FFT.
    (신호 전체를 한번에 FFT 하려면 그 전체 분량을 메모리에 올려야 함 -> 비효율적)
    But audio files can be Hours long
    Can't fit in RAM (메모리에 다 모든 신호를 적재해두어야 하는데 오디오 신호가 너무 김)

    [Solution]
    Process in chunks (청크단위로 처리)
        : 알고리즘이 한번에 처리하는 데이터 단위
        : 한번 FFT 를 시행하는 단위가 chunk 크기 만큼이기때문에 (블록 단위로 나눠서)
        전체 파일을 RAM 에 올려둘 필요가 없음
        (애초에 실시간 오디오에서는 파일 전체를 알수도 없음. 마이크 입력을 받고있다면!)

    1. Break signal into chunks (overlap-add)
        - chunk 1: samples 0 - 1023
        - chunk 2: samples 512 - 1535 (overlaps!)
        - chunk 3: samples 1024 - 2047
            ...
    2. FFT-convolve each chunk
        - chunk 1 output : 0-1123 ( 1024 + filter_length -1 )
        - chunk 2 output : 512-1635 (얘도 output 길어짐)
            ...
    3. Add overlapping parts (합 계산)
        - Overlapping region: Add the results together
        - Non-overlapping: Use directly
    4. Stream output
        - Send processed chunks to speaker / file
    => Result: Process infinite-length signals (무한 길이의 신호도 처리 가능)
    
        
    **CPU 연산량 = 약 (샘플레이트) * (채널 수) * (샘플 당 연산수)
        
        => 여기서 샘플당 연산수가 FIR FIlter 의 tap이 101 개 라면 
        하나의 샘플당 101번의 곱셈 + 100번의 덧셈이 필요함 . 약 한 샘플에 200번의 연산이 필요하다.


    **현재 계산할 샘플이 뭔지를 그걸 RAM 에 올려놓고 CPU 처리를 기다려야 하기 때문에
    효율적으로 원형버퍼라는걸 사용함 ( RAM 안에 존재 )
    """







# demonstrate_fft_convolution()

# visualize_convolution_process()



# demonstrate_direct_convolution()





