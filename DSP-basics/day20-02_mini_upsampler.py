"""[MINI UPSAMPLER]
: 원래 신호
-> Zero insertion
-> Image 가 생성되어 버림
-> LPF (원래의 sample rate의 절반인 Nyquist를 cutoff 로 설정
-> 이미지 제거 + 샘플 사이가 자연스럽게 보간 됨 (interpolation)


[SUMMARY]
**우선 Oversampling 을 먼저하여서 nyquist 를 올리고, 모듈레이션이나 다양한 이펙팅으로 생기는
원래의 샘플레이트의 nyquist 이상의 주파수가 오버샘플링 한 샘플레이트의 nyquist 이하로 들어오게 된다.
그 후, 다시 downsampling 을 해야하는데, 
그 필터링 전에! LPF 를 걸어서 그 old_nyquist 이상으로 생긴 모든 주파수를 없앤다.

44.1
↓
Upsampling : 이미 alias 가 생긴이후에는 LPF 가 없앨 수 없으니, 
             일단 샘플레이트를 높여서 nyquist의 주파수가 해당 주파수에 생기게 한다(alias x)
↓
176.4
↓
Distortion
↓
고조파 많이 생김 (original_nyquist 이상의 고조파 많이 생김)
↓
LPF
(88.2kHz보다 위 제거) -> 이미지를 제거하는 interpolation filter
↓
Downsampling
↓
44.1

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from scipy.fft import fft, ifft #=> 이거 거의 안쓰고 있어서 삭제가능




def upsampling_visualization():
    # 1) 간단한 신호 만들기
    fs = 48000
    duration = 0.1 #s
    freq = 100

    upsampling = 2

    t = np.linspace(0, duration, int(fs*duration), endpoint=False) 
        # endpoint=False 를 넣어주지 않으면, 0초 / 0.1 초 둘다 들어가게 됨
        # => 샘플간격이 아주 조금 틀어진다.
    t_up = np.linspace(0, duration, int(fs*duration*upsampling))
    x = np.sin(2*np.pi*freq*t)



    # 2) 새 배열 만들기 (zero insertion) 원래길이의 두배의!
    upsampled = np.zeros(len(t)*upsampling)

    # 그리고 그 upsampled 배열에 0, 2, 4, 6 번째에만 원래 값을 넣음
    upsampled[::upsampling] = x
        # print(upsampled) : 한칸 건너 뛰고 0 이 다 들어가 있음 


    # 3) FFT 보기
    X = np.fft.rfft(x)
        # 그럼 크기는 np.abs(X)
        # 주파수 축은 = np.fft.rfftfreq(...)
    fft_freq = np.fft.rfftfreq(len(x), d=1/fs)
        # 주파수 축 구할때는 len(x), d=1/fs - 샘플 간격
        # FFT 는 원래 0, 1, 2..번째 bin 을 구해줘야함 => 그래서 d = 1 /fs 넣는것
    magnitude_db = 20*np.log10(np.abs(X)+ 1e-12)

    # 4) upsampling 한 것의 FFT 
    X_up = np.fft.rfft(upsampled)
    fft_freq_up = np.fft.rfftfreq(len(upsampled), d=1/(fs*upsampling))
        # 저기 d 부분의 괄호가 잘못되어 1/fs*upsampling 으로 처리하게 된다면
        # python에서는 (1/fs) * upsampling 으로 계산해버림! 
        # => 1 / (fs*upsampling) 으로 수정해줘야함 
    magnitude_db_up = 20*np.log10(np.abs(X_up) + 1e-12)




    fig, axes = plt.subplots(2, 3, figsize=(14, 8))


    # plot1: 원래 sample rate 신호 (plot)
    ax = axes[0, 0]
    ax.plot(t, x, linewidth=1.5) #여기는 이제 ax. __ 이렇게 함수 넣어야 함
    ax.set_title(f'Sample rate: {fs}')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, alpha=0.3)


    # plot2: 원래 sample rate 신호 (stem) 
    # -> 근데 스템으로 찍으니까 뭔가 이상함. 걍 점을 찍어야 할것 같은데 점 어케 찍지? 
    # : ax.scatter( ) ! 로 점찍기
    ax = axes[1, 0]
    ax.scatter(t, x, s=0.5)
        # 점으로 찍는법. 점 크기 작게 하려면 s = 15 이렇게 크기 지정하기
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, alpha=0.3)

    # plot3: upsampling 한 신호
    ax = axes[0, 1]
    ax.plot(t_up, upsampled, linewidth=1.5)
        # x, y 가 같은 첫번째 dimension 을 가지고 있어야 그 (x, y) 딱 그 지점에 좌표를 찍는데
        # 지금은 x 보다 y 가 1/2 임. y 가 업샘플링한 배열이니까
        # 그렇다고! 여기에 t*2를 하면 그냥 [0, 1, 2] 배열이 -> [0, 2, 4] 가 되는것
        # 그게 아니고, 이제 시간이 촘촘해져야 함 
        # [0.00, 0.01, 0.02, 0.03] 이게 -> [0.000, 0.005, 0.010, 0.015] 이렇게!
        # 그래서 t_up 을 새로 만들어야 함 (시간간격을 좀 더 촘촘하게 만든!)
    ax.set_title(f'Sample rate: {fs*2}')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, alpha=0.3)

    # plot4: upsampling 한 신호 scatter
    ax = axes[1, 1]
    ax.scatter(t_up, upsampled, s=0.5)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, alpha=0.3)

    #plot 5: fft 한 그래프
    ax = axes[0, 2]
    ax.plot(fft_freq, magnitude_db)
        # 0Hz 는 정의 되지 않음 (log(0))

    ax.set_xscale('log')
    # ax.set_xlim(0, fs)
        # 근데 set_xscale 을 log 로 셋팅해두면, xlim 은 무시되는듯
    ax.set_ylim(-5, 100)
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('Magnitude(dB)')
    ax.set_title('FFT: old Sample rate')
    ax.grid(True, alpha=0.3)

    #plot 6: upsampled 신호를 fft 한 그래프
    ax = axes[1, 2]
    ax.plot(fft_freq_up, magnitude_db_up)
    ax.set_xscale('log')
    # ax.set_xlim(0, fs)
    ax.set_ylim(-5, 100)
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('Magnitude(dB)')
    ax.set_title(f'FFT: x{upsampling} Sample rate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()







def upsampling_applying_antialiasing_filter():
    # 1) 간단한 신호 만들기
    fs = 48000
    freq = 8000
    duration = 0.1

    upsampling = 2

    t = np.linspace(0, duration, int(fs*duration), endpoint=False) #사실 여기서는 꼭 sample rate 만큼으로 1까지를 나눌 필요 없음
    t_up = np.linspace(0, duration, int(fs*duration*upsampling), endpoint=False)
    x = np.sin(2*np.pi*freq*t)



    # 2) 새 배열 만들기 (zero insertion) 원래길이의 두배의!
    upsampled = np.zeros(len(t)*upsampling)

    # 그리고 그 upsampled 배열에 0, 2, 4, 6 번째에만 원래 값을 넣음
    upsampled[::upsampling] = x
        # print(upsampled) : 한칸 건너 뛰고 0 이 다 들어가 있음 


    # 3) FFT 보기
    X = np.fft.rfft(x)
        # 그럼 크기는 np.abs(X)
        # 주파수 축은 = np.fft.rfftfreq(...)
    fft_freq = np.fft.rfftfreq(len(x), d=1/fs)
        # 주파수 축 구할때는 len(x), d=1/fs - 샘플 간격
        # FFT 는 원래 0, 1, 2..번째 bin 을 구해줘야함 => 그래서 d = 1 /fs 넣는것
    magnitude_db = 20*np.log10(np.abs(X)+ 1e-12)
        # 1e-12 는 1000000000000분의 1
        # log(0) = - 무한대 이기때문에 저게 없으면 0인 값의 경우 에러가 남
        # epsilon : 여기서의 e는 exponent(지수)를 표현하는 과학적 표기법

    """20log(1e-12) 는? 

    log(10^x) = x 이기 때문에 

    log10(100) = 2
    log10(1000) = log10(10^3) = 3 => 로그는 지수를 꺼내오는 함수이기 때문에 

    log(10^(-12)) = -12 

    20 * -12 = -240dB 가 나옴
        
        """


    # 4) upsampling 한 것의 FFT 
    X_up = np.fft.rfft(upsampled)
    fft_freq_up = np.fft.rfftfreq(len(upsampled), d=1/(fs*upsampling))
    magnitude_db_up = 20*np.log10(np.abs(X_up) + 1e-12)

    # 5) FIR 필터 만들어보기 
    fir_filter = signal.firwin(401, fs/2, fs=fs*2)
        # fir filter 계수(b) 만들기, fir 이니까 분모는 그냥 [1]
        # cut off 는 upsampling 한 sample rate 의 절반
        # fs = 44100 을 쓴다면, 실제 Hz 값이 그 앞에 들어가게 되는것
        # 그게 없으면 그냥 nyquist 가 1로 정규화된 수치
    up_filtered = signal.lfilter(fir_filter, [1], upsampled)
        #signal.freqz : 그 필터의 freq response 를 보여줌
            # 그냥 단순 필터에서의 각 주파수의 응답을 보여줌
        #signal.lfilter : 신호를 실제로 필터링함. 입력신호 필요
        #signal.lfilter(b, a, x) 에서의 x 는 입력신호(시간영역의 입력신호!) => FFT 한 신호 넣지 않음
        #lfilter() 는 시간영역에서 동작하기 때문

    # 6) FIT filtering 한것을 FFT해서 frequency domain 으로 확인
    fft_up_filtered = np.fft.rfft(up_filtered)
    fftfreq_up_filtered = np.fft.rfftfreq(len(up_filtered), d=1/(fs*upsampling))
        # upsampling 한 그 sample rate 의 간격으로 x축 설정
    up_filtered_db = 20*np.log10(abs(fft_up_filtered) + 1e-12)
    

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # plot 1: Before Filter
    # 왜 여기는 그래프 내부가 채워지고, 밑에는 안채워지지...?
    # : 이게 사사ㅣㄹ 그 내부를 채운것이 아니고, 1, 0, 2, 0, 3, 0 ... 이런식으로
    # 0하고 매우 빠르게 왔다갔다 하니까 우리눈에는 그 내부의 색상을 채운것 처럼 보이는것!
    ax = axes[0, 0]
    ax.plot(t_up, upsampled, linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.set_title('Before Filtering_Time')
    ax.set_xlim(0, 0.002)
        # 위의 질문처럼 내부가 다 채워진것 처럼 보였기 때문에
        # 이렇게 x축을 매우 좁게해서 늘려봤더니 1 0 2 0 .. 이렇게 0점으로 왔다갔다 하는것이 보임
    ax.grid(True, alpha=0.3)

    # plot 2: Before filtering FFT
    ax = axes[0, 1]
    ax.plot(fft_freq_up, magnitude_db_up, linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.set_title('Before filtering_FFT')
    ax.set_xlim(0, fs*upsampling)
    ax.set_ylim(-10, 100)
    ax.grid(True, alpha=0.3)

    # plot 3: FIR direct (time domain)
        # 여기에서의 그래프상 (101-1) / 2 = 50 samples 만큼의 딜레이 때문에 파형이 살짝 밀려있음
    ax = axes[1, 0]
    ax.plot(t_up, up_filtered, linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    # plot 4: FFT_Filtered
    ax = axes[1, 1]
    ax.plot(fftfreq_up_filtered, up_filtered_db, linewidth=1.5)
    ax.set_ylabel('Amplitude')
    ax.set_title('upsampling_FFT')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 100)
    ax.set_xlim(0, fs*upsampling)
    # ax.set_xscale('log')

    plt.tight_layout()
    plt.show()

    w, H = signal.freqz(fir_filter, [1], worN=8192, fs=fs*upsampling)
        # w 는 Hz 가 아니라 rad/sample 로 반환함
        # 0 ~ 3.14 까지!
        # 그래서 파라미터에 fs=fs*upsampling 넣어주면 알아서 freq 로 출력
    frequency_res_db = 20*np.log10(np.abs(H) + 1e-12)

    plt.figure(figsize=(8, 4))
        # 새로운 도화지 만듦
    plt.plot(w, frequency_res_db)
    plt.title('Filter:\nFrequency response')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Freuqency(Hz)')
    plt.ylabel('Magnitude(dB)')
    
    plt.show()
        # 지금 까지 만든 figure 를 출력하는 것이라서 figure 마다 plt.show() 넣어줘야 함


upsampling_visualization()
upsampling_applying_antialiasing_filter()



# Convolution
# FFT → X×H → IFFT



"""

**260719 - 코드상의 개선점
1) 우선 위의 코드에서 t, x, upsampled, fft, plot 등은 거의 대부분은 
아래 함수에서도 복붙이기 때문에 

    def generate_sine(..):
        return t, x <- 이렇게 return 하게 해서 이 값을 재사용해도 좋을것 같다.

    그리고 다른 함수 내부에서 쓸때는 

        t, x = generate_signal(
        fs= 48000,
        duration = 0.1,
        freq=100
        ) 
        => 이렇게 가져와서 쓸 수 있음

2) 한 함수내에서는 한가지 일만 하게 해서 재사용성을 키워보자!


"""

