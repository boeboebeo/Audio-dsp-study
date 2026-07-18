import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft


def upsampling_visualization():
    # 1) 간단한 신호 만들기
    fs = 48000
    duration = 0.1 #s
    freq = 100

    upsampling = 2

    t = np.linspace(0, duration, int(fs*duration)) #사실 여기서는 꼭 sample rate 만큼으로 1까지를 나눌 필요 없음
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
    fft_freq_up = np.fft.rfftfreq(len(upsampled), d=1/fs*upsampling)
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


upsampling_visualization()


def upsampling_applying_antialiasing_filter():
    # 1) 간단한 신호 만들기
    fs = 48000
    freq = 100

    upsampling = 2

    t = np.linspace(0, 0.1, 1000) #사실 여기서는 꼭 sample rate 만큼으로 1까지를 나눌 필요 없음
    t_up = np.linspace(0, 0.1, 1000*upsampling)
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
    fft_freq_up = np.fft.rfftfreq(len(upsampled), d=1/fs*upsampling)
    magnitude_db_up = 20*np.log10(np.abs(X_up) + 1e-12)

    # 5) FIR 필터 만들어보기 

    h = signal.firwin(101, fs/2)
    
