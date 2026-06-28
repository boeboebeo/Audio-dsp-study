""" mini coding assignment

DC blocker : DC 를 빼야 적분기가 안 드리프트 함 (BLIT sawtooth)

**assignment : DC (0Hz)는 완전히 죽이고 나머지는 거의 그대로 통과시키는 1차 필터를 설계하라

1. DC(w = 0) 를 죽이려면 영점을 z=1 에 두어야 함 (1-z^(-1))의 분자또한 필요
2. 그것만 쓰면 통과대역이 기울어지니까, 극점을 영점 근처 z = R(R = 0.995 정도)에 두어서
    평탄하게 => H(z) = 1 - z^(-1) / (1 - 0.995z^(-1))
3. b, a 배열로 옮기고 -> tf2zpk 로 극점, 영점 출력 -> 안정성 확인
4. freqz 로 주파수 응답 그려서 DC 에서 - 무한대 dB로 떨어지는 지 확인
5. impulse response 도 lfilter 로 그려보기 

 ++ bonus : R을 0.995 대신 0.5로 바꾸면 통과대역 모양이 어떻게 달라짐?

============================================================

DSP 에서의 DC = 0Hz = 안변하는 성분 = 신호의 평균값(offset)

    w = 0 인 안도는 신호
    DC offset 이 있다 = 0이 아닌 값을 중심으로 진동시킴 => 평균값(시간평균)이 0이 아니다

        - 떠 있는 정도가 DC offset 
        - DC 는 직류 => 변하지 않음 = 진동안함

**DC blocker 에서 0Hz 를 없애면 DC offset 이 0이됨 
    (DC blocker : 아주 낮은 cutoff freq 가진 high pass filter)

    why? 

    e^jw 에서의 w=0이라면 e^j0 = 1 이 됨 => 회전을 하나도 안함 1 1 1 1 1 ...
    DC = 시간이 지나도 절대 변하지 않는 신호

    0을 기준으로 움직이는 sine 파의 평균은 0 
    3을 기준으로 움직이는 sine 파의 평균은 3 (위로 전체적으로 떠있으므로)

    x[n] = sin(wn) + 3     => 근데 여기서의 3 3 3 3 3.. 은 주파수가 0Hz 이다 
                                ( 안 움직이니까 !)

    DC blocker 의 역할 sin(wn) + 3 에서 - 3 해서 sine 파만 남기는것
        => 다시 평균이 0이 되게 하는것 

    How? (그럼 3을 빼는 법은?)

    3을 지우는게 아니고, 시간이 지나도 변하지 않는 성분을 지우는 것 
        (3 3 3 3... 이니까 시간이 지나도 변하지 않음)

        -DC : 안변하는 것 (상수성분)
        -Audio : 변하는 것

    High pass filter 로 변하지 않는 0Hz 를 없애는것!!! 변하는것(Audio)만 남김

    => 전 샘플과 현 샘플이 같다면 사라짐 (0Hz 는 변하지 않으니까 사라짐)
    y[n] = x[n] - x[n-1] 그래서 앞부분 식이 이렇게 나옴
    
"""

"""코드 작성 순서
1. 라이브러리 가져오기
2. 필터 계수 b, a 정하기
    # 이게 핵심. DC 를 없애려면 zero = 1이 나오게끔 H(z) 만들어야 함 
    => 곧 분자가 0이 되는 z가 1이여야 하므로 1 - z^(-1) 이 분자 (b는 z^0, z^(-1)의 계수 순서)
    => 1 - Rz^(-1)의 R이 0.995 로 셋팅해줘야 기울기가 가팔라짐 분모는 1 - 0.995z^(-1)
    ❗️왜 R 이 그 숫자인지? 
3. 극점/영점 뽑아서 출력 
4. 안정성 확인
5. 주파수 응답 계산
6. 그래프 그리기 (DC 에서 0으로 떨어지는지)
7. 임펄스 응답 그려보기 

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

# 2. 필터 계수 b, a 구하기
b = np.array([1, -1]) 
a = np.array([1, -0.995]) # np.array 붙여서 배열 만들기


# 3. 극점/영점 뽑아서 출력
z, p, _ = signal.tf2zpk(b, a)

print(f"pole = {p}, zero = {z}")
    # pole = [0.995], zero = [1.] 이 됨
    # zero 가 1 이기 때문에 DC 는 0이 됨
    # pole 은 0.995 이기 때문에 .. 극점을 영점 바로 옆 0.995에 두면 DC 를 살짝 벗어난
    # 다른 주파수는 영점과 극점의 효과가 서로 상쇄돼서 이득이 거의 1로 평평해짐 
        # => 기울기 가파르게 해서 다른 주파수 대역에 영향 안가게함

# 4. 안정성 확인 (안정성은 pole(극점)만의 문제 =>)
# 임펄스 응답 h[n] = (분자) * (극점)^n 꼴 이니까 거기서 시간이 지날수록의 크기를 결정하는 건 극점
print(f"stability : all poles |pole| < 1? {np.all(np.abs(p) < 1)}")
    # ❗️ np.all 은? 모든 배열의 원소가 전부 True 인지를 검사 -> 하나라도 False 면 False
    # 여기선 모든 극점이 1보다 작은지를 확인해야 함

# 5. 주파수 응답 계산 
w, h = signal.freqz(b, a, worN=500)

magnitude = np.abs(h)
    # h 는 원점과의 거리 표현하는 복소수 형태 => 라서 크기를 뽑아냄

magnitude_db = 20 * np.log10(magnitude + 1e-10) #magnitude 의 dB화

fig, axes = plt.subplots(1, 2, figsize =(12, 8))

# 6. 그래프 그리기 
ax = axes[1]

# ax.plot(w/np.pi, magnitude_db, linewidth=2.5, color='blue')
    # x축 정규화, y축 먼저 넣고 그 다음 linewidth ...
ax.semilogx(w[1:]/np.pi, magnitude_db[1:], linewidth=2.5, color='blue')
    # semilogx : x축만 로그로 y축은 그대로
    # [1:] 으로 첫 점 (0Hz) 빼고 그리기
ax.set_xlim(1e-3, 1)
    # 아주 낮은 주파수 부터 보이게
ax.axhline(0, color='r', linestyle='--', alpha=0.5, label="-3dB(cuttoff)")
ax.set_xlabel('Normalized Frequency')
ax.set_ylabel('Magnitude(dB)')
ax.set_title('Frequency response - DC blocker')
ax.grid(True, alpha=0.3, which='both')
ax.legend()
ax.set_ylim(-60, 5)
    # set_ylime(-40, 5)의 기준 : 영점이 만약 있다면 -200(+1e-10)붙여놓은 그값 까지 갈 수 도있음
    # 최대값이 0dB 이니까 그것의 +5한 여유공간까지 , 그리고 아래는 한 -40까지 보기좋게 다듬음
    # ax.legend() : 범례 표시 (label이 붙은 요소를 찾아서 범례 박스를 그려라)

    # 이렇게 했더니 0Hz 부근이 매우 좁아서 잘 안보임 -> log 로 x 축을 처리해서 아래주파수 넓게 보기
    # => 이렇게해서 0Hz 없앰! 약간 그래프상 0Hz 가 안보여서 좀 떠보이게 보이지만 로그는 0Hz 표현 불가하므로 어쩔 수 없음

# 7. impulse response 그리기
ax = axes[0]

impulse = np.zeros(50)
impulse[0] = 1

y = signal.lfilter(b, a, impulse)
    # 임펄스 응답 알아낼 수 있음

ax.stem(y, basefmt = ' ')
    # x 는 안채우면 그냥 인덱스 (0-49까지로 자동으로 채워짐)
ax.set_ylabel('Amplitude')
ax.set_title('Impulse response')
ax.grid(True, alpha = 0.3)
ax.set_ylim(-0.006, 0.001)
    # DC blocker 의 impulse response 는 index 0 에서 1 크게 올라갔다가 음수로 살짝 내려갔다가
    # 다시 원래대로 0으로 수렵하는데, 지금의 음수는 너무 작아서 안보임
ax.text(0.5, 0.1, 'Slowly decays to zero\nDC blocker = IIR filter',
        transform=ax.transAxes, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    # ha= 'center' : 가운데 정렬
    # facecolor = 박스색
    

""" 직접 그려보는 h[n] impulse response

    (분자)*(극점)^n => 이 계산법은 여기서 못씀! (특정 상황에서만 쓸수있음)

        - 분자가 상수일때만 위 공식으로 h[n]을 쓸수있음 
        - 여기서는 1 - z^(-1) 이렇게 있으므로 그냥 아래처럼 점화식으로 (차분방정식) 푸는게 편함

    H(z) = (1 - z^(-1)) / (1 - 0.995z^(-1))이였는데 이걸 차분방정식으로 풀어내면
    => Y(z)/X(z) = H(z)
    => Y(z) = H(z) * X(z)

        -z^(-1) : 어떤 신호의 한 샘플 전
            ex. z^(-1)Y(z) = y[n-1], z^(-1)X(z) = x(n-1)
    => Y(z) / X(z) = 1 - z^(-1) / (1 - 0.995z^(-1))
    => X(z)(1-z^(-1)) = Y(z)(1-0.995z^(-1))   둘이 같으니 대각선으로 곱해서 = 붙일수있음 (십자곱)
    => X(z) - z^(-1)X(z) = Y(z) - 0.995z^(-1)Y(z)
    => Y(z) = X(z) - z^(-1)X(z) + 0.995z^(-1)Y(z)

    **여기서 각 항을 시간영역으로 바꿈! (z^(-1)은 한 샘플전으로 번역!)
    => y[n] = x[n] - x[n-1] + 0.995y[n-1]
            **이렇게 차분방정식이 나오게 됨! 

    차분방정식 y[n] = x[n] - x[n-1] + 0.995y[n-1] 
    입력은 [1, 0, 0, ...]

    h[0] = 1 - 0 + 0 = 1
    h[1] = 0 - 1 + 0.995 = -0.005 
    h[2] = 0 - 0 + (-0.005)*0.995 = -0.004975
    h[3] = 0 - 0 + (-0.004975)*0.995 = -0.00495012
    ....
    이렇게 가다가 0으로 수렴함
    (근데 0.995라는 수가 1과 매우 가까워서 매우 느리게 0으로 가긴 함 1000샘플 지나도 완전히 0 은 아님)
    

    => **DC blocker**는 IIR filter : 출력 피드백이 함께 쓰임
    """


plt.tight_layout()
plt.show()




