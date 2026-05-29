"""
Real-time SVF Filter with wav file input
=========================================
Commands:
  lp    : Lowpass
  hp    : Highpass
  bp    : Bandpass
  c+    : Cutoff up
  c-    : Cutoff down
  q+    : Resonance up
  q-    : Resonance down
  quit  : Quit
"""

import sounddevice as sd
import soundfile as sf
import numpy as np

# ── Load wav file ──────────────────────────────────
WAV_PATH = '/Users/goeun/Desktop/0/ai-learning/Librosa-basics/audio_sample/noise.wav'

audio, SAMPLE_RATE = sf.read(WAV_PATH, dtype='float32')

# if stereo, convert to mono
if audio.ndim > 1:
    audio = audio[:, 0]

print(f"  Loaded: {WAV_PATH}")
print(f"  Sample rate: {SAMPLE_RATE} Hz")
print(f"  Duration: {len(audio)/SAMPLE_RATE:.2f} sec")

# ── Filter parameters ──────────────────────────────
cutoff    = 1000.0
resonance = 1.0
mode      = 'lowpass'

# ── SVF state ──────────────────────────────────────
lp, bp = 0.0, 0.0

# ── Playback position ──────────────────────────────
pos = 0  # current sample index

BLOCK_SIZE = 256

    #block : wav에서 잘라온 256개 샘플덩어리.
    #44100개의 샘플들을 한번에 처리하면 너무 느려서 256 개 씩 잘라서 처리함
        # block 1: [0.5, 0.3, 0.1, ... 256개]
        # block 2: [다음 256개]
        # block 3: [그 다음 256개]
        # ...
    # 그 256개의 샘플 하나하나 각각을 block 함수에서 처리함 for문으로 
        # SVF 는 각 샘플이 이전 샘플에 의존하기 때문에 (lp, bp 상태가 계속 바뀌니까) 순서대로 처리해야 해서 for 문이 불가피 함
    # but, FIR 필터처럼 Feedback 이 없는 필터는 numpy 연산으로 한번에 처리 가능함
        # 그래서 Python 에서 IIR 이 느림 (for 문으로 순차적으로 처리해야해서)
        # C++ 은 그런 작업에 최적화 되어있어서 빠름

def svf_block(block):
    global lp, bp
    f = float(np.clip(2 * np.sin(np.pi * cutoff / SAMPLE_RATE), 0.0001, 1.9999))
    q = 1 / resonance
    out = np.zeros_like(block)
    for i, x in enumerate(block):
        hp = x - q * bp - lp
        bp = f * hp + bp
        lp = f * bp + lp
        # reset if values explode
        if not (np.isfinite(lp) and np.isfinite(bp)):
            lp, bp = 0.0, 0.0
        if mode == 'lowpass':
            out[i] = np.clip(lp, -1.0, 1.0)
        elif mode == 'highpass':
            out[i] = np.clip(hp, -1.0, 1.0)
        elif mode == 'bandpass':
            out[i] = np.clip(bp, -1.0, 1.0)
    return out


    # **비유를 하자면..**
    # block (256개) -> 버퍼 (한번에 가져오는 단위)
    # 약간 callback() 이 OS 스케줄러 (언제 가져올지 정함)
    # Block size = 256 <- 버퍼 크기 

""" **흐름**

    wav 파일
        ↓ 256개씩 가져옴 (버퍼링)
    callback()
        ↓ 블록 전달
    svf_block()     ← CPU가 여기서 실제 계산
        ↓ 필터된 256개
    스피커 출력

"""

def callback(outdata, frames, time, status):
    global pos
    # read next block from wav (loop when end reached)
    end = pos + frames
    if end <= len(audio):
        block = audio[pos:end]
    else:
        # loop back to start
        block = np.concatenate([audio[pos:], audio[:end - len(audio)]])
        pos = 0
    pos = end % len(audio)

    filtered = svf_block(block)
    outdata[:, 0] = filtered
    if outdata.shape[1] > 1:
        outdata[:, 1] = filtered

# ── Start stream ───────────────────────────────────
stream = sd.OutputStream(
    samplerate=SAMPLE_RATE,
    blocksize=BLOCK_SIZE,
    dtype='float32',
    channels=2,
    callback=callback
)

print()
print("Commands: lp / hp / bp / c+ / c- / q+ / q- / quit")
print()

with stream:
    while True:
        cmd = input(">> ").strip()
        if cmd == 'quit':
            break
        elif cmd == 'lp':
            mode = 'lowpass'
        elif cmd == 'hp':
            mode = 'highpass'
        elif cmd == 'bp':
            mode = 'bandpass'
        elif cmd == 'c+':
            cutoff = min(cutoff * 1.5, 18000)
        elif cmd == 'c-':
            cutoff = max(cutoff / 1.5, 20)
        elif cmd == 'q+':
            resonance = min(resonance * 1.5, 20)
        elif cmd == 'q-':
            resonance = max(resonance / 1.5, 0.1)

        print(f"  mode={mode}  cutoff={cutoff:.0f}Hz  Q={resonance:.2f}")

print("Stopped.")