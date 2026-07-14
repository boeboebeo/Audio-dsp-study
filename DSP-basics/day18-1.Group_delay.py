"""
==============================================
DAY 18: Group Delay & All-pass Filters
==============================================

MATHEMATICAL PREREQUISITES:
- Derivatives (미분): Rate of change of phase
- All-pass transfer function structure
- Cascade systems (다단 연결)

KEY CONCEPT:
Group Delay = "Speed" at which signal travels through filter
Different frequencies travel at different speeds
= Temporal smearing of transients

All-pass filter = Only changes phase, NOT magnitude
Uses: Compensate for phase distortion, create effects
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_group_delay():
    """what is group delay? 
    
    Group delay 

        - Formula : τ_g(ω) = -dφ(ω) / dω
                    (Negative derivative of phase)

            +Meaning : How many samples does each frequency delay? 
            (각 주파수가 몇 샘플 지연되는가?)

    [THE PROBLEM] Temporal Smearing (시간적 번짐)

    Imagine sharp transient (like impulse)
        - Contains all frequencies
        - All frequencies arrive at DIFFERENT TIMES
        - Transient becomes Blurred!

    Audible as:
        - Less punch
        - muddy bass
        - unclear transients
        
    [THE IDEAL] Constant Group delay
    If all frequencies have same group delay?
        - all arrive at same time
        - transient shape preserved
        - punch and clarity maintained
        - when do you get constant group delay? : Linear phase filters
    
    """

def calculate_group_delay():
    # Visualize and Calculate 

    # Design filter
    # FIR (linear phase) - should have constant group delay
    fir_linear = signal.firwin(51, 0.3)
        # symmetric 

    # IIR (minimum phase) - variable group delay
    b_iir, a_iir = signal.butter(6, 0.3)
        # IIR 6th order 

    # Calculate group delay
    w_fir, gd_fir = signal.group_delay((fir_linear, [1]), w=np.linspace(0, np.pi, 500))
    w_iir, gd_iir = signal.group_delay((b_iir, a_iir), w=np.linspace(0, np.pi, 500))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # plot 1: FIR group delay (should be flat)
    ax = axes[0, 0]
    ax.plot(w_fir/np.pi, gd_fir, linewidth=2.5, color='blue')
    ax.set_ylabel('Group delay (samples)')
    ax.set_ylim(20, 30)
        # matplotlib 은 변화값이 극도로 작을때, 자동으로 offset notation 으로 표기하는데 (y축을)
        # ylim 없다면 1e-8 + 2.5e1 이렇게 처리함 
        # 2.5 e1 = 2.6 * 10^(1) = 25
        # 1e-8 = 10 * (-8) = 0.00000001 
        # 이렇다는건 실제 y = 0 인 눈금은 실제로는 25.00000000
        # 눈금 1 = 25.00000001 을 표기함
        # 매우 변동이 미세하기 때문에 이렇게 확대해서 보여줌
    ax.set_title('Linear phase FIR\nGroup delay')
    ax.grid(True, alpha=0.3)

    ax.text(0.5, 0.9, 'FLAT!\n(same delay with all frequencies)', 
           transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 2: IIR group delay (should vary!)
    ax = axes[0, 1]
    ax.plot(w_iir/np.pi, gd_iir, linewidth=2.5, color='red')
    ax.set_ylabel('Group delay (samples)')
    ax.set_title('Minimum phase iir\nGroup delay')
    ax.grid(True, alpha=0.3)

    ax.text(0.5, 0.9, 'VARIABLE!\n(different delay with frequencies)', 
           transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
        # Plot 3: Comparison
    ax = axes[1, 0]
    ax.plot(w_fir/np.pi, gd_fir, linewidth=2.5, label='FIR (constant)', color='blue')
    ax.plot(w_iir/np.pi, gd_iir, linewidth=2.5, linestyle='--', label='IIR (variable)', color='red')
    ax.set_ylabel('Group Delay (samples)')
    ax.set_xlabel('Normalized Frequency')
    ax.set_title('Comparison\n(FIR is flat, IIR varies)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = """
    GROUP DELAY SUMMARY
    ──────────────────
    
    What it measures:
    • Sample delay at each frequency
    • Calculated from phase derivative
    • τ_g(ω) = -dφ/dω
    
    Flat group delay :
    • All frequencies same delay
    • Transient preserved
    • Punchy sound
    • Linear phase filters have this!
    
    Variable group delay :
    • Different frequencies delayed differently
    • Transient blurred
    • Muddy sound
    • Minimum phase filters have this
    
    In practice:
    • Check GD for mastering filters
    • Constant GD = better transients
    • Especially important for drums
    """
    
    ax.text(0.05, 0.95, summary, fontsize=8.5, verticalalignment='top',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # group delay 의 정의
    # = 각 주파수 성분이 필터를 지나면서 시간이 얼마나 지연되는가 (샘플 또는 초 단위)
    # group delay 는 FIR, IIR 에 둘다 존재하며 모양이 다름 
    # group delay = τg​(ω) = D (상수, fir 에서는) 
    

    plt.tight_layout()
    plt.show()

def understand_allpass_filters():
    """All-pass filter : Only change phase( not magnitude )
    
    Transfer function(1st order):
        H(z) = (a + z^(-1) / (1 + a*z^(-1))
        where 0 < a < 1

        **이렇게 transfer fuction 이 배치되어 있어야 단위원 기준으로 거울처럼 배치됨
        ex. Pole = 0.6 이라면 Zero = 1/0.6 에 있다.
            => 이렇게 해야 magnitude 가 완벽하게 상쇄됨
            + Pole 은 크기를 키우려 하고
            + Zero 는 크기를 줄이려 하니까 => 둘이 정확히 상쇄해서 항상 magnitude 는 1이 된다. (변화 x)

        but, phase 는 바뀐다.
        크기는 pole 이 +3dB 만들때, zero 는 -3dB 처리하지만
        시간은 pole 이 조금 늦출때, zero 도 조금 늦춘다.

    + Magnitude property
        |H(e^jω)| = 1 for ALL frequencies
        (모든 주파수에서 크기 = 1)

    + Phase property
        Angle H(e^jω) varies with frequency
        (위상은 주파수에 따라 변함)

    [what does it do?]

    Input signal -> All-pass -> Output signal

        - Phase compensation (위상보정)
        - Group delay equalization
        - Creating phase effects
        - Converting linear <-> minimum phase

    [pole-zero pattern]

    all-pass filters have special property

        - Pole at z = a
        - Zero at z = 1/a
        - Zero is Outside unit circle (영점이 단위원 밖에)

    This unusual structure
        - keeps magnitude flat (|H| = 1)
        - But changes phase 

    """

def create_allpass_filter():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Three all-pass filters with different 'a' values
    a_values = [0.3, 0.7, 0.9]

    for idx, a in enumerate(a_values):
        # All-pass transfer function
        b = np.array([a, 1]) # Numerator
        denom = np.array([1, a]) # Denominator (분모)
        
        # Freuqeuncy response
        w, h = signal.freqz(b, denom, worN=500)
        magnitude = np.abs(h)
        magnitude_db = 20*np.log10(magnitude + 1e-10)
        phase = np.unwrap(np.angle(h))

        # plot magnitude
        ax = axes[0, idx]
        ax.plot(w/np.pi, magnitude_db, linewidth=2.5, color='blue')
        ax.set_ylim(-0.5, 0.5)
        ax.set_ylabel('Magnitude(dB)')
        ax.set_title(f'All pass (a= {a}\n Magnitude)')
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.8, f'FLAT! |H|=1 always', transform=ax.transAxes,
               fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # plot phase
        ax = axes[1, idx]
        ax.plot(w/np.pi, phase, linewidth=2.5, color='red')
        ax.set_ylabel('Phase(radians)')
        ax.set_title(f'All-pass (a={a}\n Phase)')
        ax.grid(True, alpha=0.3)

        # More annotations for a=0.9 (most dramatic)
        if a == 0.9:
            ax.text(0.5, 0.1, 'Steeper phase change', 
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.show()

    """
    All pass filter {idx+1}: a = {a}
    H(z) = {a} + z^(-1) / 1 + {a}*z^(-1)

        - All magnitudes are flat (|H| = 1)
        - Phase changes with frequency
        - Larger 'a' = stronger phase effect
        
    """





# calculate_group_delay()
create_allpass_filter()