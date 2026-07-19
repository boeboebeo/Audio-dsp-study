"""
==============================================
DAY 21: Polyphase Filters & Efficient Resampling
==============================================

MATHEMATICAL PREREQUISITES:
- Filter decomposition (필터 분해)
- Noble identities (고귀한 항등식)
- Parallel processing (병렬 처리)

KEY CONCEPT:
Polyphase = Split a filter into M parallel filters
Each operates on different "phase" of the signal
Result: M times faster resampling!

This is what Ableton/Serum/Vital use for high-quality resampling
Used everywhere in professional audio tools
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

SAMPLE_RATE = 44100

def understand_polyphase_concept():
    """
    What is polyphase decomposition?
    Why would we want to split a filter?
    (다상 분해가 뭐고, 왜 필터를 분해할까?)
    """
    
    print("="*70)
    print("DAY 21: Polyphase Filters & Efficient Resampling")
    print("="*70)
    
    print("\n[MOTIVATION] The Problem with Naive Resampling")
    print("-" * 70)
    print("Interpolation by 3 (3배 업샘플):")
    print("  1. Insert 2 zeros between each sample")
    print("  2. Apply low-pass filter @ 1x sample rate")
    print("  3. Process 3x more samples!")
    print("\nExample:")
    print("  Input: 1000 samples")
    print("  After zero insertion: 3000 samples")
    print("  Filter length: 1000 taps")
    print("  Operations: 3000 × 1000 = 3,000,000 multiplies!")
    print("  (300만 번의 곱하기!)")
    print("\nWaste:")
    print("  2/3 of filter taps process ZEROS!")
    print("  Wasting CPU on nothing")
    
    print("\n[SOLUTION] Polyphase Decomposition")
    print("-" * 70)
    print("Idea: Don't process the zeros!")
    print("\nInstead:")
    print("  1. Split filter into 3 parallel filters")
    print("     Each processes different 'phase' of signal")
    print("  2. Each filter processes original 1000 samples")
    print("     (No zeros!)")
    print("  3. Interleave outputs")
    print("\nResult:")
    print("  3 × (1000 × 333) = 1,000,000 multiplies")
    print("  3배 빠름! (3x faster!)")
    print("\nTrade:")
    print("  More complex code")
    print("  But way faster!")

def understand_polyphase_mathematics():
    """
    How polyphase decomposition actually works
    Mathematical explanation with examples
    (다상 분해의 수학적 설명)
    """
    
    print("\n" + "="*70)
    print("[MATHEMATICS] Polyphase Decomposition")
    print("="*70)
    
    print("\nConcept: Split filter into phases")
    print("-" * 70)
    print("Original filter H(z) = [h0, h1, h2, h3, h4, h5, ...]")
    print("                       51 taps (example)")
    print("\nFor 3-phase decomposition:")
    print("  H0(z) = [h0, h3, h6, h9, ...]  (every 3rd, starting at 0)")
    print("  H1(z) = [h1, h4, h7, h10, ...]  (every 3rd, starting at 1)")
    print("  H2(z) = [h2, h5, h8, h11, ...]  (every 3rd, starting at 2)")
    print("\nEach phase filter:")
    print("  Much shorter!")
    print("  Original: 51 taps")
    print("  Each phase: ~17 taps (51÷3)")
    
    print("\n[EXAMPLE] With Numbers")
    print("-" * 70)
    print("Input signal: x[n] = [1, 2, 3, 4, 5, ...]")
    print("Filter: h[n] = [0.5, 0.3, 0.1]  (3 taps)")
    print("\nNaive upsampling by 2:")
    print("  Insert zeros: [1, 0, 2, 0, 3, 0, 4, 0, 5, ...]")
    print("  Filter this: Process 6 inputs for each real input")
    print("  Wasteful!")
    print("\nPolyphase approach:")
    print("  Phase 0 filters even samples: [1, 2, 3, 4, 5] @ h0 = [0.5]")
    print("  Phase 1 filters odd samples: [1, 2, 3, 4, 5] @ h1 = [0.3]")
    print("  Interleave: [1×0.5, 1×0.3, 2×0.5, 2×0.3, ...]")
    print("  Same result, no zeros processed!")

def visualize_polyphase_decomposition():
    """
    Show polyphase decomposition visually
    How coefficients split up
    (다상 분해를 시각적으로 보여주기)
    """
    
    print("\n" + "="*70)
    print("[VISUALIZATION] Polyphase Decomposition")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Create a filter
    b = signal.firwin(51, 0.3)
    
    # Decompose into 3 phases (interpolation by 3)
    L = 3
    
    print(f"\nDecomposing {len(b)}-tap filter into {L} phases:")
    
    # Extract phases
    phases = []
    for phase in range(L):
        phase_coeffs = b[phase::L]
        phases.append(phase_coeffs)
        print(f"  Phase {phase}: {len(phase_coeffs)} coefficients")
    
    # Plot 1: Original filter
    ax = axes[0, 0]
    n = np.arange(len(b))
    ax.stem(n, b, basefmt=' ')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Original Filter\n(51 taps)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Polyphase decomposition
    ax = axes[0, 1]
    colors = ['red', 'green', 'blue']
    for phase in range(L):
        indices = np.arange(phase, len(b), L)
        ax.scatter(indices, b[phase::L], s=100, color=colors[phase],
                  label=f'Phase {phase} ({len(phases[phase])} taps)',
                  marker=['o', 's', '^'][phase])
    
    ax.set_ylabel('Coefficient Value')
    ax.set_xlabel('Tap Index')
    ax.set_title('Polyphase Decomposition\n(Split into 3 phases)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Phase 0 filter
    ax = axes[1, 0]
    ax.stem(range(len(phases[0])), phases[0], basefmt=' ', linefmt='r-', markerfmt='ro')
    ax.set_ylabel('Coefficient Value')
    ax.set_title(f'Phase 0 Filter\n({len(phases[0])} taps)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    efficiency_text = """
    EFFICIENCY COMPARISON
    ────────────────────
    
    Naive Upsampling by 3:
    • Insert 2 zeros per sample
    • Filter processes 3× samples
    • Multiplies: N × 51 × 3 = 153N
    
    Polyphase Approach:
    • Split 51-tap → 3×17-tap
    • Process original samples only
    • Multiplies: N × 17 × 3 = 51N
    
    Speedup: 153N / 51N = 3x faster!
    (L배 빠름, L=보간 인수)
    
    Real example:
    • 44.1 → 96 kHz: 2.18x faster
    • 44.1 → 192 kHz: 4.35x faster
    • 44.1 → 384 kHz: 8.7x faster!
    (엄청 빠름!)
    
    Used in:
    ✓ Ableton Live
    ✓ Native Instruments
    ✓ iZotope
    ✓ All pro tools (프로 도구들)
    """
    
    ax.text(0.05, 0.95, efficiency_text, fontsize=8.5, verticalalignment='top',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/home/claude/day21_polyphase_decomposition.png', dpi=150)
    plt.close()
    
    print("\n✓ Polyphase decomposition visualization saved")

def understand_fractional_delay():
    """
    Bonus: Polyphase for fractional delays
    Why this is useful
    (분수 지연: 추가 응용)
    """
    
    print("\n" + "="*70)
    print("[BONUS] Polyphase for Fractional Delay")
    print("="*70)
    
    print("\nUse Case: Precise timing adjustments")
    print("-" * 70)
    print("Problem: Need to delay signal by 2.7 samples")
    print("Can't just delay by 2 or 3 (wrong!)")
    print("\nSolution: Polyphase as fractional delay")
    print("  Each phase = delay by 1/L samples")
    print("  Select appropriate phase = get fractional delay")
    print("\nExample: L=10 phases")
    print("  Phase 0 → 0.0 sample delay")
    print("  Phase 1 → 0.1 sample delay")
    print("  Phase 2 → 0.2 sample delay")
    print("  ...")
    print("  Phase 7 → 0.7 sample delay")
    print("\nUse Case:")
    print("  Pitch shifting (피치 시프팅)")
    print("  Time stretching (타임 스트레칭)")
    print("  Fine timing adjustments")
    print("  Crossfade control in mixers")

def demonstrate_polyphase_efficiency():
    """
    Show actual efficiency gains
    Measure computation time
    (실제 효율성 이득 측정)
    """
    
    print("\n" + "="*70)
    print("[PRACTICAL] Polyphase Efficiency")
    print("="*70)
    
    # Signals
    input_length = 44100  # 1 second @ 44.1 kHz
    filter_length = 2001   # 2001-tap filter (long!)
    interpolation_factor = 3
    
    print(f"\nTest case:")
    print(f"  Input: {input_length} samples")
    print(f"  Filter: {filter_length} taps")
    print(f"  Interpolation: {interpolation_factor}x")
    
    # Computational costs
    naive_samples = input_length * interpolation_factor
    naive_ops = naive_samples * filter_length
    
    polyphase_ops = input_length * (filter_length // interpolation_factor) * interpolation_factor
    # Approximately, since decomposition is cleaner
    
    print(f"\nNaive method:")
    print(f"  After zero-insertion: {naive_samples} samples")
    print(f"  Operations: {naive_samples} × {filter_length} = {naive_ops:,}")
    
    print(f"\nPolyphase method:")
    print(f"  Input stays: {input_length} samples")
    print(f"  Each phase: {filter_length // interpolation_factor} taps")
    print(f"  Operations: ~{polyphase_ops:,}")
    
    speedup = naive_ops / polyphase_ops
    print(f"\nSpeedup: {speedup:.1f}x faster!")
    print(f"(약 {interpolation_factor}배 빠름)")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    
    methods = ['Naive\n(with zeros)', 'Polyphase\n(decomposed)']
    operations = [naive_ops, polyphase_ops]
    colors = ['red', 'green']
    
    bars = ax.bar(methods, operations, color=colors, alpha=0.7, width=0.6)
    
    ax.set_ylabel('Operations (millions)', fontsize=11)
    ax.set_title('Computational Cost: Naive vs Polyphase\n(Lower is better!)', fontsize=12)
    ax.set_ylim(0, max(operations)*1.2)
    
    # Add value labels
    for bar, ops in zip(bars, operations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ops/1e6:.1f}M', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add speedup annotation
    ax.text(0.5, 0.9, f'Speedup: {speedup:.1f}x', 
           transform=ax.transAxes, fontsize=14, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           ha='center')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/day21_polyphase_efficiency.png', dpi=150)
    plt.close()
    
    print("\n✓ Efficiency comparison visualization saved")

def practical_polyphase_implementation():
    """
    How to actually use polyphase in practice
    Code examples
    (실제로 코드에서 사용하는 방법)
    """
    
    print("\n" + "="*70)
    print("[IMPLEMENTATION] Using Polyphase in Practice")
    print("="*70)
    
    print("\nIn scipy.signal:")
    print("-" * 70)
    print("scipy.signal.resample_poly does polyphase internally!")
    print("\nCode:")
    print("  from scipy.signal import resample_poly")
    print("  # Upsample by 3, downsample by 2 = 1.5x speed up")
    print("  y = resample_poly(x, up=3, down=2)")
    print("  # Internally uses polyphase decomposition")
    print("  # Automatically applies anti-aliasing!")
    print("\nAdvantages:")
    print("  ✓ Handles filtering for you")
    print("  ✓ Efficient polyphase implementation")
    print("  ✓ Just works!")
    
    print("\nUsage example:")
    print("-" * 70)
    
    # Create signal
    fs1 = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs1*duration), endpoint=False)
    signal_test = np.sin(2*np.pi*1000*t) + 0.3*np.sin(2*np.pi*5000*t)
    
    # Resample from 44.1 to 96 kHz
    fs2 = 96000
    ratio = fs2 / fs1  # 96000/44100 = 32/147 after reduction
    
    print(f"Input: {fs1} Hz, {len(signal_test)} samples")
    print(f"Output: {fs2} Hz")
    
    # Use resample_poly for exact resampling
    from fractions import Fraction
    frac = Fraction(fs2, fs1).limit_denominator(1000)
    
    print(f"Ratio: {fs2}/{fs1} = {frac.numerator}/{frac.denominator}")
    
    y = signal.resample_poly(signal_test, frac.numerator, frac.denominator)
    
    print(f"Output length: {len(y)} samples")
    print(f"✓ High-quality resampling with polyphase!")
    
    print("\nWhy professional tools use this:")
    print("-" * 70)
    print("• Fast: 3-10x speedup vs naive")
    print("• Accurate: Proper anti-aliasing built-in")
    print("• Flexible: Any ratio (not just powers of 2)")
    print("• Used everywhere: DAWs, plugins, hardware")

# Execute Day 21
if __name__ == "__main__":
    understand_polyphase_concept()
    understand_polyphase_mathematics()
    visualize_polyphase_decomposition()
    understand_fractional_delay()
    demonstrate_polyphase_efficiency()
    practical_polyphase_implementation()
    
    print("\n" + "="*70)
    print("📚 DAY 21 SUMMARY")
    print("="*70)
    print("\n1. Polyphase Decomposition:")
    print("   Split filter into L parallel filters")
    print("   Each phase handles 1/L of the signal")
    print("\n2. Why Useful:")
    print("   Naive interpolation by L: processes L× zeros")
    print("   Polyphase: skip zeros entirely")
    print("   Result: L× speedup!")
    print("\n3. Mathematics:")
    print("   H0(z) = h[0], h[L], h[2L], ...")
    print("   H1(z) = h[1], h[L+1], h[2L+1], ...")
    print("   Each phase ~1/L the length of original")
    print("\n4. Fractional Delay:")
    print("   Polyphase phase selection = fine delay control")
    print("   Used in: pitch shift, time stretch")
    print("\n5. Practical Use:")
    print("   scipy.signal.resample_poly uses polyphase")
    print("   Modern DAWs use this internally")
    print("   Ableton, Native Instruments, iZotope, etc.")
    print("\n6. Efficiency:")
    print("   3x interpolation: 3x speedup")
    print("   8x interpolation: 8x speedup")
    print("   Professional quality resampling!")
    print("="*70)