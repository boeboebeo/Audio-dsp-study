import numpy as np
import matplotlib.pyplot as plt

# Parameters
sample_rate = 44100
freq = 100  # Hz
duration = 0.05  # 50ms (short for easy viewing)
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# Step 1: Generate Impulse Train
period_samples = int(sample_rate / freq)
impulse_train = np.zeros_like(t)
impulse_train[::period_samples] = 1

# Step 2: Leaky Integration
leak = 0.995  # Slow decay (typically 0.999 in practice)
sawtooth = np.zeros_like(t)
accumulator = 0

# Store each step (for animation)
acc_history = []

for i in range(len(impulse_train)):
    accumulator = leak * accumulator + impulse_train[i]
    sawtooth[i] = accumulator
    acc_history.append(accumulator)

# Step 3: Normalization
sawtooth_normalized = sawtooth / np.max(np.abs(sawtooth))
sawtooth_normalized = 2 * sawtooth_normalized - 1  # -1 ~ +1

# ============================================
# Visualization 1: Complete Process (4 steps)
# ============================================
fig, axes = plt.subplots(4, 1, figsize=(10, 8))

# 1. Impulse Train
axes[0].stem(t * 1000, impulse_train, basefmt=' ')
axes[0].set_ylabel('Amplitude', fontsize=9)
axes[0].set_title('Step 1: Impulse Train (Input)', fontsize=10, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, duration * 1000)

# Mark impulse positions
impulse_positions = np.where(impulse_train > 0)[0]
for pos in impulse_positions[:3]:
    axes[0].annotate(f'Imp\n#{impulse_positions.tolist().index(pos)+1}', 
                     xy=(t[pos] * 1000, 1), 
                     xytext=(t[pos] * 1000, 1.3),
                     ha='center', fontsize=7,
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# 2. Leaky Integration (before normalization)
axes[1].plot(t * 1000, sawtooth, linewidth=1.5, color='blue')
axes[1].set_ylabel('Amplitude', fontsize=9)
axes[1].set_title('Step 2: Leaky Integration (Accumulation)', fontsize=10, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, duration * 1000)

# Show leak effect
mid_point = len(t) // 4
axes[1].annotate('', xy=(t[mid_point + 100] * 1000, sawtooth[mid_point + 100]),
                xytext=(t[mid_point] * 1000, sawtooth[mid_point]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
axes[1].text(t[mid_point + 50] * 1000, sawtooth[mid_point] * 0.9,
            f'Decay by\nleak ({leak}×)',
            fontsize=7, color='red',
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

# 3. Normalized Sawtooth
axes[2].plot(t * 1000, sawtooth_normalized, linewidth=1.5, color='green')
axes[2].set_ylabel('Amplitude', fontsize=9)
axes[2].set_title('Step 3: Normalized Sawtooth (Output)', fontsize=10, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(1, color='red', linestyle='--', alpha=0.3, linewidth=1)
axes[2].axhline(-1, color='red', linestyle='--', alpha=0.3, linewidth=1)
axes[2].set_xlim(0, duration * 1000)
axes[2].set_ylim(-1.5, 1.5)

# 4. Zoomed (one period)
one_period_samples = period_samples * 2
axes[3].plot(t[:one_period_samples] * 1000, 
             impulse_train[:one_period_samples], 
             'o-', label='Impulse', markersize=3, alpha=0.7)
axes[3].plot(t[:one_period_samples] * 1000, 
             sawtooth_normalized[:one_period_samples], 
             's-', label='Sawtooth', markersize=2, linewidth=1.5)
axes[3].set_xlabel('Time (ms)', fontsize=9)
axes[3].set_ylabel('Amplitude', fontsize=9)
axes[3].set_title('Step 4: Zoom (One Period)', fontsize=10, fontweight='bold')
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# Visualization 2: Step-by-step Animation Effect
# ============================================
fig, axes = plt.subplots(3, 2, figsize=(10, 8))

# Three time points (before, at, after impulse)
positions_to_show = [
    impulse_positions[0] - 20,  # Before impulse
    impulse_positions[0],       # At impulse
    impulse_positions[0] + 20,  # After impulse
]

for idx, pos in enumerate(positions_to_show):
    if pos < 0:
        pos = 0
    
    # Left: Full graph up to current point
    axes[idx, 0].plot(t[:pos+1] * 1000, sawtooth[:pos+1], 
                      'b-', linewidth=1.5)
    axes[idx, 0].stem([t[pos] * 1000], [sawtooth[pos]], 
                      linefmt='r-', markerfmt='ro', basefmt=' ')
    axes[idx, 0].set_xlim(0, t[positions_to_show[-1] + 50] * 1000)
    axes[idx, 0].set_ylim(0, np.max(sawtooth[:positions_to_show[-1] + 50]) * 1.1)
    axes[idx, 0].set_ylabel('Accumulator', fontsize=8)
    axes[idx, 0].grid(True, alpha=0.3)
    
    # Title
    if idx == 0:
        title = f'Before Impulse (t={t[pos]*1000:.2f}ms)'
    elif idx == 1:
        title = f'AT Impulse! (t={t[pos]*1000:.2f}ms)'
    else:
        title = f'After Impulse (t={t[pos]*1000:.2f}ms)'
    axes[idx, 0].set_title(title, fontsize=9, fontweight='bold')
    
    # Right: Formula/value display
    axes[idx, 1].axis('off')
    
    # Current state
    if pos > 0:
        prev_acc = sawtooth[pos-1]
    else:
        prev_acc = 0
    
    curr_impulse = impulse_train[pos]
    curr_acc = sawtooth[pos]
    
    info_text = f"""
    Sample: {pos}
    Time: {t[pos]*1000:.3f} ms
    
    ━━━━━━━━━━━━━━━━━━━
    
    Leaky Integration:
    acc[n] = leak × acc[n-1] + imp[n]
    
    ━━━━━━━━━━━━━━━━━━━
    
    Calculation:
    acc[{pos}] = {leak} × {prev_acc:.4f} + {curr_impulse}
           = {leak * prev_acc:.4f} + {curr_impulse}
           = {curr_acc:.4f}
    
    ━━━━━━━━━━━━━━━━━━━
    """
    
    if curr_impulse == 1:
        info_text += "\n     Impulse occurs!\n    → Value jumps"
        bg_color = 'yellow'
    else:
        info_text += f"\n    → Decaying by {(1-leak)*100:.1f}%"
        bg_color = 'lightblue'
    
    axes[idx, 1].text(0.1, 0.5, info_text, 
                     fontsize=7, family='monospace',
                     verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.3))

axes[-1, 0].set_xlabel('Time (ms)', fontsize=8)
plt.tight_layout()
plt.show()

# ============================================
# Visualization 3: Leak Value Comparison
# ============================================
leak_values = [1.0, 0.999, 0.995, 0.99, 0.95, 0.9]

fig, axes = plt.subplots(len(leak_values), 1, figsize=(10, 8))

for idx, leak_val in enumerate(leak_values):
    saw = np.zeros_like(t)
    acc = 0
    
    for i in range(len(impulse_train)):
        acc = leak_val * acc + impulse_train[i]
        saw[i] = acc
    
    # Normalization
    if np.max(np.abs(saw)) > 0:
        saw_norm = saw / np.max(np.abs(saw))
        saw_norm = 2 * saw_norm - 1
    else:
        saw_norm = saw
    
    axes[idx].plot(t * 1000, saw_norm, linewidth=1.2)
    axes[idx].set_ylabel('Amp', fontsize=8)
    axes[idx].set_title(f'leak = {leak_val}', fontsize=9, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim(0, duration * 1000)
    axes[idx].set_ylim(-1.5, 1.5)
    
    # Evaluation
    if leak_val == 1.0:
        comment = "[X] No decay (steps)"
        color = 'red'
    elif leak_val < 0.95:
        comment = "[X] Too fast decay"
        color = 'orange'
    elif leak_val == 0.999:
        comment = "[O] Ideal!"
        color = 'green'
    else:
        comment = "[o] Sawtooth formed"
        color = 'blue'
    
    axes[idx].text(0.98, 0.95, comment, 
                  transform=axes[idx].transAxes,
                  fontsize=7, color=color, fontweight='bold',
                  ha='right', va='top',
                  bbox=dict(boxstyle='round', facecolor='white', 
                           alpha=0.8, edgecolor=color, linewidth=1.5))

axes[-1].set_xlabel('Time (ms)', fontsize=8)
plt.suptitle('Sawtooth Shape vs Leak Value', 
             fontsize=11, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ============================================
# Visualization 4: Numerical Analysis
# ============================================
print("=" * 60)
print("Leaky Integration Analysis")
print("=" * 60)
print(f"Frequency: {freq} Hz")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Leak Coefficient: {leak}")
print(f"Period: {1/freq*1000:.2f} ms")
print(f"Samples per Period: {period_samples}")
print()

# One cycle analysis
one_cycle = sawtooth[:period_samples]
print(f"One Cycle Analysis:")
print(f"  Max value: {np.max(one_cycle):.6f}")
print(f"  Min value: {np.min(one_cycle):.6f}")
print(f"  Avg decay rate: {(1 - np.mean(np.diff(one_cycle[1:]))):.6f}")
print()

# Before/after impulse values
imp_idx = impulse_positions[1]
print(f"Impulse #{2} Before/After:")
print(f"  Before: {sawtooth[imp_idx-1]:.6f}")
print(f"  Impulse: {impulse_train[imp_idx]:.6f}")
print(f"  After: {sawtooth[imp_idx]:.6f}")
print(f"  Jump size: {sawtooth[imp_idx] - sawtooth[imp_idx-1]:.6f}")
print()

print("=" * 60)