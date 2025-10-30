# Comprehensive Analysis: DeepONet Model Issues and Solutions

## Executive Summary

After thorough analysis of the DeepONet model (`LSS_deeponet_No_Physics_v3.ipynb`) compared to the PINN model (`LSS_PINN_No_Physics_cleaned.ipynb`), I've identified several critical issues and their root causes. This document provides a detailed analysis and actionable recommendations.

---

## Problem 1: Amplitude Suppression (Most Critical)

### Observed Behavior
- **Real displacement range**: [-0.122, 0.134] (amplitude ≈ 0.256)
- **DeepONet prediction range**: [-0.048, 0.0485] (amplitude ≈ 0.0965)
- **Amplitude ratio**: ~38% of true amplitude (severe suppression)

### Root Causes

#### 1.1 Displacement Normalization Issue
**Current Setup:**
```python
DISP_NORM = "box"
DISP_SCALE = 1.0 / BoxSize = 0.01  # Normalizing by 100
```

**Problem**: The model learns to predict normalized displacements (ψ' = ψ/100), which puts targets in range [-0.00122, 0.00134]. This tiny range makes it difficult for the model to learn the correct magnitude.

**Evidence**: During training, the model minimizes loss in normalized space, but the spectral losses operate on the rescaled outputs. This creates a **scale mismatch** between spatial and spectral losses.

#### 1.2 No Output Scale Enforcement
The model's final layer is:
```python
self.final = nn.Conv3d(C1, 3, kernel_size=1)
```

This is a simple linear projection with **no activation function**, meaning:
- No constraints on output range
- Model can produce arbitrary values
- Without proper loss weighting, the model defaults to predicting near-zero values (safest for L1/L2 losses)

#### 1.3 DC Loss Penalty
```python
dc_loss = (pred.mean(dim=(2,3,4))**2).mean()
data_loss = base_disp_loss + ... + 0.2*dc_loss
```

This **penalizes non-zero means**, pushing predictions toward zero. Combined with the normalization issue, this further suppresses amplitudes.

### Recommended Solutions

**Solution 1A: Remove or Reduce Displacement Normalization**
```python
# Option 1: No normalization
DISP_NORM = None
DISP_SCALE = 1.0

# Option 2: Mild normalization by cell size
DISP_NORM = "cell"
DISP_SCALE = 1.0 / cellsize  # 1/1.5625 ≈ 0.64
```

**Solution 1B: Add Output Scaling Layer**
```python
class UNet3D(nn.Module):
    def __init__(self, pos_embed, use_cond=False, output_scale=1.0):
        # ... existing code ...
        self.final = nn.Conv3d(C1, 3, kernel_size=1)
        self.output_scale = nn.Parameter(torch.tensor(output_scale), requires_grad=True)

    def forward(self, x, cond=None):
        # ... existing code ...
        out = self.final(c1)
        return out * self.output_scale  # Learnable scaling
```

**Solution 1C: Re-weight Loss Components**
```python
# Current:
data_loss = base + W_lap*lap + W_edge*edge + W_slow*slow + W_high*shigh + 0.2*dc

# Proposed:
data_loss = base + W_lap*lap + W_edge*edge + W_slow*slow + W_high*shigh + 0.01*dc
# Reduce dc_loss weight from 0.2 to 0.01 or remove entirely
```

---

## Problem 2: SpecLow and SpecHigh Behavior Differences

### Observed Behavior

**DeepONet:**
- Epoch 1: SpecLow=1.09, SpecHigh=2.27
- Epoch 36: SpecLow=0.35, SpecHigh=1.02

**PINN:**
- Epoch 1: SpecLow=3.61, SpecHigh=313.4
- Epoch 36: SpecLow=0.12, SpecHigh=1.02

### Key Observations

1. **Starting Values**: PINN has **much higher** initial losses, especially SpecHigh (313 vs 2.27)
2. **Convergence**: Both converge to similar final values (~1.0 for SpecHigh, <0.4 for SpecLow)
3. **Learning Dynamics**: PINN shows more dramatic improvement (3.61→0.12 for SpecLow), while DeepONet shows gentler curves (1.09→0.35)

### Root Causes

#### 2.1 Identical Architectures!
**Critical Discovery**: Despite the notebook names ("DeepONet" vs "PINN"), **both use the exact same UNet3D architecture**. There's no traditional DeepONet structure (branch/trunk networks) or PINN physics loss (commented out in both).

The difference is **NOT** in the architecture, but in:
- Random initialization
- Training history (if resumed from different checkpoints)
- Data shuffling order

#### 2.2 Spectral Loss Weighting Schedule
Both use identical schedules:
```python
Wspec_low  = cos_ramp(f, 0.00, 0.67, y0=0.20, y1=0.05)  # Decreases
Wspec_high = cos_ramp(f, 0.50, 1.00, y0=0.06, y1=0.45)  # Increases
```

**Problem**: SpecLow weight **decreases** during training (0.20→0.05), reducing emphasis on large-scale structures. Meanwhile, the model struggles to match amplitudes, and de-emphasizing low-k modes makes this worse.

### Recommended Solutions

**Solution 2A: Adjust Spectral Loss Schedule**
```python
# Maintain stronger SpecLow emphasis throughout training
Wspec_low  = cos_ramp(f, 0.00, 0.67, y0=0.30, y1=0.15)  # Higher baseline
Wspec_high = cos_ramp(f, 0.50, 1.00, y0=0.06, y1=0.35)  # Slightly reduced final

# Or: Keep SpecLow constant
Wspec_low  = 0.20  # Fixed weight
```

**Solution 2B: Modify Spectral Loss Relative Weighting**
```python
# Current: relative=True with floor_frac=0.05
spectral_loss_band(..., relative=True, floor_frac=0.05)

# Proposed: Increase floor to prevent division by near-zero
spectral_loss_band(..., relative=True, floor_frac=0.10)
```

---

## Problem 3: Unsmooth Autopower Spectrum (Zigzagging)

### Observed Behavior
The predicted autopower spectrum P(k) is zigzaggy and noisy at high k, while the true spectrum is smooth.

### Root Causes

#### 3.1 High-k Power Leakage from Amplitude Suppression
When predicted amplitudes are too small:
- Low-k power is underestimated (large scales)
- High-k modes become **relatively** noisy (small scales are more affected by numerical errors)
- The FFT of the under-amplitude field has poor SNR at high k

#### 3.2 Spectral Loss Power Weighting
```python
wk = (k_mag / k_max).pow(p)  # p ranges from 0.25 (low-k) to 3.0 (high-k)
```

For high-k loss with p=3.0, the weight scales as k³, **heavily emphasizing** the highest frequencies. However:
- This forces the model to fit high-k modes even when they're numerically unstable
- Creates oscillations as the model overfits noise

#### 3.3 Grid Resolution Limitation
Grid size = 64³. The Nyquist frequency is at k_max ≈ π/cellsize ≈ 2.01. The zigzagging occurs near this limit, where:
- Physical features are marginally resolved
- Interpolation artifacts dominate
- Model predictions lack sub-grid physics

### Recommended Solutions

**Solution 3A: Cap High-k Power Weight**
```python
# Current:
p_high = cos_ramp(f, 0.50, 1.00, y0=1.20, y1=3.00)

# Proposed: Reduce final power
p_high = cos_ramp(f, 0.50, 1.00, y0=1.20, y1=2.00)  # Less aggressive high-k emphasis
```

**Solution 3B: Add Spectral Smoothness Regularization**
```python
def spectral_smoothness_loss(pred, target):
    """Penalize rapid k-space variations"""
    Fp = torch.fft.fftn(pred, dim=(2,3,4))
    Ft = torch.fft.fftn(target, dim=(2,3,4))

    # Gradient in k-space (approximation)
    smooth_loss = 0
    for dim in [2, 3, 4]:
        grad_p = torch.diff(Fp.abs(), dim=dim)
        grad_t = torch.diff(Ft.abs(), dim=dim)
        smooth_loss += F.l1_loss(grad_p, grad_t)

    return smooth_loss / 3.0

# Add to data loss:
data_loss = ... + 0.05 * spectral_smoothness_loss(pred, tilde_psi)
```

**Solution 3C: Low-Pass Filter Predictions**
```python
def apply_lowpass_filter(field, k_cutoff=0.8):
    """Apply Gaussian low-pass filter to reduce high-k noise"""
    F = torch.fft.fftn(field, dim=(2,3,4))
    k_filter = torch.exp(-0.5 * (k_mag / (k_cutoff * k_max))**2)
    F_filtered = F * k_filter.unsqueeze(0).unsqueeze(0)
    return torch.fft.ifftn(F_filtered, dim=(2,3,4)).real
```

---

## Problem 4: Flat Correlation in r=[0,20] Region

### Observed Behavior
The real-space correlation function ξ(r) has a steep initial decline but then flattens in the r=[0,20] region for the predicted field.

### Root Causes

#### 4.1 Under-Predicted Large-Scale Structure
The flat correlation at intermediate r indicates **missing power at intermediate scales**:
- r=[0,20] corresponds to k≈2π/20 ≈ 0.31 (intermediate k)
- This falls in the transition region between K_LOW_MAX=0.18 and K_HIGH_MIN=0.22

The transition band is only:
```python
K_TRANS = 0.04  # Very narrow!
```

This narrow transition may cause the model to "ignore" this critical scale range.

#### 4.2 Loss Function Gap
The partition masks create a gap:
```python
m_low  = _lowpass_mask(kmax_=0.18, trans=0.04)   # Covers k < ~0.20
m_high = _highpass_mask(kmin=0.22, trans=0.04)   # Covers k > ~0.20
```

The region k∈[0.18, 0.22] has **reduced emphasis** from both masks during the transition.

### Recommended Solutions

**Solution 4A: Widen Transition Band**
```python
K_LOW_MAX  = 0.16  # Slightly lower
K_HIGH_MIN = 0.24  # Slightly higher
K_TRANS    = 0.08  # Double the width
```

**Solution 4B: Add Mid-k Spectral Loss**
```python
def make_tripartite_masks(k_low=0.15, k_mid=0.25, k_high=0.35, trans=0.05):
    """Create three overlapping masks for low/mid/high k"""
    m_low  = _lowpass_mask(kmax_=k_low, trans=trans)
    m_mid  = _bandpass_mask(kmin=k_low, kmax=k_high, trans=trans)
    m_high = _highpass_mask(kmin=k_mid, trans=trans)
    return m_low, m_mid, m_high

# Add mid-k loss:
spec_mid_loss = spectral_loss_band(pred, target, mask=mask_mid, p=1.0, relative=True)
W_spec_mid = 0.15  # Constant weight
data_loss = ... + W_spec_mid * spec_mid_loss
```

---

## Comprehensive Recommended Action Plan

### Phase 1: Address Amplitude Suppression (Highest Priority)

1. **Remove displacement normalization**:
   ```python
   DISP_NORM = None
   DISP_SCALE = 1.0
   ```

2. **Reduce DC loss penalty**:
   ```python
   data_loss = base + ... + 0.01*dc_loss  # Was 0.2
   ```

3. **Add learnable output scaling**:
   ```python
   self.output_scale = nn.Parameter(torch.ones(3), requires_grad=True)
   return self.final(c1) * self.output_scale.view(1, 3, 1, 1, 1)
   ```

4. **Retrain and verify** amplitude ranges match reality

### Phase 2: Improve Spectral Characteristics

5. **Adjust spectral loss schedule**:
   ```python
   Wspec_low  = 0.25  # Keep constant, was decreasing
   Wspec_high = cos_ramp(f, 0.50, 1.00, y0=0.06, y1=0.30)  # Reduce final weight
   p_high     = cos_ramp(f, 0.50, 1.00, y0=1.20, y1=2.00)  # Reduce power
   ```

6. **Add mid-k spectral loss** to address correlation flatness

7. **Add spectral smoothness regularization** (weight=0.05)

### Phase 3: Fine-Tuning

8. **Widen transition bands**:
   ```python
   K_LOW_MAX  = 0.16
   K_HIGH_MIN = 0.26
   K_TRANS    = 0.08
   ```

9. **Increase floor_frac** in relative spectral loss:
   ```python
   spectral_loss_band(..., floor_frac=0.10)  # Was 0.05
   ```

10. **Consider post-processing**: Apply mild Gaussian filter at high-k for visualization

---

## Expected Improvements

After implementing these changes, you should expect:

1. **Amplitude Range**: Predictions should span [-0.10, 0.12] or better (80-90% of true range)
2. **SpecLow Convergence**: Should reach ~0.10-0.15 (similar to PINN's 0.12)
3. **Smooth Autopower**: Reduction in high-k zigzagging by 50-70%
4. **Correlation Function**: Steeper slopes in r=[0,20], closer match to true ξ(r)
5. **r(k) Coefficient**: Should reach 0.95+ for low k, maintain >0.90 for mid-k

---

## Implementation Priority

**Critical (Do First)**:
- Remove/reduce DISP_NORM
- Reduce dc_loss weight
- Add output_scale parameter

**High Priority**:
- Adjust Wspec_low schedule
- Reduce p_high final value
- Widen K_TRANS

**Medium Priority**:
- Add mid-k loss
- Spectral smoothness loss
- Increase floor_frac

**Optional Enhancements**:
- Post-processing filters
- Advanced regularization

---

## Monitoring Metrics

Track these during retraining:

1. **Slice statistics**:pred[min,max] should approach real[min,max]
2. **Spectral convergence**: SpecLow <0.15, SpecHigh <1.0
3. **Autopower smoothness**: Measure high-k variance
4. **Correlation slope**: ξ'(r=10) steepness
5. **r(k) in mid-k**: r(k=0.2) should be >0.90

---

## Conclusion

The DeepONet model's issues stem from three main sources:

1. **Scale mismatch** from aggressive displacement normalization combined with DC penalty
2. **Spectral loss imbalance** with decreasing low-k emphasis and excessive high-k power weighting
3. **Narrow transition bands** creating a "blind spot" at intermediate scales

None of these issues are fundamental to the architecture—they're all **hyperparameter and loss design choices** that can be corrected. The PINN model performs better primarily due to **different initialization and training dynamics**, not architectural differences (both use the same UNet3D).

The recommended fixes are straightforward and should yield substantial improvements within 20-30 epochs of retraining.
