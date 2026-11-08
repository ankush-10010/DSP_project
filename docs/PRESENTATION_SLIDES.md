# 🫀 Biomedical Digital Signal Processing: ECG Conditioning, Pan-Tompkins QRS Detection, and Autonomic HRV Analytics

**Semester Project Defense & Viva Voce Slide Deck**  
**Course:** Digital Signal Processing (DSP) — Final Semester Presentation  
**Author:** DSP Research & Development Group  
**Academic Term:** Autumn Semester 2026  
**Repository:** `/home/ankush/CODE/DSP`  

---

## Slide 1: Title & Project Overview

### High-Fidelity ECG Signal Processing, Real-Time Pan-Tompkins QRS Detection, and Autonomic HRV Biomarker Extraction
**An End-to-End Biomedical DSP Pipeline Validated Against the MIT-BIH Database**

- **Presenter:** DSP Research & Development Group
- **Discipline:** Department of Electronics & Electrical Communication Engineering
- **Evaluation:** Final Semester Capstone Examination & Viva Voce Defense
- **Primary Toolchain:** Python 3.14, NumPy 2.x, SciPy Signal, Matplotlib, PhysioNet WFDB

> **Core Research Mission:** Design and validate a robust, mathematically rigorous, and ultra-fast real-time DSP pipeline that achieves $>99\%$ QRS detection sensitivity across clean, noisy, and pathologically arrhythmic electrocardiograms.

---

## Slide 2: Clinical Motivation & Biomedical Significance

### Why Automated ECG Signal Processing Matters

- **Global Cardiovascular Burden:**
  - Cardiovascular Diseases (CVDs) cause $17.9\text{ million}$ deaths annually (~$32\%$ of global mortality).
  - Continuous holter monitoring and ICU telemetry provide early warning for fatal arrhythmias (Ventricular Fibrillation, Asystole).
- **The QRS Complex as the Clinical Gold Standard:**
  - Corresponds to ventricular depolarization (mechanical pumping action of the heart).
  - High-energy, rapid electrical dipole ($5\text{ to }15\text{ Hz}$ spectral dominance).
  - Localization of the R-peak is the mandatory foundation for:
    1. Real-time Heart Rate (HR) tracking;
    2. Arrhythmia classification (AAMI EC57 standards);
    3. Heart Rate Variability (HRV) analysis of autonomic nervous system health.
- **Translational Objective:** Transition from high-power hospital racks to low-power wearable edge devices (smartwatches, patches) requiring computationally lightweight deterministic DSP algorithms.

---

## Slide 3: Signal Degradation & Biomedical Noise Sources

### Real-World Noise Modalities in Ambulatory Electrocardiograms

```mermaid
flowchart TD
    Bio["Raw Cardiac Bio-potential"] --> Contam["Ambulatory Contaminants"]
    Contam --> BW["Baseline Wander (<0.5 Hz)\nRespiration & Skin Impedance"]
    Contam --> PLI["Powerline Mains (50/60 Hz)\nCapacitive/Inductive Coupling"]
    Contam --> EMG["Electromyogram (EMG >100 Hz)\nMuscle Tremor & Motion Artifact"]
    Contam --> ARR["Pathological Morphologies\nPVCs, Bundle Branch Blocks, Peaked T-waves"]
```

- **Baseline Drift ($< 0.5\text{ Hz}$):** Thoracic expansion and perspiration modulate the half-cell electrode potential, causing large amplitude drifts that saturate ADCs.
- **Powerline Interference ($50/60\text{ Hz}$):** Utility line displacement currents couple into high-impedance lead cables.
- **Electromyographic (EMG) Noise ($> 100\text{ Hz}$):** Random motor unit potentials from patient shivering or movement introduce high-frequency stochastic jitter.
- **Morphological Traps:** Giant T-waves mimic QRS slew rates; broad ectopic PVCs lack sharp high-frequency edges.

---

## Slide 4: End-to-End Pipeline Architecture

### Modular DSP Processing Flow

```mermaid
flowchart LR
    A["Raw ECG (360 Hz)"] --> B["Phase 2: Noise Characterization\nFFT & Welch PSD"]
    B --> C["Phase 3: Digital Filtering\nHP 0.5Hz → Notch 50Hz → LP 150Hz"]
    C --> D["Phase 4: Pan-Tompkins Engine\nBandpass → Diff → Square → MWI → Thresh"]
    D --> E["Phase 5: Feature Extraction\nHR, SDNN, RMSSD, Welch LF/HF"]
    D --> F["Phase 6: Clinical Validation\nAAMI EC57 ±150ms Bipartite Matching"]
```

- **Guiding Architecture Principles:**
  - **Zero Group Delay:** Preserves exact temporal fiducial landmarks.
  - **Vectorized Array Primitives:** Avoids iterative Python loops; sub-millisecond execution.
  - **Adaptive Thresholding:** Dynamic dual-threshold state machine tracking time-varying noise levels.
  - **Full Standards Compliance:** Evaluated against PhysioNet ground truth using ANSI/AAMI EC57 metrics.

---

## Slide 5: Dataset & Ground-Truth Standards

### PhysioNet MIT-BIH Arrhythmia Database

- **Database Topology:** 48 two-channel ambulatory ECG recordings, 30 minutes each ($650,000$ samples/record).
- **Acquisition Hardware:**
  - Sampling Frequency: $f_s = 360\text{ Hz}$
  - Resolution: 11-bit uniform ADC across $\pm 5\text{ mV}$ dynamic range ($200\text{ ADC units/mV}$).
  - Lead Configuration: Channel 0 is Modified Lead II (MLII); Channel 1 is Precordial $V_1, V_2, V_4,\text{ or }V_5$.
- **ANSI/AAMI EC57 Annotation Standard:**
  - Strips non-beat rhythm markers (`+`, `~`, `|`).
  - Evaluates true physiological heartbeats across 5 clinical classes: Normal (`N`), Supraventricular Ectopic (`S`), Ventricular Ectopic (`V`), Fusion (`F`), and Paced/Unknown (`Q`).
- **Benchmark Corpus:** Evaluated across 6 diverse records ($13,537$ total annotated beats), testing clean baselines (100, 101, 103), severe noise (105), severe baseline drift (108), and complex ventricular ectopy (203).

![Raw ECG Channels](../results/figures/01_raw_ecg_channels.png)

---

## Slide 6: Spectral Noise Characterization: FFT & Welch PSD

### Frequency Decomposition of Clean vs. Noisy ECG Signals

- **FFT Magnitude Spectrum:**
  $$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j \frac{2\pi}{N} k n}$$
  - Highlights exact discrete frequency peaks (e.g., $50\text{ Hz}$ mains spikes).
  - Limitation: High variance periodogram estimate; does not reduce as $N \to \infty$.
- **Welch’s Averaged Modified Periodogram:**
  - Partitions signal into $50\%$ overlapping segments with Hann windowing to suppress spectral leakage.
  - Yields smooth, low-variance Power Spectral Density (PSD in $\text{mV}^2/\text{Hz}$).
- **Key Empirical Observations:**
  - Clean Record 100: Pristine cardiac energy peak at $10\text{ to }12\text{ Hz}$; noise floor below $-40\text{ dB}$.
  - Noisy Record 105: Severe broadband EMG elevation ($+18\text{ dB}$ across $30\text{–}150\text{ Hz}$) and prominent $60\text{ Hz}$ line spikes.

![PSD Clean vs Noisy](../results/figures/02_psd_clean_vs_noisy.png)

---

## Slide 7: Filter Design: Linear & Zero-Phase Filtering

### Multi-Stage Cascaded Conditioning Chain

1. **High-Pass Filter (0.5 Hz, 4th-Order Butterworth):**
   - Eliminates respiratory baseline wander without distorting ST-segment morphology.
   - Bilinear transform with frequency pre-warping: $\Omega = 2 f_s \tan(\pi f_c / f_s)$.
2. **Narrow-Band IIR Notch Filter (50 Hz, Quality Factor $Q = 30$):**
   $$H_{\text{notch}}(z) = b_0 \frac{1 - 2\cos(\omega_0)z^{-1} + z^{-2}}{1 - 2 r \cos(\omega_0)z^{-1} + r^2 z^{-2}}, \quad B_{3\text{dB}} = \frac{f_0}{Q} = 1.67\text{ Hz}$$
   - Deep null ($>45\text{ dB}$ suppression) rejecting powerline hum with negligible cardiac energy loss.
3. **Low-Pass Filter (150 Hz, 4th-Order Butterworth):**
   - Enforces AHA/ACC diagnostic high-frequency limit, rejecting radio-frequency and electromyographic spikes.

![Filter Responses](../results/figures/03_filter_frequency_responses.png)

---

## Slide 8: Why Zero-Phase? Group Delay & Peak Distortion

### Bidirectional Filtering (`filtfilt`) vs. Causal Real-Time Filtering

- **The Problem with Causal IIR Filters:**
  - Non-linear phase response $\phi(\omega)$ causes frequency-dependent group delay:
    $$\tau_g(\omega) = -\frac{d\phi(\omega)}{d\omega} \ne 0$$
  - Shifts localized R-peaks by $10\text{ to }15\text{ samples}$ ($30\text{–}45\text{ ms}$), skewing peak-to-peak RR intervals and corrupting downstream HRV metrics.
- **The Zero-Phase Forward-Backward Paradigm:**
  $$\text{Pass 1: } v[n] = x[n] * h[n] \quad \longrightarrow \quad \text{Pass 2: } y[n] = v[-n] * h[n]$$
  $$Y(e^{j\omega}) = X(e^{j\omega}) \cdot |H(e^{j\omega})|^2$$
  - Effective Phase: $\angle H_{\text{zp}}(e^{j\omega}) \equiv 0\text{ rad}$.
  - Effective Group Delay: $\tau_g \equiv 0\text{ samples}$.
  - Effective Filter Order: Doubles ($N_{\text{eff}} = 2N = 8$), sharpening transition rolloff.

![Raw vs Filtered](../results/figures/03_raw_vs_filtered_ecg.png)

---

## Slide 9: Pan-Tompkins Algorithm: Mathematical Formulation

### The Five Transformational Stages (Pan & Tompkins, IEEE TBME 1985)

1. **Bandpass Filter ($5\text{–}15\text{ Hz}$):** Isolates QRS spectral energy while suppressing P-waves, T-waves, and 50 Hz hum.
2. **Five-Point Derivative Filter:**
   $$y[n] = \frac{1}{8} \left[ -x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2] \right]$$
   - Approximates an ideal differentiator $H(e^{j\omega}) \approx j\omega$ with exact linear phase; isolates high slopes.
3. **Non-Linear Squaring Function:**
   $$y[n] = (x[n])^2$$
   - Forces unipolarity; non-linearly boosts large QRS slopes while heavily attenuating small residual ripples ($x^2 \ll x$ when $x < 1$).
4. **Moving Window Integration (MWI):**
   $$y[n] = \frac{1}{N} \sum_{i=0}^{N-1} x[n-i], \quad N = \lfloor 0.150 \times f_s \rfloor = 54\text{ samples}$$
   - Captures aggregate waveform energy envelope (slope + duration).

---

## Slide 10: Pan-Tompkins Stages: Visualized Step-by-Step

### Morphological Waveform Evolution Across Stages

![Pan-Tompkins Stages](../results/figures/04_pan_tompkins_stages.png)

- **Stage Analysis:**
  - Raw ECG $\to$ Bandpass: Cleans baseline wander and high-frequency baseline ripple.
  - Bandpass $\to$ Derivative: Biphasic anti-symmetric excursion centered on R-peak.
  - Derivative $\to$ Squared: Unipolar positive energy spikes at Q-R and R-S edges.
  - Squared $\to$ Integrated: Single smooth, bell-shaped energy pulse spanning the entire QRS duration.

---

## Slide 11: Adaptive Dual-Thresholding & Search-Back Logic

### Dynamic Decision Engine for Time-Varying ECG Amplitudes

- **State-Variable Tracking:**
  - $SPKI$: Running Signal Peak Level estimate (true QRS energy)
  - $NPKI$: Running Noise Peak Level estimate (background noise energy)
- **Dynamic Decision Thresholds:**
  $$THR\_SIG = NPKI + 0.25 \, (SPKI - NPKI)$$
  $$THR\_NOISE = 0.5 \times THR\_SIG$$
- **Recursive Exponential Smoothing ($\alpha = 0.125$):**
  - If $PEAK > THR\_SIG$: $SPKI \leftarrow 0.125 \, PEAK + 0.875 \, SPKI$
  - If $PEAK \le THR\_SIG$: $NPKI \leftarrow 0.125 \, PEAK + 0.875 \, NPKI$
- **Physiological Blanking & Search-Back:**
  - Refractory period: $200\text{ ms}$ ($72$ samples) post-detection blanking prevents double-triggering.
  - Search-back trigger: If $\Delta t > 1.66 \times \overline{RR}_8$, retroactively rescan interval with $THR\_NOISE$.
  - Fiducial Refinement: Search $\pm 75\text{ ms}$ window in bandpass signal to lock onto true R-crest.

---

## Slide 12: Quantitative Validation Results (Multi-Record Benchmark)

### ANSI/AAMI EC57 Evaluation Across 13,537 Clinical Beats

| Record ID | Condition / Morphology | Ref Beats | Detected | Sensitivity ($Se$) | Precision ($PPV$) | $F_1$-Score | Error ($DER$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **100** | Normal sinus baseline (clean) | 2,273 | 2,273 | **100.00%** | **100.00%** | **100.00%** | **0.00%** |
| **101** | Normal rhythm, slight wander | 1,865 | 1,868 | **99.95%** | **99.79%** | **99.87%** | 0.27% |
| **103** | High SNR normal baseline | 2,084 | 2,083 | **99.95%** | **100.00%** | **99.98%** | 0.05% |
| **105** | High motion artifact & EMG | 2,572 | 2,620 | **99.84%** | **98.02%** | **98.92%** | 2.18% |
| **108** | Severe wander & peaked T-waves | 1,763 | 1,915 | **99.77%** | **91.85%** | **95.65%** | 9.08% |
| **203** | Complex ventricular ectopy (PVCs) | 2,980 | 2,929 | **97.65%** | **99.35%** | **98.49%** | 2.99% |
| **OVERALL** | **Aggregate Across 6 Records** | **13,537** | **13,688** | **99.41%** | **98.31%** | **98.86%** | **2.30%** |

![Benchmark Chart](../results/figures/06_validation_benchmark_chart.png)

---

## Slide 13: Failure Case Deep-Dive 1: Record 108 (T-Wave Oversensing)

### Diagnostic Analysis of False Positives ($FP = 156$, $PPV = 91.85\%$)

- **Clinical Morphology:** Low-amplitude biphasic R-waves followed by abnormally tall, sharp, inverted T-waves.
- **Root Cause Mechanism:**
  - Derivative filter is a slope detector ($H(e^{j\omega}) \propto \omega$).
  - The steep descending limb of the giant T-wave exhibited slope energy comparable to the attenuated QRS complex.
  - The squared signal elevated T-waves above $THR\_SIG$, creating false-positive double triggers.
- **DSP Countermeasures & Lessons Learned:**
  - Setting MWI window strictly to $150\text{ ms}$ prevented QRS and T-wave energy pulses from coalescing into a single plateau.
  - Post-MWI fiducial search restricted to $\pm 75\text{ ms}$ localized detections to the true electrical peak.
  - Dynamic $SPKI / NPKI$ tracking enabled the detector to maintain $95.65\% \text{ } F_1$-score despite extreme morphological corruption.

---

## Slide 14: Failure Case Deep-Dive 2: Record 203 (Ectopic PVC Morphology)

### Diagnostic Analysis of Missed Beats ($FN = 70$, $Se = 97.65\%$)

- **Clinical Morphology:** Frequent polymorphic Premature Ventricular Contractions (PVCs), bigeminy, and non-sustained Ventricular Tachycardia (VT).
- **Electrophysiological Root Cause:**
  - PVCs originate in ventricular myocardium and conduct via slow cell-to-cell spread ($QRS > 140\text{–}180\text{ ms}$).
  - Ectopic beats lack the high slew-rate of normal His-Purkinje depolarization.
  - The 5-point differentiator output was blunted; the energy failed to cross $THR\_SIG$ set by preceding sharp sinus beats.
- **Search-Back Recovery:**
  - Physiological search-back successfully rescued $42$ low-amplitude PVCs.
  - However, during rapid ventricular tachycardia runs, the short RR interval ($\Delta t < 1.66 \times \overline{RR}_8$) suppressed search-back activation.
- **Engineering Conclusion:** High specificity against T-waves creates an inherent physical trade-off against wide ectopic beats in classical linear filtering.

---

## Slide 15: Ablation Studies: Integration Window & Filter Necessity

### Hyperparameter Sensitivity Sweeps Across 13,537 Beats

- **Moving Window Integration (MWI) Sweep:**
  - $50\text{ ms}$ ($18$ samples): $F_1 = 96.74\%$ — High false positives due to polyphasic QRS splitting.
  - $100\text{ ms}$ ($36$ samples): $F_1 = 98.33\%$ — Improved but susceptible to notch noise.
  - **$150\text{ ms}$ ($54$ samples): $F_1 = 98.86\%$ — Global optimal balance.**
  - $200\text{ ms}$ ($72$ samples): $F_1 = 97.47\%$ — T-wave merging increases false positives.
  - $250\text{ ms}$ ($90$ samples): $F_1 = 94.80\%$ — Severe QRS-T blurring and loss of rapid ectopic beats.
- **Filtering Topology Necessity:**
  - Without High-Pass ($0.5\text{ Hz}$): Baseline drift causes false-negative dropout during respiratory excursions ($DER \uparrow 4.8\times$).
  - Without Notch ($50\text{ Hz}$): Powerline hum creates continuous ripples in differentiator output, elevating $NPKI$.
  - Causal vs. Zero-Phase: Causal IIR introduces $38.8\text{ ms}$ mean peak jitter, corrupting RMSSD by $+12.4\%$.

---

## Slide 16: Feature Extraction: HR & Time/Frequency HRV

### Autonomic Nervous System Biomarker Computation

- **Extracted Metrics for Record 100 (30-Minute Recording):**
  - Mean Heart Rate: **$75.39\text{ BPM}$** (normal resting sinus rhythm)
  - Time-Domain Variability:
    - $\text{SDNN} = \mathbf{46.85\text{ ms}}$ (overall autonomic circadian variability)
    - $\text{RMSSD} = \mathbf{27.05\text{ ms}}$ (parasympathetic vagal tone index)
    - $\text{pNN50} = \mathbf{4.85\%}$ (percentage of successive RR differences $>50\text{ ms}$)
- **The Non-Equidistant Sampling Solution:**
  - Raw RR intervals form an irregular event series.
  - Resampled onto a uniform $4.0\text{ Hz}$ time grid via **cubic spline interpolation**.
  - Welch PSD integrates Task Force clinical bands: $\text{VLF} (<0.04\text{ Hz})$, $\text{LF} (0.04\text{–}0.15\text{ Hz})$, $\text{HF} (0.15\text{–}0.40\text{ Hz})$.
  - Sympathovagal Balance: $\mathbf{LF/HF = 1.56}$ (balanced sympathetic/parasympathetic tone).

![HRV PSD Spectrum](../results/figures/05_hrv_psd.png)

---

## Slide 17: Non-Linear HRV: Poincaré Phase-Space Analysis

### Phase-Space Attractor Geometry ($RR_n \text{ vs. } RR_{n+1}$)

- **Poincaré Plot Principles:**
  - Embeds the non-linear dynamics of heart rate into a return map ($RR_n$ vs $RR_{n+1}$).
  - An ellipse is fitted to the scatter plot along the line of identity ($RR_{n+1} = RR_n$).
- **Mathematical Descriptors:**
  - **$SD_1$ (Minor Semi-Axis):**
    $$SD_1 = \sqrt{\frac{1}{2}\text{Var}(RR_n - RR_{n+1})} = \frac{1}{\sqrt{2}} \text{RMSSD}$$
    Measures short-term, beat-to-beat variability driven by respiratory sinus arrhythmia.
  - **$SD_2$ (Major Semi-Axis):**
    $$SD_2 = \sqrt{2\,\text{SDNN}^2 - \frac{1}{2}\text{RMSSD}^2}$$
    Measures continuous long-term autonomic variability across the entire recording.
  - **Axis Ratio ($SD_1 / SD_2$):** Non-linear indicator of parasympathetic vs. sympathetic dominance.

---

## Slide 18: Summary of Contributions & Engineering Highlights

### Key Academic & Technical Achievements

1. **High-Performance Detection Benchmark:**
   - Validated on $13,537$ clinical heartbeats: **$99.41\% \text{ } Se$**, **$98.31\% \text{ } PPV$**, **$98.86\% \text{ } F_1$-Score**.
   - Perfect $100.00\%$ detection score on benchmark Record 100.
2. **Computational Velocity via Vectorization:**
   - Sliced NumPy array operations accelerated the differentiator by **$77.4\times$** (from $4.52\text{ s}$ down to $58.4\text{ ms}$ per 30-minute record).
   - Real-time factor of $>30,000\times$, ideal for edge deployment.
3. **Rigorous Signal Conditioning:**
   - Zero-phase Butterworth and notch filtering guarantees zero group delay and sub-millisecond fiducial peak alignment.
4. **Comprehensive Feature Suite:**
   - Integrated time-domain, cubic-spline resampled spectral, and non-linear Poincaré autonomic biomarkers.

---

## Slide 19: Translational Roadmap & Future Directions

### Moving from Desktop Research to Embedded Clinical Devices

- **1. Microcontroller Deployment (ARM Cortex-M4/M33):**
  - Translate the Python NumPy pipeline into fixed-point Q15/Q31 C/C++ primitives using ARM CMSIS-DSP.
  - Target: Ultra-low-power wearables ($<150\text{ }\mu\text{W}$ consumption, multi-week battery life).
- **2. Dynamic Wavelet Decomposition for Arrhythmias:**
  - Incorporate Stationary Discrete Wavelet Transforms (DWT, Symlet-4) to dynamically adapt decomposition levels to broad PVCs (Record 203).
- **3. Hybrid DSP + Lightweight Edge Neural Networks:**
  - Utilize Pan-Tompkins as an ultra-fast candidate generator.
  - Feed candidate beat windows into a tiny 1D-CNN ($<10\text{k}$ parameters) for AAMI arrhythmia classification (PVC vs. Normal vs. Paced).

---

## Slide 20: Defense Q&A & Acknowledgments

### Thank You!

- **Questions & Discussion Invited:**
  - Mathematical derivations & Z-transform filter topologies
  - SciPy zero-phase boundary mechanics & vectorization strategies
  - Failure mode diagnostics on Records 108 & 203
  - Clinical translation and embedded edge implementation
- **Project Artifacts:**
  - Complete Codebase: `src/` (filters, Pan-Tompkins, feature extraction, validation)
  - Unit & Integration Test Suites: `tests/test_pipeline.py`
  - Engineering Lab Log: `LAB_JOURNAL.md` (8-week technical progression)
  - Full Semester Report: `docs/FINAL_PROJECT_REPORT.md`

---
*End of Presentation Slide Deck*
