
# Rationale Behind the Adaptive τ Design

## 1. Problem Setting

In the UMich CASI pipeline, a mixture is represented as a weighted combination of its component molecules.  
The parameter **τ** controls how molecule intensities are converted into weights:

- **Small τ** → sharp weights (one molecule dominates)
- **Large τ** → smooth weights (weights spread more evenly)

The original CASI approach uses a **single fixed τ** for all mixtures, implicitly assuming all mixtures should be treated the same.

Our hypothesis was that **different mixtures require different levels of smoothing**.

---

## 2. Desired Behavior of Adaptive τ

We identified two qualitatively different mixture regimes based on intensity structure:

### Dominant mixtures
- One molecule is noticeably stronger than the others.
- Desired behavior: **sharper weights** so the dominant molecule has more influence.

### Balanced mixtures
- Molecules have similar intensities.
- Desired behavior: **smoother weights** so no molecule artificially dominates.

This leads to a clear design goal:

| Mixture type | Desired τ behavior | Effect on weights |
|-------------|-------------------|------------------|
| Dominant    | Smaller τ         | Sharper weights |
| Balanced    | Larger τ          | Smoother weights |

---

## 3. Evaluation Metric

Since τ only affects the **weighting step**, we evaluated it directly using:

**Top weight** = largest weight assigned to any molecule in a mixture.

This metric is:
- Simple and interpretable
- Directly tied to the role of τ
- Suitable for comparing fixed and adaptive τ approaches

---

## 4. Lessons from Initial Attempts

### Fixed τ baseline
- A single τ value is a compromise.
- It cannot simultaneously handle dominant and balanced mixtures optimally.

### Single-formula adaptive τ (e.g., inverse-IQR)
- Improved dominant mixtures
- Over-sharpened balanced mixtures
- Or collapsed back to a constant τ when tuned

**Key insight:**  
Intensity dispersion (IQR) varies too little across mixtures to support a single global adaptive rule.

---

## 5. Why a Single Adaptive Formula Failed

Multiple formulations were tested (inverse IQR, scaled inverse IQR, dominance-ratio-based).
All failed for the same reason:

> A single smooth function of intensities cannot simultaneously improve both dominant and balanced regimes.

The data naturally separates into two qualitatively different cases.

---

## 6. Critical Realization

Instead of forcing one formula to work everywhere, we recognized:

> **Dominant and balanced mixtures are fundamentally different problems.**

This motivated a regime-specific design.

---

## 7. Final Design: Two-Regime Adaptive τ

We define τ using a **piecewise rule**:

### Dominant mixtures
- Use an **inverse-IQR-based τ**
- Produces sharper weights
- Correctly emphasizes the strongest molecule

### Balanced mixtures
- Use a **smoothing τ based on dominance ratio**
- If the top two intensities are very similar, τ is increased
- Prevents artificial dominance

The mixture type is determined **only from intensities**, before any weighting.

---

## 8. Why This Is Methodologically Sound

This approach is not “cheating” because:
- Regime assignment uses only input intensities
- τ is computed deterministically
- The same rules apply to all mixtures
- No downstream labels or targets are used

It encodes domain structure rather than overfitting.

---

## 9. Outcome

The final adaptive τ satisfies the original goals:

- **Balanced mixtures:** adaptive top weight < fixed τ top weight
- **Dominant mixtures:** adaptive top weight > fixed τ top weight

This indicates that adaptive τ better matches mixture structure than any single fixed τ.

---

## 10. Core Takeaway

A single global τ is insufficient because mixtures differ qualitatively in their intensity structure.  
By explicitly separating dominant and balanced regimes and applying appropriate smoothing in each case, the adaptive τ produces more meaningful mixture weights.
