# Experimental Scope for the Project Update

## What has been completed

The preliminary noise experiment uses a 9 by 9 Gaussian blur kernel with blur
standard deviation 1.6 and additive Gaussian noise with standard deviations
0.01, 0.03, and 0.05. It compares the degraded input, stabilized inverse
filtering, Wiener-Hunt deconvolution, and TV-ADMM using PSNR, SSIM, and runtime.
The implementation also records the TV-ADMM objective, primal residual, and dual
residual. A fixed random seed makes the observations reproducible.

The preliminary kernel-mismatch experiment fixes the noise standard deviation at
0.03. It tests Gaussian-width errors of -20%, -10%, 0%, +10%, and +20%, as well
as errors in motion-blur length and angle. Reconstruction parameters are selected
with the correct kernel and held fixed when the assumed kernel changes. The same
images and noise realizations are used for all methods.

## Why these results are preliminary

The current experiments use one validation image, fixed ADMM rho = 0.1, and a
fixed limit of 100 ADMM iterations. The noise-level sweep currently uses Gaussian
blur only. The figures provide an overall visual comparison but do not yet contain
matched image crops or intensity profiles. These choices keep the Project Update
experiment manageable while still providing a complete implementation and an
initial comparison.

## Planned improvements

1. Use a multi-image validation set and tune each method using mean validation PSNR.
2. Repeat the complete noise sweep with linear motion blur.
3. Tune rho using convergence residuals and runtime, then add common stopping tolerances.
4. Report results on 20 to 30 test images with confidence intervals or standard deviations.
5. Add two or three representative images with marked, matched crops and identical display ranges.
6. Plot intensity profiles through the same edges to measure sharpness, ringing, and plateaus.
7. Consider LPIPS as an optional perceptual metric if resources permit.

## Suggested wording for the update report

These experiments are intended as an initial validation of the implementation.
They establish the expected noise-sensitivity trends, verify ADMM convergence,
and provide preliminary evidence about sensitivity to blur-kernel mismatch. The
next stage will strengthen parameter selection, extend the noise study to motion
blur, and add more detailed visual analysis.
