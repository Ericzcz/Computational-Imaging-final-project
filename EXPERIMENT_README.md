# TV-Regularized Image Deblurring Using ADMM

This code completes the first research question from the proposal:

> With the blur kernel known exactly, how does additive Gaussian noise affect
> inverse filtering, Wiener--Hunt filtering, and TV-ADMM?

## 1. Prepare the images

Download BSDS500 and copy at least six PNG/JPEG images into `data/images/`.
The script searches that directory recursively. It converts every image to
grayscale, center-crops it, and resizes it to 256 by 256 pixels.

The first image (alphabetically) is used only to choose parameters. Up to five
remaining images are used as the test set. For the final report, use more test
images by increasing `--max-test-images`.

## 2. Install and test

```bash
python3 -m pip install -r requirements.txt
python3 experiment.py --smoke-test
```

If `data/images/` is empty, smoke-test mode generates synthetic images. Those
outputs verify the implementation but should not be reported as BSDS500 results.

## 3. Run the first experiment

```bash
python3 experiment.py
```

To test more images:

```bash
python3 experiment.py --max-test-images 20 --max-iter 100
```

## 4. Experimental settings

- Gaussian blur: 9 by 9, blur standard deviation 1.6
- Noise standard deviations: 0.01, 0.03, and 0.05
- Methods: stabilized inverse filter, Wiener--Hunt filter, and TV-ADMM
- Metrics: PSNR, SSIM, and reconstruction runtime
- TV-ADMM penalty parameter: rho = 0.1
- ADMM iterations: 100

For each noise level, the inverse-filter parameter, Wiener--Hunt balance, and
TV regularization weight are selected on the validation image. Test images are
not used during parameter selection.

## 5. Generated results

The `results/noise_experiment/` directory contains:

- `metrics.csv`: one row per image, noise level, and method
- `summary.csv`: mean and standard deviation across test images
- `visual_comparison.png`: qualitative reconstruction comparison
- `noise_metric_curves.png`: mean PSNR and SSIM versus noise level
- `admm_convergence.png`: objective and residual curves
- `experiment_config.json`: settings and selected parameters

The implementation uses circular convolution for both data generation and
reconstruction. This keeps the forward model and optimization model consistent.

The noise experiment is intentionally a preliminary Project Update experiment.
It now includes the degraded input as a numerical baseline and uses wider,
predefined parameter grids for a fairer comparison. It still uses one validation
image, a fixed ADMM penalty and iteration count, and a Gaussian-only noise sweep.
Those limitations are documented as planned improvements rather than presented
as the final experimental design.

## 6. Run the blur-kernel mismatch experiment

The second experiment fixes the noise standard deviation at 0.03 and changes
the kernel used during reconstruction. It tests Gaussian-width error, motion-
length error, and motion-angle error. Parameters are selected with the correct
kernel and then held fixed across all mismatch levels.

```bash
python3 kernel_mismatch_experiment.py --max-test-images 20 --max-iter 100
```

Results are written to `results/kernel_mismatch_experiment/`. The directory
contains complete and summary CSV files, three metric figures, and three visual
comparison figures. `summary.csv` also reports the PSNR and SSIM change relative
to reconstruction with the correct kernel.

## 7. Planned improvements after the Project Update

- Use about five validation images and select parameters by mean validation PSNR.
- Add the full noise-level sweep for motion blur.
- Select ADMM rho using convergence residuals and runtime.
- Add common primal and dual stopping tolerances instead of only a fixed limit.
- Show two or three representative images with marked, matched crops.
- Add intensity profiles through selected edges and textures.
- Optionally evaluate LPIPS if time and compute resources permit.
