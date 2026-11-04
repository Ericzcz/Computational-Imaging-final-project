#!/usr/bin/env python3
"""Preliminary noise-robust image deblurring experiment for the project update.

Research question:
    With the blur kernel known exactly, how does additive Gaussian noise
    affect inverse filtering, Wiener--Hunt filtering, and TV-ADMM?

Put BSDS500 images (or any PNG/JPEG images) under data/images and run:
    python3 experiment.py

The script uses circular convolution consistently in data generation and all
three reconstruction methods. The first image is used only for parameter
selection; the remaining images are used for the reported test results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate


EPS = 1e-12


@dataclass(frozen=True)
class Config:
    image_size: int = 256
    max_test_images: int = 5
    noise_levels: tuple[float, ...] = (0.01, 0.03, 0.05)
    kernel_size: int = 9
    blur_sigma: float = 1.6
    rho: float = 0.1
    max_iter: int = 100
    gamma_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    balance_grid: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
    lambda_grid: tuple[float, ...] = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
    seed: int = 2026


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Return a normalized, centered 2-D Gaussian point-spread function."""
    if size % 2 == 0:
        raise ValueError("kernel size must be odd")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def motion_kernel(size: int, length: int, angle_deg: float) -> np.ndarray:
    """Optional helper for the later blur-kernel mismatch experiment."""
    if size % 2 == 0 or length > size:
        raise ValueError("size must be odd and length must not exceed size")
    kernel = np.zeros((size, size), dtype=np.float64)
    c = size // 2
    start = c - length // 2
    kernel[c, start : start + length] = 1.0
    kernel = rotate(kernel, angle_deg, reshape=False, order=1, mode="constant")
    kernel = np.maximum(kernel, 0.0)
    return kernel / kernel.sum()


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Embed a centered PSF and convert it to its FFT transfer function."""
    padded = np.zeros(shape, dtype=np.float64)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def circular_convolve(image: np.ndarray, h_fft: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(h_fft * np.fft.fft2(image)).real


def gradient(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic forward differences: horizontal and vertical."""
    return np.roll(x, -1, axis=1) - x, np.roll(x, -1, axis=0) - x


def divergence(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Adjoint of gradient (D^T), using the same periodic boundary."""
    return np.roll(px, 1, axis=1) - px + np.roll(py, 1, axis=0) - py


def stabilized_inverse(y: np.ndarray, h_fft: np.ndarray, gamma: float) -> np.ndarray:
    y_fft = np.fft.fft2(y)
    x = np.fft.ifft2(np.conj(h_fft) * y_fft / (np.abs(h_fft) ** 2 + gamma)).real
    return np.clip(x, 0.0, 1.0)


def wiener_hunt(y: np.ndarray, h_fft: np.ndarray, balance: float) -> np.ndarray:
    """Wiener--Hunt estimate with a Laplacian smoothness prior."""
    rows, cols = y.shape
    wy = 2.0 * np.pi * np.fft.fftfreq(rows)
    wx = 2.0 * np.pi * np.fft.fftfreq(cols)
    lap = 4.0 - 2.0 * np.cos(wy[:, None]) - 2.0 * np.cos(wx[None, :])
    denominator = np.abs(h_fft) ** 2 + balance * np.abs(lap) ** 2 + EPS
    x = np.fft.ifft2(np.conj(h_fft) * np.fft.fft2(y) / denominator).real
    return np.clip(x, 0.0, 1.0)


def tv_admm(
    y: np.ndarray,
    h_fft: np.ndarray,
    lam: float,
    rho: float,
    max_iter: int,
    record_history: bool = False,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Solve 0.5||Hx-y||_2^2 + lam*||Dx||_2,1 with ADMM."""
    rows, cols = y.shape
    wy = 2.0 * np.pi * np.fft.fftfreq(rows)
    wx = 2.0 * np.pi * np.fft.fftfreq(cols)
    d2 = (2.0 - 2.0 * np.cos(wx))[None, :] + (2.0 - 2.0 * np.cos(wy))[:, None]
    denominator = np.abs(h_fft) ** 2 + rho * d2 + EPS
    hty_fft = np.conj(h_fft) * np.fft.fft2(y)

    x = y.copy()
    zx = np.zeros_like(y)
    zy = np.zeros_like(y)
    ux = np.zeros_like(y)
    uy = np.zeros_like(y)
    history: dict[str, list[float]] = {"objective": [], "primal": [], "dual": []}

    for _ in range(max_iter):
        # x update: solve the quadratic subproblem exactly in the FFT domain.
        rhs = hty_fft + rho * np.fft.fft2(divergence(zx - ux, zy - uy))
        x = np.fft.ifft2(rhs / denominator).real

        # z update: isotropic vector soft-thresholding.
        gx, gy = gradient(x)
        vx, vy = gx + ux, gy + uy
        magnitude = np.sqrt(vx**2 + vy**2)
        shrink = np.maximum(0.0, 1.0 - (lam / rho) / np.maximum(magnitude, EPS))
        old_zx, old_zy = zx, zy
        zx, zy = shrink * vx, shrink * vy

        # Scaled dual update.
        ux += gx - zx
        uy += gy - zy

        if record_history:
            residual_x, residual_y = gx - zx, gy - zy
            primal = math.sqrt(float(np.sum(residual_x**2 + residual_y**2)))
            dual_array = divergence(zx - old_zx, zy - old_zy)
            dual = rho * float(np.linalg.norm(dual_array))
            data_term = 0.5 * float(np.sum((circular_convolve(x, h_fft) - y) ** 2))
            tv_term = lam * float(np.sum(np.sqrt(gx**2 + gy**2)))
            history["objective"].append(data_term + tv_term)
            history["primal"].append(primal)
            history["dual"].append(dual)

    return np.clip(x, 0.0, 1.0), history


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Standard local SSIM with an 11x11 Gaussian window and data range 1."""
    sigma = 1.5
    truncate = 3.5  # gives an approximately 11x11 support
    mu_x = gaussian_filter(reference, sigma=sigma, truncate=truncate, mode="reflect")
    mu_y = gaussian_filter(estimate, sigma=sigma, truncate=truncate, mode="reflect")
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    var_x = gaussian_filter(reference**2, sigma, truncate=truncate, mode="reflect") - mu_x2
    var_y = gaussian_filter(estimate**2, sigma, truncate=truncate, mode="reflect") - mu_y2
    cov_xy = gaussian_filter(reference * estimate, sigma, truncate=truncate, mode="reflect") - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * mu_xy + c1) * (2.0 * cov_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (var_x + var_y + c2) + EPS
    )
    return float(np.mean(score))


def preprocess_image(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float64) / 255.0


def synthetic_images(size: int) -> list[tuple[str, np.ndarray]]:
    """Generate deterministic images only for a code smoke test."""
    yy, xx = np.mgrid[0:size, 0:size] / float(size)
    images: list[tuple[str, np.ndarray]] = []
    for index in range(6):
        base = 0.15 + 0.35 * xx + 0.15 * np.sin((index + 2) * np.pi * yy)
        circle = ((xx - 0.30 - 0.04 * index) ** 2 + (yy - 0.35) ** 2) < 0.10**2
        square = (np.abs(xx - 0.68) < 0.12) & (np.abs(yy - 0.67 + 0.02 * index) < 0.10)
        diagonal = np.abs(yy - (0.2 + 0.55 * xx)) < (0.012 + 0.002 * index)
        image = base + 0.45 * circle + 0.30 * square + 0.25 * diagonal
        images.append((f"synthetic_{index + 1}", np.clip(image, 0.0, 1.0)))
    return images


def load_images(data_dir: Path, config: Config) -> tuple[list[tuple[str, np.ndarray]], bool]:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in extensions)
    needed = config.max_test_images + 1
    if paths:
        if len(paths) < 2:
            raise ValueError("At least two images are required: one validation and one test image.")
        images = [(p.stem, preprocess_image(p, config.image_size)) for p in paths[:needed]]
        return images, False
    return synthetic_images(config.image_size)[:needed], True


def add_noise(clean_blurred: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = clean_blurred + rng.normal(0.0, sigma, size=clean_blurred.shape)
    return np.clip(noisy, 0.0, 1.0)


def select_parameters(
    reference: np.ndarray,
    observation: np.ndarray,
    h_fft: np.ndarray,
    config: Config,
) -> dict[str, float]:
    """Select each method's parameter by validation-image PSNR only."""
    best: dict[str, tuple[float, float]] = {}

    for gamma in config.gamma_grid:
        score = psnr(reference, stabilized_inverse(observation, h_fft, gamma))
        if "inverse" not in best or score > best["inverse"][0]:
            best["inverse"] = (score, gamma)

    for balance in config.balance_grid:
        score = psnr(reference, wiener_hunt(observation, h_fft, balance))
        if "wiener" not in best or score > best["wiener"][0]:
            best["wiener"] = (score, balance)

    for lam in config.lambda_grid:
        estimate, _ = tv_admm(observation, h_fft, lam, config.rho, config.max_iter)
        score = psnr(reference, estimate)
        if "tv_admm" not in best or score > best["tv_admm"][0]:
            best["tv_admm"] = (score, lam)

    return {method: value[1] for method, value in best.items()}


def restore_all(
    observation: np.ndarray,
    h_fft: np.ndarray,
    params: dict[str, float],
    config: Config,
    history: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, list[float]]]:
    estimates: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}

    start = time.perf_counter()
    estimates["Inverse"] = stabilized_inverse(observation, h_fft, params["inverse"])
    runtimes["Inverse"] = time.perf_counter() - start

    start = time.perf_counter()
    estimates["Wiener-Hunt"] = wiener_hunt(observation, h_fft, params["wiener"])
    runtimes["Wiener-Hunt"] = time.perf_counter() - start

    start = time.perf_counter()
    estimates["TV-ADMM"], admm_history = tv_admm(
        observation,
        h_fft,
        params["tv_admm"],
        config.rho,
        config.max_iter,
        record_history=history,
    )
    runtimes["TV-ADMM"] = time.perf_counter() - start
    return estimates, runtimes, admm_history


def write_metrics(rows: list[dict[str, object]], output_dir: Path) -> None:
    fieldnames = ["image", "noise_sigma", "method", "psnr_db", "ssim", "runtime_sec"]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (float(row["noise_sigma"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary_fields = [
        "noise_sigma",
        "method",
        "mean_psnr_db",
        "std_psnr_db",
        "mean_ssim",
        "std_ssim",
        "mean_runtime_sec",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for (noise, method), values in sorted(grouped.items()):
            psnrs = np.array([float(v["psnr_db"]) for v in values])
            ssims = np.array([float(v["ssim"]) for v in values])
            runtimes = np.array([float(v["runtime_sec"]) for v in values])
            writer.writerow(
                {
                    "noise_sigma": noise,
                    "method": method,
                    "mean_psnr_db": f"{psnrs.mean():.4f}",
                    "std_psnr_db": f"{psnrs.std(ddof=0):.4f}",
                    "mean_ssim": f"{ssims.mean():.5f}",
                    "std_ssim": f"{ssims.std(ddof=0):.5f}",
                    "mean_runtime_sec": f"{runtimes.mean():.6f}",
                }
            )


def plot_comparison(
    examples: dict[float, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    output_dir: Path,
) -> None:
    columns = ["Reference", "Blurred + noise", "Inverse", "Wiener-Hunt", "TV-ADMM"]
    fig, axes = plt.subplots(len(examples), len(columns), figsize=(14, 3 * len(examples)), squeeze=False)
    for row, (noise, (reference, observation, estimates)) in enumerate(sorted(examples.items())):
        images = [reference, observation, estimates["Inverse"], estimates["Wiener-Hunt"], estimates["TV-ADMM"]]
        for col, (title, image) in enumerate(zip(columns, images)):
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title)
        axes[row, 0].set_ylabel(f"noise $\\sigma$={noise:.2f}")
    fig.tight_layout()
    fig.savefig(output_dir / "visual_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curves(rows: list[dict[str, object]], output_dir: Path) -> None:
    methods = ["Blurred+Noise", "Inverse", "Wiener-Hunt", "TV-ADMM"]
    colors = {
        "Blurred+Noise": "#111827",
        "Inverse": "#6b7280",
        "Wiener-Hunt": "#2563eb",
        "TV-ADMM": "#dc2626",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        noises = sorted({float(r["noise_sigma"]) for r in method_rows})
        mean_psnr = [np.mean([float(r["psnr_db"]) for r in method_rows if float(r["noise_sigma"]) == n]) for n in noises]
        mean_ssim = [np.mean([float(r["ssim"]) for r in method_rows if float(r["noise_sigma"]) == n]) for n in noises]
        axes[0].plot(noises, mean_psnr, "o-", label=method, color=colors[method])
        axes[1].plot(noises, mean_ssim, "o-", label=method, color=colors[method])
    axes[0].set(xlabel="Noise standard deviation", ylabel="Mean PSNR (dB)", title="Noise sensitivity: PSNR")
    axes[1].set(xlabel="Noise standard deviation", ylabel="Mean SSIM", title="Noise sensitivity: SSIM")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "noise_metric_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history: dict[str, list[float]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    iterations = np.arange(1, len(history["objective"]) + 1)
    axes[0].plot(iterations, history["objective"], color="#111827")
    axes[0].set(xlabel="Iteration", ylabel="Objective", title="TV-ADMM objective")
    axes[1].semilogy(iterations, history["primal"], label="Primal residual")
    axes[1].semilogy(iterations, history["dual"], label="Dual residual")
    axes[1].set(xlabel="Iteration", ylabel="Residual norm", title="TV-ADMM convergence")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "admm_convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(data_dir: Path, output_dir: Path, config: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images, synthetic = load_images(data_dir, config)
    validation_name, validation = images[0]
    test_images = images[1:]
    if not test_images:
        raise RuntimeError("No test images were loaded.")

    kernel = gaussian_kernel(config.kernel_size, config.blur_sigma)
    h_fft = psf_to_otf(kernel, validation.shape)
    selected: dict[str, dict[str, float]] = {}

    # Tune separately at each noise level on one held-out validation image.
    validation_blur = circular_convolve(validation, h_fft)
    for noise_index, noise in enumerate(config.noise_levels):
        observation = add_noise(validation_blur, noise, config.seed + 1000 + noise_index)
        selected[f"{noise:.4f}"] = select_parameters(validation, observation, h_fft, config)

    metadata = {
        "research_question": "Effect of noise with a correctly known blur kernel",
        "data_source": "synthetic smoke-test images" if synthetic else str(data_dir.resolve()),
        "validation_image": validation_name,
        "test_images": [name for name, _ in test_images],
        "image_size": config.image_size,
        "gaussian_kernel_size": config.kernel_size,
        "gaussian_blur_sigma": config.blur_sigma,
        "noise_levels": list(config.noise_levels),
        "random_seed": config.seed,
        "rho": config.rho,
        "max_iter": config.max_iter,
        "stage": "preliminary project-update experiment",
        "known_limitations": [
            "one validation image rather than a multi-image validation set",
            "Gaussian-blur noise sweep only",
            "fixed ADMM rho and iteration count rather than residual-based stopping",
            "one qualitative example without matched crops or intensity profiles",
        ],
        "selected_parameters": selected,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    examples: dict[float, tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    convergence_history: dict[str, list[float]] | None = None
    middle_noise = config.noise_levels[len(config.noise_levels) // 2]

    for image_index, (name, reference) in enumerate(test_images):
        clean_blurred = circular_convolve(reference, h_fft)
        for noise_index, noise in enumerate(config.noise_levels):
            observation = add_noise(clean_blurred, noise, config.seed + image_index * 100 + noise_index)
            should_record = image_index == 0 and noise == middle_noise
            estimates, runtimes, history = restore_all(
                observation,
                h_fft,
                selected[f"{noise:.4f}"],
                config,
                history=should_record,
            )
            if image_index == 0:
                examples[noise] = (reference, observation, estimates)
            if should_record:
                convergence_history = history
            rows.append(
                {
                    "image": name,
                    "noise_sigma": noise,
                    "method": "Blurred+Noise",
                    "psnr_db": f"{psnr(reference, observation):.4f}",
                    "ssim": f"{ssim(reference, observation):.5f}",
                    "runtime_sec": "0.000000",
                }
            )
            for method, estimate in estimates.items():
                rows.append(
                    {
                        "image": name,
                        "noise_sigma": noise,
                        "method": method,
                        "psnr_db": f"{psnr(reference, estimate):.4f}",
                        "ssim": f"{ssim(reference, estimate):.5f}",
                        "runtime_sec": f"{runtimes[method]:.6f}",
                    }
                )

    write_metrics(rows, output_dir)
    plot_comparison(examples, output_dir)
    plot_metric_curves(rows, output_dir)
    if convergence_history is not None:
        plot_convergence(convergence_history, output_dir)

    print(f"Loaded {len(images)} images ({len(test_images)} test, 1 validation).")
    if synthetic:
        print("WARNING: data/images was empty; outputs are a smoke test, not reportable BSDS500 results.")
    print(f"Results written to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/noise_experiment"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-test-images", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast implementation check using small images and reduced parameter grids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        config = Config(
            image_size=64,
            max_test_images=2,
            noise_levels=(0.01, 0.03),
            max_iter=12,
            gamma_grid=(1e-4, 1e-3),
            balance_grid=(1e-3, 1e-2),
            lambda_grid=(3e-3, 1e-2),
        )
    else:
        config = Config(
            image_size=args.size,
            max_test_images=args.max_test_images,
            max_iter=args.max_iter,
        )
    run(args.data_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
