#!/usr/bin/env python3
"""Blur-kernel mismatch experiment for the TV-ADMM course project.

Research question:
    At a fixed noise level, how sensitive are inverse filtering,
    Wiener--Hunt filtering, and TV-ADMM to an inaccurate blur kernel?

The observation is always generated with the true kernel. Reconstruction is
then repeated with deliberately incorrect Gaussian widths, motion lengths,
and motion angles. Parameters are tuned only once with the correct kernel and
are held fixed across all mismatch levels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import (
    Config,
    add_noise,
    circular_convolve,
    gaussian_kernel,
    load_images,
    motion_kernel,
    psf_to_otf,
    psnr,
    restore_all,
    select_parameters,
    ssim,
)


METHODS = ("Blurred+Noise", "Inverse", "Wiener-Hunt", "TV-ADMM")
COLORS = {
    "Blurred+Noise": "#111827",
    "Inverse": "#6b7280",
    "Wiener-Hunt": "#2563eb",
    "TV-ADMM": "#dc2626",
}


def experiment_config(image_size: int, max_test_images: int, max_iter: int) -> Config:
    """Use wider baseline grids because the first experiment hit grid limits."""
    return Config(
        image_size=image_size,
        max_test_images=max_test_images,
        noise_levels=(0.03,),
        rho=0.1,
        max_iter=max_iter,
        gamma_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        balance_grid=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0),
        lambda_grid=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        seed=2026,
    )


def kernel_cases(shape: tuple[int, int]) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    true_gaussian = gaussian_kernel(9, 1.6)
    true_motion = motion_kernel(21, 15, 30.0)
    true_otfs = {
        "gaussian": psf_to_otf(true_gaussian, shape),
        "motion": psf_to_otf(true_motion, shape),
    }

    cases: list[dict[str, object]] = []
    for sigma in (1.28, 1.44, 1.60, 1.76, 1.92):
        cases.append(
            {
                "experiment": "gaussian_sigma",
                "blur_type": "Gaussian",
                "parameter": "sigma",
                "true_value": 1.6,
                "assumed_value": sigma,
                "error_value": 100.0 * (sigma - 1.6) / 1.6,
                "x_value": 100.0 * (sigma - 1.6) / 1.6,
                "x_label": "Gaussian sigma error (%)",
                "assumed_otf": psf_to_otf(gaussian_kernel(9, sigma), shape),
            }
        )

    for length in (12, 14, 15, 17, 18):
        cases.append(
            {
                "experiment": "motion_length",
                "blur_type": "Motion",
                "parameter": "length",
                "true_value": 15.0,
                "assumed_value": float(length),
                "error_value": 100.0 * (length - 15.0) / 15.0,
                "x_value": float(length),
                "x_label": "Assumed motion length (pixels)",
                "assumed_otf": psf_to_otf(motion_kernel(21, length, 30.0), shape),
            }
        )

    for angle in (20.0, 25.0, 30.0, 35.0, 40.0):
        cases.append(
            {
                "experiment": "motion_angle",
                "blur_type": "Motion",
                "parameter": "angle",
                "true_value": 30.0,
                "assumed_value": angle,
                "error_value": angle - 30.0,
                "x_value": angle,
                "x_label": "Assumed motion angle (degrees)",
                "assumed_otf": psf_to_otf(motion_kernel(21, 15, angle), shape),
            }
        )
    return true_otfs, cases


def tune_for_true_kernels(
    validation: np.ndarray,
    true_otfs: dict[str, np.ndarray],
    config: Config,
    noise_sigma: float,
) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    for offset, blur_type in enumerate(("gaussian", "motion")):
        h_fft = true_otfs[blur_type]
        clean_blurred = circular_convolve(validation, h_fft)
        observation = add_noise(clean_blurred, noise_sigma, config.seed + 9000 + offset)
        selected[blur_type] = select_parameters(validation, observation, h_fft, config)
    return selected


def metric_row(
    image_name: str,
    case: dict[str, object],
    method: str,
    reference: np.ndarray,
    estimate: np.ndarray,
    runtime: float,
) -> dict[str, object]:
    return {
        "image": image_name,
        "experiment": case["experiment"],
        "blur_type": case["blur_type"],
        "mismatch_parameter": case["parameter"],
        "true_value": case["true_value"],
        "assumed_value": case["assumed_value"],
        "error_value": case["error_value"],
        "method": method,
        "psnr_db": psnr(reference, estimate),
        "ssim": ssim(reference, estimate),
        "runtime_sec": runtime,
    }


def write_csv_files(rows: list[dict[str, object]], output_dir: Path) -> list[dict[str, object]]:
    fields = [
        "image",
        "experiment",
        "blur_type",
        "mismatch_parameter",
        "true_value",
        "assumed_value",
        "error_value",
        "method",
        "psnr_db",
        "ssim",
        "runtime_sec",
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["psnr_db"] = f"{float(row['psnr_db']):.4f}"
            formatted["ssim"] = f"{float(row['ssim']):.5f}"
            formatted["runtime_sec"] = f"{float(row['runtime_sec']):.6f}"
            writer.writerow(formatted)

    groups: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    case_metadata: dict[tuple[str, float], dict[str, object]] = {}
    for row in rows:
        key = (str(row["experiment"]), float(row["assumed_value"]), str(row["method"]))
        groups[key].append(row)
        case_metadata[(str(row["experiment"]), float(row["assumed_value"]))] = row

    summary: list[dict[str, object]] = []
    for (experiment, assumed, method), values in sorted(groups.items()):
        meta = case_metadata[(experiment, assumed)]
        psnrs = np.array([float(v["psnr_db"]) for v in values])
        ssims = np.array([float(v["ssim"]) for v in values])
        runtimes = np.array([float(v["runtime_sec"]) for v in values])
        summary.append(
            {
                "experiment": experiment,
                "blur_type": meta["blur_type"],
                "mismatch_parameter": meta["mismatch_parameter"],
                "true_value": float(meta["true_value"]),
                "assumed_value": assumed,
                "error_value": float(meta["error_value"]),
                "method": method,
                "mean_psnr_db": float(psnrs.mean()),
                "std_psnr_db": float(psnrs.std(ddof=0)),
                "mean_ssim": float(ssims.mean()),
                "std_ssim": float(ssims.std(ddof=0)),
                "mean_runtime_sec": float(runtimes.mean()),
            }
        )

    # Report the performance loss relative to reconstruction with the exact kernel.
    correct: dict[tuple[str, str], tuple[float, float]] = {}
    for row in summary:
        if abs(float(row["assumed_value"]) - float(row["true_value"])) < 1e-9:
            correct[(str(row["experiment"]), str(row["method"]))] = (
                float(row["mean_psnr_db"]),
                float(row["mean_ssim"]),
            )
    for row in summary:
        baseline = correct[(str(row["experiment"]), str(row["method"]))]
        row["psnr_change_from_correct_db"] = float(row["mean_psnr_db"]) - baseline[0]
        row["ssim_change_from_correct"] = float(row["mean_ssim"]) - baseline[1]

    summary_fields = [
        "experiment",
        "blur_type",
        "mismatch_parameter",
        "true_value",
        "assumed_value",
        "error_value",
        "method",
        "mean_psnr_db",
        "std_psnr_db",
        "mean_ssim",
        "std_ssim",
        "mean_runtime_sec",
        "psnr_change_from_correct_db",
        "ssim_change_from_correct",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary:
            formatted = dict(row)
            for name in (
                "mean_psnr_db",
                "std_psnr_db",
                "mean_ssim",
                "std_ssim",
                "mean_runtime_sec",
                "psnr_change_from_correct_db",
                "ssim_change_from_correct",
            ):
                formatted[name] = f"{float(row[name]):.6f}"
            writer.writerow(formatted)
    return summary


def plot_summary(summary: list[dict[str, object]], cases: list[dict[str, object]], output_dir: Path) -> None:
    titles = {
        "gaussian_sigma": "Gaussian width mismatch",
        "motion_length": "Motion-length mismatch",
        "motion_angle": "Motion-angle mismatch",
    }
    for experiment in ("gaussian_sigma", "motion_length", "motion_angle"):
        exp_rows = [row for row in summary if row["experiment"] == experiment]
        exp_cases = [case for case in cases if case["experiment"] == experiment]
        x_label = str(exp_cases[0]["x_label"])
        x_by_assumed = {float(case["assumed_value"]): float(case["x_value"]) for case in exp_cases}
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for method in METHODS:
            method_rows = sorted(
                (row for row in exp_rows if row["method"] == method),
                key=lambda row: x_by_assumed[float(row["assumed_value"])],
            )
            x = [x_by_assumed[float(row["assumed_value"])] for row in method_rows]
            axes[0].plot(x, [row["mean_psnr_db"] for row in method_rows], "o-", label=method, color=COLORS[method])
            axes[1].plot(x, [row["mean_ssim"] for row in method_rows], "o-", label=method, color=COLORS[method])
            if method != "Blurred+Noise":
                axes[2].plot(
                    x,
                    [row["psnr_change_from_correct_db"] for row in method_rows],
                    "o-",
                    label=method,
                    color=COLORS[method],
                )
        axes[0].set(ylabel="Mean PSNR (dB)", title=f"{titles[experiment]}: PSNR")
        axes[1].set(ylabel="Mean SSIM", title=f"{titles[experiment]}: SSIM")
        axes[2].set(ylabel="PSNR change from correct kernel (dB)", title="Sensitivity relative to correct kernel")
        for axis in axes:
            axis.set_xlabel(x_label)
            axis.grid(alpha=0.25)
            axis.axhline(0, color="#9ca3af", linewidth=0.8, zorder=0) if axis is axes[2] else None
            axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{experiment}_curves.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_visual_examples(
    visual_examples: dict[str, list[tuple[str, np.ndarray, np.ndarray, dict[str, np.ndarray]]]],
    output_dir: Path,
) -> None:
    for experiment, examples in visual_examples.items():
        fig, axes = plt.subplots(len(examples), 5, figsize=(14, 3 * len(examples)), squeeze=False)
        for row_index, (label, reference, observation, estimates) in enumerate(examples):
            images = [reference, observation, estimates["Inverse"], estimates["Wiener-Hunt"], estimates["TV-ADMM"]]
            titles = ["Reference", "Blurred + noise", "Inverse", "Wiener-Hunt", "TV-ADMM"]
            for col_index, (title, image) in enumerate(zip(titles, images)):
                axes[row_index, col_index].imshow(image, cmap="gray", vmin=0, vmax=1)
                axes[row_index, col_index].axis("off")
                if row_index == 0:
                    axes[row_index, col_index].set_title(title)
            axes[row_index, 0].set_ylabel(label)
        fig.tight_layout()
        fig.savefig(output_dir / f"{experiment}_visual_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def run(data_dir: Path, output_dir: Path, config: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images, synthetic = load_images(data_dir, config)
    validation_name, validation = images[0]
    test_images = images[1:]
    if synthetic:
        raise RuntimeError("Real images are required for the kernel-mismatch experiment.")
    if not test_images:
        raise RuntimeError("At least one test image is required.")

    noise_sigma = 0.03
    true_otfs, cases = kernel_cases(validation.shape)
    selected = tune_for_true_kernels(validation, true_otfs, config, noise_sigma)

    metadata = {
        "research_question": "Sensitivity to blur-kernel mismatch at fixed noise",
        "data_source": str(data_dir.resolve()),
        "validation_image": validation_name,
        "test_images": [name for name, _ in test_images],
        "image_size": config.image_size,
        "noise_sigma": noise_sigma,
        "random_seed": config.seed,
        "stage": "preliminary project-update experiment",
        "known_limitations": [
            "one validation image rather than a multi-image validation set",
            "fixed ADMM rho and iteration count rather than residual-based stopping",
            "one qualitative example per mismatch type without matched crops or intensity profiles",
        ],
        "true_gaussian_kernel": {"size": 9, "sigma": 1.6},
        "assumed_gaussian_sigmas": [1.28, 1.44, 1.6, 1.76, 1.92],
        "true_motion_kernel": {"size": 21, "length": 15, "angle_degrees": 30.0},
        "assumed_motion_lengths": [12, 14, 15, 17, 18],
        "assumed_motion_angles_degrees": [20, 25, 30, 35, 40],
        "rho": config.rho,
        "max_iter": config.max_iter,
        "selected_parameters_with_correct_kernel": selected,
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    visual_examples: dict[str, list[tuple[str, np.ndarray, np.ndarray, dict[str, np.ndarray]]]] = defaultdict(list)
    example_index = min(1, len(test_images) - 1)  # reproducibly use the second test image when available

    cases_by_experiment: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        cases_by_experiment[str(case["experiment"])].append(case)

    for image_index, (image_name, reference) in enumerate(test_images):
        observations: dict[str, np.ndarray] = {}
        for offset, blur_type in enumerate(("gaussian", "motion")):
            clean_blurred = circular_convolve(reference, true_otfs[blur_type])
            observations[blur_type] = add_noise(
                clean_blurred,
                noise_sigma,
                config.seed + image_index * 100 + offset,
            )

        for experiment, experiment_cases in cases_by_experiment.items():
            blur_key = "gaussian" if experiment == "gaussian_sigma" else "motion"
            observation = observations[blur_key]
            for case_index, case in enumerate(experiment_cases):
                estimates, runtimes, _ = restore_all(
                    observation,
                    case["assumed_otf"],
                    selected[blur_key],
                    config,
                )
                rows.append(metric_row(image_name, case, "Blurred+Noise", reference, observation, 0.0))
                for method, estimate in estimates.items():
                    rows.append(metric_row(image_name, case, method, reference, estimate, runtimes[method]))

                if image_index == example_index and case_index in (0, 2, 4):
                    if experiment == "gaussian_sigma":
                        label = f"sigma={float(case['assumed_value']):.2f}"
                    elif experiment == "motion_length":
                        label = f"length={int(float(case['assumed_value']))}"
                    else:
                        label = f"angle={int(float(case['assumed_value']))} deg"
                    visual_examples[experiment].append((label, reference, observation, estimates))

    summary = write_csv_files(rows, output_dir)
    plot_summary(summary, cases, output_dir)
    plot_visual_examples(visual_examples, output_dir)

    print(f"Validation image: {validation_name}")
    print(f"Test images: {len(test_images)}")
    print(f"Selected Gaussian parameters: {selected['gaussian']}")
    print(f"Selected motion parameters: {selected['motion']}")
    print(f"Results written to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/kernel_mismatch_experiment"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max-test-images", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        config = experiment_config(64, 2, 10)
    else:
        config = experiment_config(args.size, args.max_test_images, args.max_iter)
    run(args.data_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
