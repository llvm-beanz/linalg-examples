"""Fit and print parameters for the sine-network D3D12 sample."""

import argparse

import numpy as np


HIDDEN_COUNT = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit the sine network's output weights while modeling its FP16 "
            "hidden-layer arithmetic."
        )
    )
    parser.add_argument(
        "--training-samples",
        type=int,
        default=16385,
        help="number of evenly spaced training inputs (default: 16385)",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=257,
        help="number of evenly spaced validation inputs (default: 257)",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-5,
        help="L2 regularization strength for the output weights (default: 1e-5)",
    )
    args = parser.parse_args()
    if args.training_samples < 2 or args.validation_samples < 2:
        parser.error("sample counts must be at least 2")
    if args.ridge < 0:
        parser.error("--ridge must be nonnegative")
    return args


def make_inputs(count: int) -> np.ndarray:
    return np.linspace(-np.pi, np.pi, count, dtype=np.float32)


def make_slopes() -> np.ndarray:
    return np.geomspace(0.25, 3.0, HIDDEN_COUNT).astype(np.float16)


def shader_hidden_features(inputs: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    half_inputs = inputs.astype(np.float16)
    half_products = (half_inputs[:, None] * slopes[None, :]).astype(np.float16)
    return np.tanh(half_products.astype(np.float32)).astype(np.float64)


def fit_output_weights(
    features: np.ndarray, targets: np.ndarray, ridge: float
) -> np.ndarray:
    normal_matrix = features.T @ features
    normal_matrix += ridge * np.eye(features.shape[1])
    return np.linalg.solve(normal_matrix, features.T @ targets)


def print_cpp_array(name: str, values: np.ndarray) -> None:
    print(f"constexpr std::array<float, {len(values)}> {name} = {{")
    for start in range(0, len(values), 4):
        row = values[start : start + 4]
        print("    " + " ".join(f"{value:.11g}f," for value in row))
    print("};")


def main() -> None:
    args = parse_args()
    slopes = make_slopes()

    training_inputs = make_inputs(args.training_samples)
    training_features = shader_hidden_features(training_inputs, slopes)
    training_targets = np.sin(training_inputs.astype(np.float64))
    weights = fit_output_weights(training_features, training_targets, args.ridge)

    validation_inputs = make_inputs(args.validation_samples)
    validation_features = shader_hidden_features(validation_inputs, slopes)
    validation_targets = np.sin(validation_inputs.astype(np.float64))
    errors = validation_features @ weights - validation_targets

    print(
        f"training samples: {args.training_samples}\n"
        f"validation samples: {args.validation_samples}\n"
        f"ridge: {args.ridge:g}\n"
        f"max error: {np.max(np.abs(errors)):.8g}\n"
        f"RMS error: {np.sqrt(np.mean(np.square(errors))):.8g}\n"
    )
    print_cpp_array("kSlopes", slopes.astype(np.float64))
    print()
    print_cpp_array("kOutputWeights", np.append(weights, 0.0))


if __name__ == "__main__":
    main()