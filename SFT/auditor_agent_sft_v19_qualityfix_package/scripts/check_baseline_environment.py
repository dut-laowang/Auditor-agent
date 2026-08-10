"""Fail-fast dependency and CUDA preflight for V19 neural baselines."""

import argparse
import json
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


MINIMUMS = {
    "torch": "2.4.0",
    "transformers": "4.51.0",
    "datasets": "2.18.0",
    "peft": "0.14.0",
    "accelerate": "0.30.0",
    "scikit-learn": "1.3.0",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=["qwen32b", "modernbert"], required=True)
    args = parser.parse_args()
    requirements = dict(MINIMUMS)
    if args.baseline == "qwen32b":
        requirements["bitsandbytes"] = "0.46.1"
    installed = {}
    for package, minimum in requirements.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package: {package}>={minimum}") from exc
        if Version(actual) < Version(minimum):
            raise RuntimeError(f"{package}=={actual} is too old; require >={minimum}")
        installed[package] = actual
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    properties = torch.cuda.get_device_properties(0)
    report = {
        "status": "PASS",
        "baseline": args.baseline,
        "packages": installed,
        "cuda": torch.version.cuda,
        "gpu": properties.name,
        "gpu_memory_gib": round(properties.total_memory / 1024**3, 2),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
