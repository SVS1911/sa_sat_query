#!/usr/bin/env python3
"""
tools/test_model.py
--------------------
Standalone check for the Hugging Face vision-language model.

Run this BEFORE starting the server. It isolates model problems (no token,
gated repo, out of memory, missing PyTorch) from application problems, and it
tells you what the model actually returns for a satellite image.

    python tools/test_model.py
    python tools/test_model.py --image data/sample/optical_t1.png --verbose
    python tools/test_model.py --model-id someone/other-adapter --4bit

Exit codes: 0 = model works, 1 = a stage failed, 2 = not configured.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from api.hf_config import load_config, resolve_device  # noqa: E402
from api.hf_model import HFModelRunner  # noqa: E402

TICK, CROSS, WARN = "\u2713", "\u2717", "\u26a0"


def synthetic_scene(size: int = 256) -> np.ndarray:
    """A deterministic RGB scene, so the check needs no sample file.

    Contains water, vegetation and a bright built-up block, giving a land-cover
    model something recognisable to respond to.
    """
    rng = np.random.RandomState(0)
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[:] = (0.45, 0.38, 0.30)
    image[: size // 2, : size // 2] = (0.10, 0.35, 0.12)
    image[size // 2:, : size // 3] = (0.12, 0.20, 0.45)
    image[size // 3: 2 * size // 3, 2 * size // 3:] = (0.72, 0.70, 0.68)
    return np.clip(image + rng.uniform(-0.02, 0.02, image.shape), 0, 1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the SatQuery vision-language model.")
    parser.add_argument("--model-id", help="Override HF_MODEL_ID for this run.")
    parser.add_argument("--base-model-id", help="Override the detected base model.")
    parser.add_argument("--token", help="Override HF_TOKEN for this run.")
    parser.add_argument("--device", help="auto | cuda | mps | cpu")
    parser.add_argument("--4bit", dest="four_bit", action="store_true",
                        help="Load in 4-bit (needs bitsandbytes + CUDA).")
    parser.add_argument("--image", help="Test image path (default: synthetic scene).")
    parser.add_argument("--query", default="What is present in this image?")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="Seconds to wait for weights to load (default 1800).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 66)
    print("SatQuery — vision-language model check")
    print("=" * 66)

    config = load_config()
    if args.model_id:
        config.model_id = args.model_id
    if args.base_model_id:
        config.base_model_id = args.base_model_id
    if args.token:
        config.token = args.token
    if args.device:
        config.device = resolve_device(args.device)
    if args.four_bit:
        config.load_in_4bit = True
    config.enabled = True

    if not config.model_id:
        print(f"\n{CROSS} HF_MODEL_ID is not set.")
        print("\n  Create a .env file next to app.py containing:")
        print("      HF_MODEL_ID=aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora")
        print("      HF_TOKEN=hf_your_token_here")
        print("  or pass --model-id on the command line.")
        return 2

    print(f"\nModel    : {config.model_id}")
    print(f"Token    : {'set' if config.token else 'NOT SET'}")
    print(f"Device   : {config.device}")
    print(f"Precision: {config.dtype}{' + 4-bit' if config.load_in_4bit else ''}")

    if config.device == "cpu":
        print(f"\n{WARN} No GPU detected. A multi-billion-parameter vision model on CPU")
        print("  can take several minutes per answer. This check will still work,")
        print("  it will just be slow.")

    # ------------------------------------------------------------ 1/4
    print("\n[1/4] Checking dependencies...")
    missing = []
    for package, hint in (("torch", "pytorch.org"), ("transformers", "pip install transformers"),
                          ("peft", "pip install peft"),
                          ("huggingface_hub", "pip install huggingface_hub")):
        try:
            __import__(package)
        except ImportError:
            missing.append((package, hint))
    if missing:
        for package, hint in missing:
            print(f"  {CROSS} {package} is not installed  ({hint})")
        return 1
    import torch
    print(f"  {TICK} torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

    # ------------------------------------------------------------ 2/4
    print("\n[2/4] Loading the model...")
    print("  The first run downloads several GB. Later runs read the local cache.")
    runner = HFModelRunner(config)
    started = time.time()
    runner.ensure_loading()

    seen = 0
    while runner.state == "loading":
        for line in runner.status()["progress"][seen:]:
            print(f"  · {line}")
            seen = len(runner.status()["progress"])
        if time.time() - started > args.timeout:
            print(f"  {CROSS} Still loading after {args.timeout:.0f}s. Giving up.")
            return 1
        time.sleep(1.5)

    for line in runner.status()["progress"][seen:]:
        print(f"  · {line}")

    status = runner.status()
    if status["state"] != "ready":
        print(f"\n  {CROSS} {status['message']}")
        print("\nMODEL STATUS: UNAVAILABLE")
        return 1

    print(f"  {TICK} Ready in {status['load_seconds']:.0f}s")
    print(f"      repo      : {status['model_id']}")
    print(f"      base      : {status['base_model_id']}")
    print(f"      is adapter: {status['is_adapter']}")
    for warning in status["warnings"]:
        print(f"  {WARN} {warning}")

    # ------------------------------------------------------------ 3/4
    print("\n[3/4] Preparing a test image...")
    if args.image:
        from utils.image_io import load_image

        try:
            image = load_image(args.image).array
        except Exception as exc:
            print(f"  {CROSS} Could not read '{args.image}': {exc}")
            return 1
        source = args.image
    else:
        image = synthetic_scene()
        source = "synthetic test scene"
    print(f"  {TICK} {source} — {image.shape[1]}x{image.shape[0]}")

    # ------------------------------------------------------------ 4/4
    print("\n[4/4] Running inference...")
    print(f"  Query: {args.query!r}")
    answer = runner.analyze(image, args.query, use_cache=False)
    if not answer.ok:
        print(f"  {CROSS} {answer.error}")
        print(f"\nMODEL STATUS: LOADED BUT NOT USABLE ({answer.error_kind})")
        return 1
    print(f"  {TICK} Answered in {answer.latency_ms / 1000:.1f}s")

    print("\n" + "-" * 66)
    print("MODEL OUTPUT")
    print("-" * 66)
    print(answer.text)

    if answer.labels:
        print("\nLand-cover classes named by the model:")
        for label in answer.labels:
            print(f"  · {label}")
    if answer.generation_confidence is not None:
        print(f"\nGeneration confidence: {answer.generation_confidence:.3f}")
        print("  (mean token probability — decoder certainty in its own wording,")
        print("   not a classification probability)")
    if args.verbose and answer.raw:
        print(f"\nStructured fields parsed from the reply: {answer.raw}")

    print("\n" + "=" * 66)
    print("MODEL STATUS: READY")
    print("=" * 66)
    print("\nStart the app with:\n    python app.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
