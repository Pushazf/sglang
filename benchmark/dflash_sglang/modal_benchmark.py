"""Modal launcher for DFlash sglang benchmark.

Usage:
  modal run benchmark/dflash_sglang/modal_benchmark.py

  # Override parameters
  modal run benchmark/dflash_sglang/modal_benchmark.py \
      --extra-args="--concurrencies 1,2,4,8,16 --max-new-tokens 1024"
"""

import modal
import os

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libnuma1", "libnuma-dev")
    .apt_install("git")
    .add_local_dir(
        os.path.join(_REPO_ROOT, "python"),
        remote_path="/root/sglang/python",
        copy=True,
    )
    .run_commands(
        "pip install --upgrade pip",
        "pip install torch==2.9.1",
        "pip install nvidia-cudnn-cu12==9.16.0.29",
        "pip install -e /root/sglang/python --no-deps",
        "pip install -e /root/sglang/python",
    )
    .pip_install(
        "accelerate==1.11.0",
        "packaging",
        "ninja",
        "rich",
        "datasets",
    )
    .add_local_dir(".", remote_path="/root/dflash_sglang")
)

app = modal.App("dflash-sglang-benchmark", image=image)

vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
qwen_vol = modal.Volume.from_name("qwen3_5_dflash", create_if_missing=True)


@app.function(
    gpu="H200:1",
    timeout=86400,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret-keweiz")],
    volumes={
        "/root/.cache/huggingface": vol,
        "/root/qwen3_5_dflash": qwen_vol,
    },
)
def run_benchmark(extra_args: str = ""):
    import glob
    import subprocess
    import shlex

    os.environ["SGLANG_DISABLE_CUDNN_CHECK"] = "1"

    cwd = "/root/dflash_sglang"

    cmd = ["bash", "run_benchmark.sh"]
    if extra_args:
        cmd += shlex.split(extra_args)

    subprocess.run(cmd, check=True, cwd=cwd)

    results: dict[str, bytes] = {}
    results_dir = os.path.join(cwd, "results")
    md_paths = sorted(glob.glob(os.path.join(results_dir, "*.md")))
    for md_path in md_paths:
        if os.path.isfile(md_path):
            with open(md_path, "rb") as f:
                results[os.path.basename(md_path)] = f.read()

    return results


@app.local_entrypoint()
def main(extra_args: str = ""):
    print("Running DFlash sglang benchmark on Modal ...")
    results = run_benchmark.remote(extra_args=extra_args)

    if not results:
        print("No markdown report was generated.")
        return

    output_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(output_dir, exist_ok=True)

    for filename, content in results.items():
        dest = os.path.join(output_dir, filename)
        with open(dest, "wb") as f:
            f.write(content)
        print(f"Saved: {dest}")

    print(f"\nDone. {len(results)} file(s) saved to {output_dir}/")
