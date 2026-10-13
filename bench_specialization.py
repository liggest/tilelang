"""Interleaved A/B benchmark: single-group specialization vs forced grouped.

Compiles the same kernel twice: once with the current sources
(specialization on), once with the trace-time branch in allreduce.py and
the if constexpr in reduce.h patched so every call uses the grouped
indexing. Both binaries are then measured in 8 interleaved do_bench
rounds so device frequency drift cancels.

Usage:
    TMPDIR=$PWD/tmp TILELANG_DISABLE_CACHE=1 python bench_specialization.py <micro128|micro64|rmsnorm> <ascend|pto>

TILELANG_DISABLE_CACHE=1 is required: the kernel cache does not key on
these source files, so a warm cache would serve the other variant's
binary. If the run is interrupted hard, restore the patched files with
git checkout --.
"""

import os
import shutil
import sys

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

FILES = {
    "tilelang/contrib/ptodsl/allreduce.py": [
        ("single_group = all_threads == threads", "single_group = False"),
    ],
    "src/tl_templates/ascend/reduce.h": [
        ("if constexpr (all_threads != threads)", "if constexpr (true)"),
    ],
}
BACKUP = os.path.join("tmp", "bench_backup")
PATCHES = [(f, old, new) for f, subs in FILES.items() for old, new in subs]


def backup():
    shutil.rmtree(BACKUP, ignore_errors=True)
    for f in FILES:
        dst = os.path.join(BACKUP, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(f, dst)


def patch():
    for f, old, new in PATCHES:
        text = open(f).read()
        assert old in text, f"{old!r} not found in {f}"
        open(f, "w").write(text.replace(old, new))


def restore():
    for f in FILES:
        shutil.copy(os.path.join(BACKUP, f), f)


def make_micro(threads, scale, dtype="float32", repeats=32):
    extent = threads // scale

    @T.prim_func
    def kernel(A: T.Tensor((scale, extent), dtype), B: T.Tensor((scale,), dtype)):
        with T.Kernel(1) as _, T.SimtVF(threads=threads):
            a = T.alloc_fragment((scale, extent), dtype)
            b1 = T.alloc_fragment((scale,), dtype)
            b2 = T.alloc_fragment((scale,), dtype)
            c = T.alloc_fragment((scale,), dtype)
            T.copy(A, a)
            T.annotate_layout({a: T.Fragment((scale, extent), forward_fn=lambda i, j: (j * scale + i, 0))})
            for _ in range(repeats):
                T.reduce_sum(a, b1, dim=1)
                T.reduce_sum(a, b2, dim=1)
            for i in T.Parallel(scale):
                c[i] = b1[i] + b2[i]
            T.copy(c, B)

    return kernel


def make_rmsnorm():
    sys.path.insert(0, os.path.join("examples", "ascend"))
    from example_rmsnorm import rms_norm_fwd

    return rms_norm_fwd(4096, 4096)


def main(name, target):
    if name == "rmsnorm":
        prog, kind = make_rmsnorm(), "rmsnorm"
    else:
        prog, kind = make_micro(128 if name == "micro128" else 64, 16 if name == "micro128" else 4), None

    backup()
    if kind == "rmsnorm":
        k_spec = tilelang.compile(prog, target=target, out_idx=[1, 3])
    else:
        k_spec = tilelang.compile(prog, target=target, out_idx=-1)
    try:
        patch()
        if kind == "rmsnorm":
            k_grouped = tilelang.compile(prog, target=target, out_idx=[1, 3])
        else:
            k_grouped = tilelang.compile(prog, target=target, out_idx=-1)
    finally:
        restore()

    torch.npu.set_device(0)
    if kind == "rmsnorm":
        x = torch.randn(4096 * 4096, device="npu")
        w = torch.randn(4096, device="npu")

        def fn(k):
            return k(x, w, 1e-6)

    else:
        x = torch.randn(*(16, 8) if name == "micro128" else (4, 16), device="npu")

        def fn(k):
            return k(x)

    out_spec, out_grouped = fn(k_spec), fn(k_grouped)
    torch.npu.synchronize()
    if kind != "rmsnorm":
        ref = 2 * x.sum(dim=1)
        assert torch.allclose(out_spec.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4), "spec wrong"
        assert torch.allclose(out_grouped.cpu(), ref.cpu(), rtol=1e-4, atol=1e-4), "grouped wrong"
    print(f"[{name}/{target}] both variants correct", flush=True)

    spec, grouped = [], []
    n = 50 if kind == "rmsnorm" else 10
    for _ in range(8):
        spec.append(do_bench(lambda: fn(k_spec), backend="msprof", _n_warmup=30, _n_repeat=n))
        grouped.append(do_bench(lambda: fn(k_grouped), backend="msprof", _n_warmup=30, _n_repeat=n))

    s = sorted(v * 1000 for v in spec)
    g = sorted(v * 1000 for v in grouped)
    sm, gm = s[3], g[3]
    delta = (sm - gm) / gm * 100
    print(
        f"[{name}/{target}] spec={sm:.3f}us grouped={gm:.3f}us spec-vs-grouped={delta:+.2f}% "
        f"rounds_spec={[f'{v:.2f}' for v in s]} rounds_grouped={[f'{v:.2f}' for v in g]}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main(sys.argv[1], sys.argv[2])
    finally:
        if os.path.exists(BACKUP):
            restore()
