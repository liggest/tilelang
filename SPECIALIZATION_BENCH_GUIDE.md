# 单组特化 A/B 测试指导

本分支在 PR #3494 的基础上实现了**编译期单组特化**：当 `all_threads == threads`
（一次 allreduce 调用只覆盖一个归约组）时，发射 PR 前的原始索引（无组算术）；
多组调用保留分组修复。本文档说明如何在本地（950DT）验证正确性与性能。

> 临时实验分支的指导文档，若后续转正式 PR 请勿携带本文件与
> `bench_specialization.py`。

## 1. 环境准备

```bash
git fetch origin
git switch cross_warp_reduce_single_specialization
bash build.sh    # 改动了 C++（reduce.cc / finalize_reducer.cc / codegen_pto.cc），必须重编
```

## 2. 正确性验证

预期 44 项全部通过（覆盖单组特化、多组分组、scale=16 交织布局，ascend 与 pto 双参数）：

```bash
TMPDIR=$PWD/tmp TILELANG_DISABLE_CACHE=1 \
    python -m pytest testing/ascend/language/test_tilelang_ascend_reduce.py -q
```

## 3. 性能 A/B（特化 vs 强制分组）

```bash
TMPDIR=$PWD/tmp TILELANG_DISABLE_CACHE=1 python bench_specialization.py micro128 ascend
TMPDIR=$PWD/tmp TILELANG_DISABLE_CACHE=1 python bench_specialization.py micro64   ascend
TMPDIR=$PWD/tmp TILELANG_DISABLE_CACHE=1 python bench_specialization.py rmsnorm   ascend
# pto target：把行尾 ascend 换成 pto，共 6 组
```

脚本流程：先 dump TIR 确认 allreduce 参数，然后用当前源码编译"特化"变体，
临时改写 `allreduce.py` / `reduce.h` 强制全走分组索引再编译"分组"变体（随后还原），
最后对两个二进制做 **8 轮交错** do_bench 测量（抵消频率漂移）。

## 4. 结果解读

每条命令输出示例：

```text
[micro128/ascend] both variants correct
[micro128/ascend] spec=14.744us grouped=14.531us spec-vs-grouped=+1.46% rounds_spec=[...] rounds_grouped=[...]
```

| 字段 | 含义 |
|---|---|
| `spec` | 特化开启（单组用原始索引） |
| `grouped` | 强制分组索引（等价于 PR #3494 head 行为） |
| `spec-vs-grouped` | 负数 = 特化更快（reviewer 观察到的方向），正数 = 分组更快 |
| `rounds_*` | 8 轮各自耗时（已排序）；**两组分布不重叠**说明差异真实，有重叠则属噪声 |

参考对照（950DT / CANN 9.3 weekly 本机结果）：

| 场景 | ascend | pto |
|---|---:|---:|
| micro128（128 线程 scale=16，64 次归约） | +1.46%（分组快） | +0.64% |
| micro64（64 线程 scale=4，64 次归约） | +1.96%（分组快） | +0.19% |
| rmsnorm 4096×4096 | −0.27%（特化快） | +0.02% |

reviewer（silentCoder-dev，950 / CANN 9.2）的数据：micro128 −1.42%、
rmsnorm −0.97%、micro64 +4.80%——与其对照看各场景**方向是否翻转**，
这正是本实验要验证的"±1~2% 差异方向随机器/工具链翻转"。

## 5. 注意事项

- **`TILELANG_DISABLE_CACHE=1` 必须带**：kernel 缓存 key 不包含这几个源文件，
  热缓存会把另一变体的二进制直接喂回来，A/B 失效
- `TMPDIR=$PWD/tmp` 防根分区紧张；根分区健康可省略
- 脚本会临时改写 `allreduce.py` 与 `reduce.h` 并在结束时还原（`finally` 保证）；
  进程被硬杀后用以下命令恢复：
  `git checkout -- tilelang/contrib/ptodsl/allreduce.py src/tl_templates/ascend/reduce.h`
- 可选结构验证：`TILELANG_CLEANUP_TEMP_FILES=0` 编译后检查
  `tmp/tvm-debug-mode-tempdirs/*/kernel.pto`——单组 kernel 的 MLIR 中应无
  `floordiv tid, <threads>` 组算术，多组 kernel 应保留
