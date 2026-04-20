# GPU Probing Agent System Report

## 1. System Overview

本系统是一个基于 GPT-5.4 的 GPU 自动探测 agent。它的目标不是调用一组预置好的 benchmark 程序去“查答案”，而是：

1. 从 target spec 读取本次要测的指标。
2. 为每个指标自动规划合适的 probe 方案。
3. 用大模型按当前计划动态生成 CUDA/C++ micro-benchmark。
4. 自动编译、运行、调用 `ncu` profiling。
5. 用 benchmark 输出与 `ncu` 指标做交叉验证、异常检测与一致性检查。
6. 在失败或结果不可信时自动生成修复反馈，驱动下一轮实验。
7. 最终对每个 target 给出数值、来源、证据、置信度和推断过程，并写入 `output.md`。

系统正式入口是 `/workspace/run.sh`，它最终调用 `python -m agent.agent_framework`。主编排逻辑在 `agent/agent_framework.py`。

---

## 2. Core Files And Responsibilities

### 2.1 Orchestrator

- `agent/agent_framework.py`

这是整个系统的主控模块。`GPUProbeAgent.run()` 的执行流程是：

1. 调用任务读取模块读取 target spec。
2. 调用策略模块生成 benchmark plans。
3. 逐个 plan 执行多轮实验。
4. 每轮内依次做：代码生成、编译、运行、解析 benchmark JSON、`ncu` profiling、解析 `ncu` 输出、交叉验证。
5. 如果当前轮可信，则接受该轮；否则调用决策模块决定是否重试，并把修复反馈传回代码生成模块。
6. 所有 plan 完成后，调用结果推断模块生成最终 estimate。
7. 调用输出模块写出最终 `output.md`。

这个文件里还实现了两项重要运行机制：

- `early-stop`
  当某个 target 的 primary plan 已经以足够高置信度解析出目标值时，后续同 target 的 cross-check plan 可以直接跳过。
- profiling controls derivation
  根据 benchmark JSON 里的 `kernel_name` 和 `warmup` 信息推导 `ncu` 的 `kernel filter` 与 `launch skip`，尽量只 profile 目标 kernel，减少 profiling 开销和超时风险。

### 2.2 Data Models

- `agent/models.py`

这个文件定义了系统里的核心数据结构：

- `TargetSpec`
  描述 target 文件来源、target 列表、以及 target 到 probe family 的映射。
- `BenchmarkPlan`
  描述一个实验计划，包括 probe family、primary target、variant、round 上限、需要采集的 `ncu` metrics、profiling timeout、profiling env 等。
- `CommandResult`
  统一表示编译、运行、profiling 的命令执行结果。
- `ProbeAttempt`
  表示某个 plan 的某一轮实验，聚合代码路径、编译结果、运行结果、benchmark 输出、`ncu` 指标、验证结果等信息。
- `ValidationResult`
  表示该轮实验是否可信、置信度、问题列表、交叉验证信息、支持性证据。
- `MetricEstimate`
  表示最终某个 target 的结果，包括值、置信度、来源、选择规则和证据。

---

## 3. Module-by-Module Description

### 3.1 Task Reading Module

- `agent/spec_reader.py`
- 类：`TaskReadingModule`

作用：

1. 读取 `/target/target_spec.json`。
2. 如果真实 target 文件不存在，则 fallback 到 `mlagent/target_spec_sample.json`。
3. 解析 `targets` 或 `metrics` 字段。
4. 根据 target 名称把指标初步映射到 probe family。

当前内置的映射规则大致是：

- `dram__bytes_*`、`device__attribute_max_mem_frequency_khz`、`device__attribute_fb_bus_width`、`gpu__compute_memory_throughput...` -> `bandwidth_probe`
- `launch__sm_count`、`device__attribute_max_gpu_frequency_khz`、`sm__throughput...` -> `frequency_probe`
- 带 `latency` 的 target -> `latency_probe`
- 带 `bank conflict` 的 target -> `bank_conflict_probe`
- 带 `cache capacity/size` 的 target -> `cache_capacity_probe`

### 3.2 Benchmark Strategy Selection Module

- `agent/strategy.py`
- `agent/probe_planner.py`
- 类：`BenchmarkStrategySelectionModule`
- 类：`TargetProbePlanningModule`

作用：

1. 根据 target spec 生成一组 `BenchmarkPlan`。
2. 每个 target 不一定只有一个 plan，而是会被拆成 `primary` 和 `cross_check` 两类 variant。
3. 每个 probe family 有自己的 blueprint，包括：
   - benchmark objective
   - benchmark requirements
   - `ncu` metrics
   - success criteria
   - parser expectations
   - max rounds
4. 每个 target 在此基础上再叠加 target-specific variant 设计。

例如：

- `dram__bytes_read.sum.per_second`
  会生成 `read_stream_primary` 和 `read_working_set_sweep`
- `device__attribute_max_mem_frequency_khz`
  会生成 `memory_clock_under_saturation` 和 `memory_clock_intensity_sweep`
- `sm__throughput...`
  会生成 `compute_saturation` 和 `occupancy_compute_sweep`

这个模块还负责给每个 plan 分配 profiling 预算：

- `profile_timeout_s`
- `profile_launch_count`
- `profile_env`

其中 `profile_env` 会控制生成的 benchmark 在 `MLAGENT_PROFILE_MODE=1` 下使用轻量 profiling 配置。

### 3.3 Micro-Benchmark Auto Generation Module

- `agent/codegen.py`
- `agent/prompts/generate_probe_system.txt`
- `agent/prompts/generate_probe.txt`
- 类：`MicroBenchmarkGenerationModule`

作用：

1. 读取 prompt 模板。
2. 把 `BenchmarkPlan` 序列化为 prompt context。
3. 把上轮失败反馈拼进 prompt。
4. 调用 GPT-5.4 生成新的 `.cu` 文件内容。
5. 记录 prompt 和 response 到 artifact 目录。

这个模块的关键点有两个：

- 它不是从内置模板库里“挑一个 CUDA 程序”，而是每轮都根据当前 plan 和反馈动态生成。
- prompt 里明确约束 benchmark 必须输出统一 JSON 合同，包括：
  - `probe_family`
  - `kernel_name`
  - `timings_ms`
  - `parameters`
  - `derived_metrics`
  - `notes`
  - 如果有 sweep，还要输出 `sweep_points`

当前 prompt 还专门加入了针对 memory-side target 的约束：

- DRAM read/write target 必须尽量做方向干净的 read-dominant / write-dominant row
- 不能只依赖 `gpu__compute_memory_throughput`，还必须关注 `dram__throughput`
- `device__attribute_max_mem_frequency_khz` 和 `device__attribute_fb_bus_width` 只允许作为 direct `ncu` target，被 benchmark 间接支撑，而不是靠 benchmark 自己“算出来”

### 3.4 LLM Client Module

- `llm/openai_client.py`
- 类：`GPTClient`

作用：

1. 从环境变量读取：
   - `API_KEY`
   - `BASE_URL`
   - `BASE_MODEL`
   - `OPENAI_REASONING_EFFORT`
2. 默认优先走 `chat.completions`，失败后 fallback 到 `responses`。
3. 对请求设置 timeout、重试次数、退避时间。
4. 兼容不同代理返回格式。
5. 统一返回 `LLMResponse(text, api_mode)`。

这层的意义是把大模型访问从业务逻辑里抽离出来，同时适配当前实验环境里的 OpenAI-compatible gateway。

### 3.5 Compilation And Execution Module

- `agent/executor.py`
- 类：`CompilationAndExecutionModule`

作用：

1. 把生成的代码写到 `generated/`。
2. 调用 `nvcc -O3 -std=c++17 -lineinfo` 编译。
3. 运行编译出来的可执行文件。
4. 保存 `stdout` / `stderr` 到 `raw/`。

它还处理一些运行时细节：

- 当环境里没有 `nvcc` 时，会返回 `returncode=127` 和清晰的错误信息。
- 当编译或执行超时时，会把 `TimeoutExpired` 里的 `stdout/stderr` 也安全落盘。

### 3.6 NCU Profiling Module

- `agent/profiler.py`
- 类：`NCUProfilingModule`

作用：

1. 调用 `ncu` 对指定 binary 做 profiling。
2. 使用 `--page raw --csv` 输出结构化原始结果。
3. 根据 orchestrator 提供的信息附加：
   - `--kernel-name-base demangled`
   - `-k regex:...`
   - `-s`
   - `-c`
   - `--kill 1`
4. 把原始 `stdout/stderr` 落到 `raw/`。

这层的目标不是 profile 整个程序的每一个 kernel，而是尽量抓到 benchmark 输出中声明的 dominant kernel，降低 profiling 开销、减少 timeout，并让解析后的 row 更容易与当前 target 对齐。

### 3.7 Parsing Module

- `agent/parsers.py`

作用：

1. 从 benchmark stdout 最后提取合法 JSON。
2. 自动清理 `inf/nan` 等非法 JSON token。
3. 解析两种 `ncu` CSV 形态：
   - legacy long-table
   - wide-table raw CSV
4. 返回：
   - `ncu_metrics`
   - `ncu_rows`
   - `ncu_metric_units`

这里的设计很重要，因为后续验证和推断不仅看每个 metric 的汇总值，还会看每一条 profiling row 的上下文。

### 3.8 Cross-Validation Module

- `agent/validation.py`
- 类：`CrossValidationModule`

作用：

1. 判断当前一轮实验是否可信。
2. 产出 `ValidationResult`。

它的判断逻辑分几层：

- 基础层
  - 编译是否成功
  - benchmark 运行是否成功
  - benchmark JSON 是否可解析
  - `ncu` profiling 是否成功且可解析
- 稳定性层
  - `timings_ms` 是否存在
  - timing 的方差是否过大
- probe-family-specific 层
  - `bandwidth_probe`：读写 benchmark 与 `ncu` 是否接近，memory-side 证据是否充分
  - `frequency_probe`：benchmark 频率估计与 `ncu` 频率是否接近，`sm__throughput` 是否足够高
  - `latency_probe`：是否产出 latency 指标
  - `bank_conflict_probe`：是否有 benchmark-side penalty 和 `ncu` conflict metric
  - `cache_capacity_probe`：是否出现稳定容量 cliff

这个模块里还区分了：

- `cross_checks`
  只有 benchmark-side 和 `ncu`-side 真正接近时，才算 cross-validation
- `supporting_evidence`
  高 throughput、低 timing variance、`ncu` 返回了某指标，本身只能算支持性证据

### 3.9 Consistency Module

- `agent/consistency.py`

作用：

这是近期加入的“物理一致性闸门”，专门解决 memory-side target 被不同 row 或不同口径结果混用的问题。

它做了几件事：

1. 从 `ncu_rows` 中为不同 target 选择最合适的 row，而不是简单按全局最大值混选。
2. 对 read target 优先选 read-dominant row，对 write target 优先选 write-dominant row。
3. 根据同一条 row 的：
   - `dram__bytes_read.sum.per_second`
   - `dram__bytes_write.sum.per_second`
   - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
   - `device__attribute_max_mem_frequency_khz`
   - `device__attribute_fb_bus_width`
   做物理一致性检查。
4. 如果某个候选值与同一条 row 的理论峰值或 DRAM throughput 百分比明显不自洽，则直接标记为不适合最终值选择。

这层的目标是避免类似“带宽数值只有十几 GB/s，但 memory throughput 却是 90% of peak”这种明显冲突的最终输出。

### 3.10 Error Handling And Retry Module

- `agent/retry.py`
- 类：`ErrorHandlingAndRetryModule`

作用：

1. 根据失败阶段生成结构化 retry directive。
2. 把错误归类为：
   - compile
   - run
   - parse
   - profile
3. 为下一轮代码生成提供针对性的修复反馈。

例如：

- compile failure -> 修语法、头文件、host/device type mismatch
- run failure -> 修非法访问、launch config、内存分配或同步问题
- parse failure -> 保证 stdout 最后只输出一个 JSON
- profile failure -> 保持 dominant kernel 语义不变，同时支持轻量 profiling mode

### 3.11 Experiment Decision And Feedback Module

- `agent/decision.py`
- 类：`ExperimentDecisionAndFeedbackModule`

作用：

当某轮验证不通过时，这个模块决定是否继续重试，并生成下一轮 prompt 的高层改进建议。

它主要基于：

- 当前 probe family
- 当前验证问题列表

自动追加类似建议：

- 增大工作集
- 提高 arithmetic intensity
- 增加 block 数
- 拉长 measurement window
- 增加 repeats
- 加密 sweep points

这使得系统不是简单“失败了就再生成一份代码”，而是把前一轮失败原因反向注入到下一轮生成过程里。

### 3.12 Result Parsing And Inference Module

- `agent/result_inference.py`
- 辅助：`agent/consistency.py`
- 类：`ResultParsingAndInferenceModule`

作用：

1. 汇总所有 `ProbeAttempt`。
2. 为每个 target 搜集 direct `ncu` 候选和 benchmark-derived 候选。
3. 按预声明的 tiered rule 选择最终值。
4. 输出 `MetricEstimate`。

当前的选择规则核心是：

1. 优先 target 自己 accepted round 的 direct `ncu` 值。
2. 再考虑 target 自己 accepted round 的 benchmark-derived 值。
3. 如果 target 自己没有解析出来，再考虑同 probe family 的 accepted shared probe。
4. 更低层 fallback 只能在前面都不成立时使用。

同时，这个模块现在还做了两类过滤：

- 对 DRAM-side target 先经过一致性筛选
- 对 benchmark-derived DRAM bandwidth 先做理论峰值 plausibility check

### 3.13 Reasoning / Logging Module

- `agent/reasoning.py`
- 类：`ReasoningLogger`

作用：

1. 以 JSONL 方式记录系统每一步事件。
2. 记录每个模块的输入输出、选择、跳过、接受、拒绝、重试等信息。

每次 run 都会生成一个新的 artifact 目录，例如：

- `mlagent_artifacts/run_<timestamp>/reasoning_log.jsonl`

这是系统的“机器可读审计轨迹”。

### 3.14 Output Generation Module

- `agent/output_writer.py`
- prompt:
  - `agent/prompts/summarize_output_system.txt`
  - `agent/prompts/summarize_output.txt`
  - `agent/prompts/summarize_per_target_system.txt`
  - `agent/prompts/summarize_per_target.txt`

作用：

1. 汇总 final estimates。
2. 构造结构化 digest：
   - metric cards
   - per-target digests
   - trial and cross-validation digest
   - retry and fix digest
   - raw trace digest
3. 调用 LLM 生成：
   - `Per-Target Findings`
   - `Inference Trace`
4. 最终写出 `output.md`。

当前 `output.md` 的结构包括：

- `Conclusion`
- `Final Value Selection Rules`
- `Results` JSON
- `Per-Target Findings`
- `Trial And Cross-Validation`
- `Retry And Fix History`
- `Method Commitments`
- `Inference Trace`
- `Raw Trace Digest`

这层不仅做文案总结，也会把系统的选择规则、fallback 规则、方法承诺明确写出来，防止最后报告只给结论、不交代依据。

---

## 4. End-to-End Workflow

下面按一次完整运行说明系统是如何工作的。

### Step 1. Launch

启动命令通常是：

```bash
source /workspace/mlagent/demand/环境变量.txt
bash /workspace/run.sh --target-spec /workspace/mlagent/target_spec_sample.json --output /workspace/output.md
```

`run.sh` 只做两件事：

1. 切到 `/workspace/mlagent`
2. 设置 `PYTHONPATH`
3. 调用 `python -m agent.agent_framework`

### Step 2. Read Targets

`TaskReadingModule` 读取 target spec，并输出：

- 本次 target 列表
- 每个 target 对应的 probe family

### Step 3. Build Plans

`BenchmarkStrategySelectionModule` 与 `TargetProbePlanningModule` 根据 target 列表生成 plan。

每个 plan 会定义：

- primary target
- probe family
- probe variant
- benchmark objective
- benchmark requirements
- `ncu` metrics
- parser expectations
- profiling timeout
- profiling env
- max rounds

### Step 4. Generate CUDA Benchmark

`MicroBenchmarkGenerationModule` 把当前 plan 和上一轮反馈一起交给 GPT-5.4，生成新的 `.cu` benchmark。

生成结果会保存在：

- `llm/<plan>_round<p>.prompt.txt`
- `llm/<plan>_round<p>.response.txt`
- `generated/<plan>_round<p>.cu`

### Step 5. Compile And Run

`CompilationAndExecutionModule`：

1. 调用 `nvcc` 编译
2. 运行 binary
3. 采集原始 stdout/stderr
4. 解析 benchmark 输出 JSON

如果 compile/run/parse 任一步失败，系统不会继续 `ncu` profiling，而是直接进入 retry path。

### Step 6. Profile With NCU

如果 benchmark 能正确运行，系统会：

1. 从 benchmark JSON 里读 `kernel_name`
2. 推导 `kernel_name_filter`
3. 推导 `launch_skip`
4. 调用 `NCUProfilingModule.profile()`

这一步会尽量只 profile 目标 kernel，而不会把整套 benchmark 的所有辅助 kernel 都混进来。

### Step 7. Parse And Validate

profiling 完成后：

1. `parsers.py` 解析 benchmark JSON 和 `ncu` CSV。
2. `CrossValidationModule` 判定这轮是否可信。
3. 对 memory-side target，还会通过 `consistency.py` 做方向与物理一致性筛选。

如果当前轮可信，则接受该轮。

如果不可信，则生成 issue、supporting evidence 和 retry feedback。

### Step 8. Retry Or Move On

如果还有轮次预算：

1. `ErrorHandlingAndRetryModule` 生成故障修复反馈。
2. `ExperimentDecisionAndFeedbackModule` 生成针对 probe family 的改进建议。
3. 下一轮 codegen 把这些反馈加入 prompt，再生成新 benchmark。

如果 primary 已经以足够高置信度解决该 target，则 cross-check plan 可能会被 early-stop 跳过。

### Step 9. Infer Final Values

所有 plan 跑完后，`ResultParsingAndInferenceModule` 汇总所有 accepted 或 fallback 候选，按预先声明的规则选出每个 target 的最终值。

这里特别重要的一点是：

- 最终值不是简单取“某列最大值”
- 也不是简单取“最后一轮值”
- 而是结合
  - accepted status
  - target specificity
  - probe family
  - source kind
  - consistency filtering
  - fallback tier
  综合选择

### Step 10. Generate Output

`OutputGenerationModule` 生成最终报告 `/workspace/output.md`，并把所有 trial、cross-validation、retry、选择规则和推断过程写清楚。

---

## 5. Artifact Layout

每次运行都会生成一个新的目录：

- `/workspace/mlagent_artifacts/run_<timestamp>/`

内部通常包括：

- `reasoning_log.jsonl`
  结构化日志
- `llm/`
  prompt 与大模型原始回复
- `generated/`
  生成的 `.cu` 源码
- `bin/`
  编译好的可执行文件
- `raw/`
  compile/run/ncu 的 stdout 与 stderr

这使得系统不仅能给最终答案，也能保留完整实验过程，方便复盘、debug 和人工审查。

---

## 6. Design Properties

### 6.1 Not Hardcoded Benchmark Selection

系统没有把 CUDA benchmark 内置成固定程序列表后直接挑一个执行。真正的 benchmark 代码由 LLM 按当前 plan 和当前反馈动态生成。

### 6.2 Model-Guided But Tool-Grounded

这个系统的核心不是“让大模型直接猜 GPU 参数”，而是：

- 大模型负责写探针
- 编译器、GPU 运行时、`ncu` 负责产出真实测量
- 验证与一致性模块负责筛除不可信结果

因此它是一个 model-guided、tool-grounded 的 agent 系统。

### 6.3 Explicit Trust Model

系统不是把所有结果都叫“交叉验证”。当前实现里：

- 只有 benchmark-side 与 `ncu`-side 数值接近时，才叫 `cross-validation`
- 高 throughput、低 timing variance、`ncu` 返回某指标，只能叫 `supporting evidence`
- memory-side target 还必须通过方向一致性与物理一致性检查

### 6.4 Adaptive Experiment Loop

系统具备最基本的自适应实验能力：

- 失败后不会机械重跑同一份程序
- 会根据 compile/run/profile/validation 的问题生成下一轮修复建议
- 同时用 lighter profiling mode 减少 `ncu` 超时
- 用 early-stop 减少不必要的 cross-check 成本

---



## 8. Summary

这个 agent 已经具备一个完整自动实验系统应有的主要组成部分：

- 任务读取
- 实验规划
- LLM 代码生成
- 编译执行
- `ncu` profiling
- 结果解析
- 交叉验证
- 错误处理与重试
- 实验决策反馈
- 最终推断
- 日志记录
- 报告输出

它的核心思想是：让 GPT-5.4 负责“生成下一轮探针”，让 CUDA 编译器、GPU 实际执行和 `ncu` 负责“给出真实观测”，再由系统内部的验证、选择和一致性机制决定“哪些观测可以进入最终答案”。这使得它不是一个静态 benchmark runner，而是一个真正可迭代、自修复、可审计的 GPU probing agent。
