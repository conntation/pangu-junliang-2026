# 俊良队｜2026 先导杯 Pangu-Weather 决赛优化代码

## 代码地图

| 文件 | 作用 |
| --- | --- |
| [`train.py`](train.py) | 学生模型、教师参数继承、蒸馏损失、训练和检查点 |
| [`train_regression.py`](train_regression.py) | 提交包内另一套训练脚本，依赖比赛软件栈 |
| [`inference.py`](inference.py) | 48 小时自回归预测、低显存执行、权重加载与输出流水 |
| [`quantize_w8a16.py`](quantize_w8a16.py) | 大权重逐第 0 维切片 INT8 存储及反量化 |
| [`quantize_w4_storage.py`](quantize_w4_storage.py) | 深层 Linear 的 W4/W5 分块打包存储 |
| [`quantize_epb.py`](quantize_epb.py) | Earth-specific position bias 的 INT8/INT4 存储 |
| [`gfx936_biasmask/`](gfx936_biasmask/) | gfx936 HIP/C++ 扩展和运行时加载器 |
| [`export.py`](export.py)、[`conv_fp16.py`](conv_fp16.py) | 提取推理权重、转换浮点张量为 FP16 |
| [`makelink.py`](makelink.py) | 从已有 ERA5 数据建立稀疏时间样本的符号链接 |
| [`conf/config.yaml`](conf/config.yaml) | 决赛提交时的数据管线和模型配置 |

## 优化思路与实现

### 1. 扩大 Patch，训练轻量学生模型

教师使用 `2×4×4` patch、192 维嵌入；[`train.py`](train.py) 默认学生使用 `2×8×8` patch、192 维嵌入和 `[6, 12, 12, 6]` 注意力头。两个空间边长翻倍，使相同输入分辨率下的空间 token 数约为教师的四分之一。网络仍保留分层 Transformer、窗口注意力和地表／高空联合预测：输入为 4 个地表动态量、3 个静态地理场、65 个高空量，最终输出完整 **4 + 65 = 69** 个气象通道。学生构建时会按新 patch 尺寸重建 `patchrecovery2d/3d`，这是参数继承能生效的前提。

`teacher_slice_interpolate` 初始化模式将同形状权重直接继承；尺寸变动的参数按各维前导切片裁剪，Q/K/V 分段处理保持段对齐，patch 卷积核做通道切片加保总量的空间插值，地球位置偏置按注意力头切片后双线性表插值。初始化报告统计 exact/sliced/qkv_sliced/interpolated/skipped 五类继承比例，低于阈值（默认 0.30）时默认拒绝继续训练；旧的单轴 legacy 切片实现已被硬禁用，仅存档保留。教师冻结，学生使用可拆分的全 69 通道教师损失、官方 15 通道教师损失及可选真实标签损失，另有气压层加权、AdamW、预热与余弦学习率、梯度裁剪、有限值和形状检查。**当前脚本的默认损失预设是 `teacher15`**，其权重为全 69 通道教师损失 0.2 + 官方 15 通道教师损失 1.0，并非纯 15 通道；其他实验需显式传入 `--loss-preset` 或各项损失权重，不能把 PDF 中对全通道蒸馏的描述当作所有运行的默认值。

训练侧的其他工程优化：

- **残差学习模式**（`--residual-output`）：学生预测增量，输出 = 持续性基线（输入的 69 通道）+ `residual-scale × delta`，scale 可设为可学习标量。
- `--pad-student-input` 默认开启：输入 replicate 对称 pad 到 patch 倍数，输出再裁剪回原尺寸。
- 默认 fp16 AMP + GradScaler，溢出治理区分可恢复溢出与真梯度故障，连续溢出上限 8 次。
- best_teacher15 / best_gt15 / best_val_loss 三路 best 检查点独立评选；保存走临时文件 + `os.replace` 的原子写；resume 时校验学生架构、教师 checkpoint sha256 与损失配置一致性。
- 可复现性设施：`--sanity-only`（单次前向检查后退出）、`--seed`、`--deterministic`。`--feature-distill` 中间层特征蒸馏是未实现的死路径，传入即报错。

[`makelink.py`](makelink.py) 对已有 ERA5 文件按年月日时选择起点，并额外保留 `start + lead_hours` **一个**标签时刻（注意：并非起止之间的连续时次），通过符号链接构建较小训练子集，不复制原始大数据。它的默认月份、日期与 `--lead-hours 24` 仅是脚本默认值，具体训练采样方案应以运行参数和数据布局为准。

### 2. 清理检查点并以低位格式存储

[`export.py`](export.py) 从完整训练检查点仅提取 `model_state_dict`，删除 `.attn_mask`、`.earth_position_index`、`device_buffer` 和 `.Fuser.` 键，不做 dtype 转换，也不下载权重。注意它**不**删除 `Sampler`/`Reconvery` 键——那一部分在 [`quantize_w8a16.py`](quantize_w8a16.py) 的 `DEFAULT_DROP_PATTERNS` 中，由量化脚本顺带丢弃。`export.py` 使用写死的输入输出路径，运行前应修改为自己的文件路径。

三个量化脚本是**叠加在同一检查点上的流水线**：先做 W8 基线量化，再在其上分别做深层 W4 和 EPB 压缩。

- **FP16**：[`conv_fp16.py`](conv_fp16.py) 把**所有**浮点张量（含 bias、norm，不止权重）转成 FP16；[`inference.py`](inference.py) 固定 FP16 模型执行。`conv_fp16.py` 的目标文件名虽然含 `bf16`，代码实际调用 `.half()`，产生的是 **FP16**。
- **W8A16**：[`quantize_w8a16.py`](quantize_w8a16.py) 对符合条件的多维 `.weight` 按第 0 维切片做对称 INT8 量化，默认最少 1024 个元素、scale 为 FP16；不适合量化的小张量被转成 FP16 而非保留原精度。第 0 维对 `ConvTranspose3d` 并非输出通道，故更准确的说法是"逐第 0 维切片"。包装器覆盖 Linear、Conv3d、ConvTranspose3d；计算阶段把权重现场反量化回 FP16 走普通 GEMM，并无 INT8 计算，低位权重只节省存储和加载占用。脚本还提供 `--mode dequantize` 还原 FP16 检查点。
- **深层 W4/W5**：[`quantize_w4_storage.py`](quantize_w4_storage.py) 在已量化的 W8 检查点上，仅对 `layer2`/`layer3` 的二维 Linear 沿输入维分块重编码（`--mode deep` 即这两层），默认 **4 bit、块长 16**，也可用 `--bits 5`。这是磁盘打包格式：4 bit 两个值拼一字节，5 bit 用 int64 字打包成 5 字节 8 值；加载时展开并可建立 FP16 权重缓存，未实现原生 INT4/INT5 GEMM。二次量化前会先反量化回原浮点权重再分块，属于有损的二次量化。
- **位置偏置 EPB**：[`quantize_epb.py`](quantize_epb.py) 将三维 `earth_position_bias_table` 按第 0 维分块量化并保存 scale；支持 INT8、INT4 和去除偏置的消融模式（`--bits 0`）。工具默认 **INT8、块长 72**；选 `--bits 4` 时默认块长 **18**，可显式覆盖，INT4 的 qmax 为 7（实际用 15 个量化级），并把相邻两行的 4 bit 值打包在一个字节。PDF 中的 INT4"72 行分块"不是当前代码的默认设置。

这些格式允许**磁盘低位存储与运行时 FP16 稠密计算解耦**。不同打包方式的磁盘容量和运行时峰值显存不能直接等同；推理脚本还会为一些权重建立缓存。原 PDF 报告过 W8 检查点约 65.23 MB、固定 mask 唯一存储约 143.33 MB→25.21 MB；这些数字属于当次实验，仓库未附可重算的原权重或完整测量日志。

### 3. 控制激活生命周期和注意力峰值

[`inference.py`](inference.py) 的 `lean_pangu_forward` 在地表／高空嵌入、token 重排、各层和 recovery 之间尽早释放不再使用的张量引用，使 PyTorch 分配器能够复用 storage。Lean Transformer block 将 norm、窗口划分、attention、残差和 MLP 结果按消费顺序组织；MLP 的 GELU 可原地执行，`torch.addmm(..., out=...)` 将 `fc2` 输出写入残差目标。**注意 GELU 默认走 tanh 近似**（`PANGU_GELU_APPROXIMATE` 默认 `tanh`），与精确 GELU 存在数值差异；输入边缘 padding 使用 **replicate** 而非零填充。默认关闭 skip-token CPU offload（`PANGU_OFFLOAD_SKIP_TOKENS=0`）。

默认注意力模式为 `hybrid_qscale`：深层拆开 Q/K/V 的生成时序，使 V 不与 Q/K 长时间同时驻留；浅层以窗口宽度方向分块，仍在每个窗口内计算完整注意力。Q scale 可折叠到权重或原地计算，score buffer 在加偏置、mask 和 softmax 阶段复用。`PANGU_ATTENTION_WIDTH_CHUNK=4` 控制浅层窗口批次，默认深层不做同类分块。固定 shifted-window mask 的值域检查通过后，`int8_shared` 模式将其转为 INT8，并对内容与形状相同的 mask 共享 buffer。

以下推理优化同样默认开启：

- **Direct window partition/reverse**（默认开）：预计算索引，把 pad + cyclic shift + window partition 融为一次 `index_select`，reverse 与裁剪同理，完全绕过原 pad/roll 路径。
- **norm + gather 融合**（`PANGU_FUSE_LAYER_NORM_GATHER=1` 默认开）：浅层用 gfx936 `layer_norm_gather` kernel 一次完成 LayerNorm 与窗口 gather。
- **Linear/Conv FP16 权重常驻缓存**（`PANGU_LINEAR_CACHE_MODE=all`、`PANGU_CONV_CACHE=1` 均默认开）：加载时把所有二维量化 Linear 和 recovery 卷积反量化为常驻 FP16（`weight_fp16`）并替换 forward，是显存换延迟的默认行为。
- **EPB 运行时瞬态反量化**：INT8/INT4 EPB 表在每次注意力调用时现场解包（`PANGU_EPB_CACHE_MODE` 默认 `none`，FP16 EPB 缓存默认关闭）；INT4 EPB 在加载时一次性解成 INT8 buffer。
- 分块注意力内参数复用（`PANGU_REUSE_CHUNK_ATTENTION_PARAMETERS=1`）、EPB bias 复用（`PANGU_REUSE_CHUNK_EPB_BIAS=1`）、投影直接写入预分配输出（`PANGU_CHUNK_LINEAR_OUT=1`）。

`PANGU_MLP_TOKEN_CHUNK=43680` 且启用 residual 输出时，浅层 MLP 按 token 分块执行 `fc1 → GELU → fc2` 并写入对应残差切片；**但分块仅在 token 数超过阈值时触发**——按 config 的 721×1440 输入，patch 8×8 后浅层 token 数为 91×180 = 16380，默认配置下该分块路径不会触发（深层还被 `PANGU_MLP_CHUNK_DEEP=0` 排除），默认实际生效的只有 `out=` 残差写回。`PANGU_RECOVERY3D_CHUNK_WIDTH=45` 沿经度分块恢复全部 65 个高空通道，并未删减输出变量。

### 4. gfx936 HIP 算子与输入输出流水

[`gfx936_biasmask/`](gfx936_biasmask/) 中的 PyTorch 扩展针对 **gfx936** 编译。代码实现了 attention bias 与 INT8 mask 融合加法（softmax 仍留给 PyTorch，mask 契约是值域恰好 `{0, -100}`）、无 affine LayerNorm（固定宽度 **192/384/768**）、残差加法加归一化、token gather（索引 `-1` 写零）、norm 加 gather，以及 FP16 网络输出到 FP32 反归一化输出的融合 kernel。推理加载阶段还能把 LayerNorm 的 affine 参数折叠进后继 Linear，并折叠 attention Q 缩放。

**该扩展没有运行时回退**：非 ROCm 环境、架构不是 gfx936 或扩展编译失败时直接 `raise RuntimeError`；fused bias/mask kernel 还要求 `PANGU_ATTENTION_MASK_MODE=int8_shared` 且注意力为 conventional 模式，否则同样报错。唯一的降级方式是启动前手动把各 `PANGU_*_MODE` 设为 `eager`。首次运行会在当前目录 `$CWD/.torch_extensions` 下 JIT 编译扩展，这发生在计时开始之前。这些优化均以代码中的检查与分支为前提，其他 GPU 架构不能直接假定兼容。

默认输入模式 `PANGU_STREAMED_MODEL_INPUT=1`：先组装 4 个地表量和 3 个静态场并做 2D embedding，再组装 65 个高空量做 3D embedding，释放各阶段输入后合并特征。默认输出模式 `PANGU_OUTPUT_PIPELINE=1`、`PANGU_OUTPUT_TRANSFER_CHUNK=8`，在两个 pinned CPU buffer 间轮换，使 FP16→FP32 affine、GPU 到 CPU 拷贝按通道块流水执行，最终同步后才结束计时。原 PDF 记录流式输入与原路径的 10 个完整输出逐字节一致；仓库没有附上该回归测试数据。

另有两个存在但默认不可用的执行模式：`EXECUTION_MODE=cuda_graph`（CUDA Graph 封装）与默认开启的流式输入互斥，默认配置下走不到；`PANGU_WARMUP_STEPS` 仅为打印用途，实际 warmup 只有 CUDA Graph runner 内的两次前向。

## 环境、使用和复现边界

比赛环境以 OneScience 的 `onescience` 包、PyTorch 2.5.1、ROCm/HIP 6.3、gfx936 为基础。需自行准备**有权使用的**教师／学生权重、ERA5 HDF5 年度数据、统计量和静态场。`conf/config.yaml` 中 `../onedatasets/ERA5_test/` 和测试年份标记保留决赛布局；它的 `model.patch_size` 仍写 `[2,4,4]`，而 [`inference.py`](inference.py) 的架构常量是 `[2,8,8]`。请按实际权重和数据管线逐项核对，不要直接把两处配置视为一致。config 中 `train_ratio`/`val_ratio` 为 `[2012]` 等占位值，与 `time_range` 字段对不上，同样需按实际数据布局核对。

```bash
# 查看训练、推理与量化参数（需要先配置原比赛依赖）
python train.py --help
python inference.py --help
python quantize_w8a16.py --help
python quantize_w4_storage.py --help
python quantize_epb.py --help

# 使用自己准备的权重、数据和统计量
python inference.py --config conf/config.yaml \
  --checkpoint /path/to/your/compatible-model.pth \
  --output-dir result/output
```

`inference.py` 采用未来 48 小时、每 6 小时一步的自回归任务流程；实际输出文件和计时受测试数据与运行参数影响。注意：`--output-dir` 参数会被循环内硬编码的 `result/output/{filename}` 覆盖，实际 npy 落盘位置不随该参数改变；计时区间内还包含每个样本结束后的 `gc.collect()` + `torch.cuda.empty_cache()` 和逐样本 `print`，这些都会计入计时。这里没有权重、数据、容器镜像或完整依赖锁定文件，**不能仅凭 clone 直接复现决赛分数**。公开前只进行了源码检查与 Python 语法检查，未在原设备上重新跑数值、延迟或显存基准。

## 来源与许可

模型背景可参见 [Pangu-Weather 论文](https://www.nature.com/articles/s41586-023-06185-3)。此代码基于赛事提供的 OneScience/Pangu 环境开发，依赖组件分别遵循其上游许可。
