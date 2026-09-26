# 山西大学俊良队｜2026 先导杯 Pangu-Weather 决赛代码

这是山西大学俊良队参加 **2026 年智能计算创新设计赛（先导杯）气象大模型赛道决赛** 的提交代码整理版。任务是在国产科学智算环境中压缩 Pangu-Weather 并优化完整气象预测的推理部署。代码保留决赛提交时的实现与配置；仓库不含权重、ERA5 数据或比赛平台提供的 OneScience 环境。

> **成绩口径**：原决赛说明文档记录的上机结果为推理时长分 18.1071、模型轻量化分 26.0759、预测性能分 19.4680，总分 **63.6510**。这些是当时比赛环境与评测规则下的记录，不是本仓库独立复测的结果。本文只引用必要的技术结论，不附带含个人信息的原 PDF。

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

教师使用 `2×4×4` patch、192 维嵌入；[`train.py`](train.py) 默认学生使用 `2×8×8` patch、192 维嵌入和 `[6, 12, 12, 6]` 注意力头。两个空间边长翻倍，使相同输入分辨率下的空间 token 数约为教师的四分之一。网络仍保留分层 Transformer、窗口注意力和地表／高空联合预测：输入为 4 个地表动态量、3 个静态地理场、65 个高空量，最终输出完整 **4 + 65 = 69** 个气象通道。

`teacher_slice_interpolate` 初始化模式将同形状权重直接继承；尺寸变动的参数按目标形状裁剪，Q/K/V 分段处理，patch 卷积核和地球位置偏置通过空间插值适配。初始化报告统计各类继承比例，低于阈值时默认拒绝继续训练。教师冻结，学生使用可拆分的全 69 通道教师损失、官方 15 通道教师损失及可选真实标签损失；代码还提供气压层加权、AdamW、预热与余弦学习率、梯度裁剪、有限值和形状检查。**当前脚本的默认损失预设是 `teacher15`**；其他实验需显式传入 `--loss-preset` 或各项损失权重，不能把 PDF 中对全通道蒸馏的描述当作所有运行的默认值。

[`makelink.py`](makelink.py) 对已有 ERA5 文件按年月日时选择起点，并为预测提前量保留连续样本，通过符号链接构建较小训练子集，不复制原始大数据。它的默认月份、日期与 `--lead-hours 24` 仅是脚本默认值，具体训练采样方案应以运行参数和数据布局为准。

### 2. 清理检查点并以低位格式存储

[`export.py`](export.py) 从完整训练检查点仅提取 `model_state_dict`，删除静态注意力 mask、位置索引、临时 buffer 和部分不在定制推理路径上的模块键；[`inference.py`](inference.py) 加载时另有 `Sampler`、`Reconvery` 等过滤规则。`export.py` 使用写死的输入输出路径，运行前应修改为自己的文件路径。它不会自动下载权重。

- **FP16**：[`conv_fp16.py`](conv_fp16.py) 将浮点权重转成 FP16；[`inference.py`](inference.py) 固定 FP16 模型执行。`conv_fp16.py` 的目标文件名虽然含 `bf16`，代码实际调用 `.half()`，产生的是 **FP16**。
- **W8A16**：[`quantize_w8a16.py`](quantize_w8a16.py) 对符合条件的多维 `.weight` 按第 0 维切片做对称 INT8 量化，默认最少 1024 个元素、scale 为 FP16；小张量和不适合的参数保留浮点。第 0 维对 `ConvTranspose3d` 并非输出通道，故更准确的说法是“逐第 0 维切片”。包装器覆盖 Linear、Conv3d、ConvTranspose3d；计算阶段使用 FP16 路径，低位权重主要节省存储和加载占用。
- **深层 W4/W5**：[`quantize_w4_storage.py`](quantize_w4_storage.py) 在已量化的 W8 检查点上，仅对 `layer2`/`layer3` 的二维 Linear 沿输入维分块重编码，默认 **4 bit、块长 16**，也可用 `--bits 5`。这是磁盘打包格式；加载时展开并可建立 FP16 权重缓存，未实现原生 INT4/INT5 GEMM。二次量化会引入额外误差。
- **位置偏置 EPB**：[`quantize_epb.py`](quantize_epb.py) 将三维 `earth_position_bias_table` 按第 0 维分块量化并保存 scale；支持 INT8、INT4 和去除偏置的消融模式。工具默认 **INT8、块长 72**；选 `--bits 4` 时默认块长 **18**，可显式覆盖。INT4 把相邻两行的 4 bit 值打包在一个字节；推理加载时一次性解包。PDF 中的 INT4“72 行分块”不是当前代码的默认设置。

这些格式允许**磁盘低位存储与运行时 FP16 稠密计算解耦**。不同打包方式的磁盘容量和运行时峰值显存不能直接等同；推理脚本还会为一些权重建立缓存。原 PDF 报告过 W8 检查点约 65.23 MB、固定 mask 唯一存储约 143.33 MB→25.21 MB；这些数字属于当次实验，仓库未附可重算的原权重或完整测量日志。

### 3. 控制激活生命周期和注意力峰值

[`inference.py`](inference.py) 的 `lean_pangu_forward` 在地表／高空嵌入、token 重排、各层和 recovery 之间尽早释放不再使用的张量引用，使 PyTorch 分配器能够复用 storage。Lean Transformer block 将 norm、window partition、attention、残差和 MLP 结果按消费顺序组织；MLP 的 GELU 可原地执行，`torch.addmm(..., out=...)` 将 `fc2` 输出写入残差目标。默认关闭 skip-token CPU offload（`PANGU_OFFLOAD_SKIP_TOKENS=0`）。

默认注意力模式为 `hybrid_qscale`：深层拆开 Q/K/V 的生成时序，使 V 不与 Q/K 长时间同时驻留；浅层以窗口宽度方向分块，仍在每个窗口内计算完整注意力。Q scale 可折叠到权重或原地计算，score buffer 在加偏置、mask 和 softmax 阶段复用。`PANGU_ATTENTION_WIDTH_CHUNK=4` 控制浅层窗口批次，默认深层不做同类分块。固定 shifted-window mask 的值域检查通过后，`int8_shared` 模式将其转为 INT8，并对内容与形状相同的 mask 共享 buffer。

默认 `PANGU_MLP_TOKEN_CHUNK=43680` 且启用 residual 输出，浅层 MLP 按 token 分块执行 `fc1 → GELU → fc2` 并写入对应残差切片；这只改变临时张量大小和 GEMM 形状。`PANGU_RECOVERY3D_CHUNK_WIDTH=45` 沿经度分块恢复全部 65 个高空通道，并未删减输出变量。

### 4. gfx936 HIP 算子与输入输出流水

[`gfx936_biasmask/`](gfx936_biasmask/) 中的 PyTorch 扩展针对 **gfx936** 编译。代码实现了 attention bias 与 INT8 mask 融合加法、固定宽度 192/384 的无 affine LayerNorm、残差加法加归一化、token gather（索引 `-1` 写零）、norm 加 gather，以及 FP16 网络输出到 FP32 反归一化输出的融合 kernel。推理加载阶段还能把 LayerNorm 的 affine 参数折叠进后继 Linear，并折叠 attention Q 缩放；这些优化均以代码中的检查与分支为前提，其他 GPU 架构不能直接假定兼容。

默认输入模式 `PANGU_STREAMED_MODEL_INPUT=1`：先组装 4 个地表量和 3 个静态场并做 2D embedding，再组装 65 个高空量做 3D embedding，释放各阶段输入后合并特征。默认输出模式 `PANGU_OUTPUT_PIPELINE=1`、`PANGU_OUTPUT_TRANSFER_CHUNK=8`，在两个 pinned CPU buffer 间轮换，使 FP16→FP32 affine、GPU 到 CPU 拷贝按通道块流水执行，最终同步后才结束计时。原 PDF 记录流式输入与原路径的 10 个完整输出逐字节一致；仓库没有附上该回归测试数据。

## 环境、使用和复现边界

比赛环境以 OneScience 的 `onescience` 包、PyTorch 2.5.1、ROCm/HIP 6.3、gfx936 为基础。需自行准备**有权使用的**教师／学生权重、ERA5 HDF5 年度数据、统计量和静态场。`conf/config.yaml` 中 `../onedatasets/ERA5_test/` 和测试年份标记保留决赛布局；它的 `model.patch_size` 仍写 `[2,4,4]`，而 [`inference.py`](inference.py) 的架构常量是 `[2,8,8]`。请按实际权重和数据管线逐项核对，不要直接把两处配置视为一致。

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

`inference.py` 采用未来 48 小时、每 6 小时一步的自回归任务流程；实际输出文件和计时受测试数据与运行参数影响。这里没有权重、数据、容器镜像或完整依赖锁定文件，**不能仅凭 clone 直接复现决赛分数**。公开前只进行了源码检查与 Python 语法检查，未在原设备上重新跑数值、延迟或显存基准。

## 来源与许可

模型背景可参见 [Pangu-Weather 论文](https://www.nature.com/articles/s41586-023-06185-3)。此代码基于赛事提供的 OneScience/Pangu 环境开发，依赖组件分别遵循其上游许可。仓库目前没有对整体代码另行授予统一许可证；若计划复用或再发布，请先核实原始基线代码及各贡献部分的授权。
