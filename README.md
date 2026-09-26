# Pangu-Weather：2026 先导杯气象赛道决赛代码

这是俊良队在 2026 年智能计算创新设计赛（先导杯）气象大模型赛道的决赛提交代码整理版。保留模型训练、推理、量化与 gfx936 HIP 算子的实现；未包含模型权重、ERA5 数据和比赛环境。代码尚未在脱离原比赛环境的机器上完成端到端复现。

## 目录

| 路径 | 用途 |
| --- | --- |
| `train.py` | 学生模型蒸馏训练与教师参数继承 |
| `train_regression.py` | 比赛包附带的另一训练脚本 |
| `inference.py` | 自回归预测、性能优化与结果输出 |
| `quantize_w8a16.py`、`quantize_w4_storage.py`、`quantize_epb.py` | 权重及位置偏置量化工具 |
| `gfx936_biasmask/` | 针对 gfx936 的 HIP 扩展 |
| `conf/config.yaml` | 比赛提交时的数据与模型配置 |
| `conv_fp16.py`、`export.py`、`makelink.py` | 权重转换与辅助工具 |

## 方法概览

学生模型将空间 patch 由 2×4×4 调整为 2×8×8，保留分层 Transformer 与窗口注意力结构。`train.py` 支持教师参数结构化继承、69 通道蒸馏与真实标签混合监督。`inference.py` 包含 FP16 推理、低位权重存储及窗口注意力相关优化；`gfx936_biasmask/` 针对比赛用 GPU 架构实现部分 HIP 算子。

## 环境与运行

代码依赖比赛所用的 OneScience（`onescience`）、PyTorch、ROCm/HIP，以及 ERA5 数据与相应归一化统计量。配置文件中的 `../onedatasets/ERA5_test/` 是原比赛环境路径，需要按本地数据布局修改。不同 OneScience 版本可能存在 API 差异。

准备好有权使用的模型权重与数据后，可查看参数：

```bash
python train.py --help
python inference.py --help
python quantize_w8a16.py --help
```

推理命令示例（权重路径仅为占位符）：

```bash
python inference.py --config conf/config.yaml --checkpoint /path/to/your/model.pth --output-dir result/output
```

`train.py` 使用 `--teacher-checkpoint` 指定教师权重；训练参数请以 `python train.py --help` 为准。`conf/config.yaml` 保留决赛提交时的测试年份标记和数据布局，改动前请核对数据管线要求。

## 公开范围与来源

本仓库未提供权重、数据集、比赛平台下载链接和包含队员个人信息的答辩文档。代码基于比赛提供的 Pangu-Weather / OneScience 环境开发，依赖组件的许可请以其上游声明为准。此仓库暂不为整体代码另行声明统一许可证。

## 复现状态

仓库整理时完成了 Python 语法检查和文件排查；尚未在公开环境中验证训练、推理数值与决赛成绩。需要原比赛软件栈、数据和权重才能做端到端测试。
