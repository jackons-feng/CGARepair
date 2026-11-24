# DeepProv 论文机制 Keras 移植版（原版 GNN 不包含在此移植中）

## 结论先行
**上一次 `weights_final_deepprov.h5` 版本不是 DeepProv 原论文的等价复现。** 它只缩放下一层的权重，无法体现论文的前向激活干预。作者官方 `mod_models.py` 在模型 `forward()` 中对选定神经元执行：

`x'[eligible] = x[eligible] - alpha*(x[eligible]-reference_benign)`，其中 `eligible = selected and abs(delta)<beta`。

此代码移植这个计算，输出含 `DeepProvPaperActOn` 的 **完整 Keras 模型** `repaired_model.h5`，不要再用 `model.load_weights('weights_final_deepprov.h5')` 来测试本方法。

## 与原论文仍有区别（必须在论文中说明）
- 官方方法使用 PyTorch IPG + 预训练 GNN explainer 的结构归因；此移植因用户的 CGARepair 模型是 Keras `.h5`，使用输入相关的 `abs(activation * d true_class_score/d activation)` **梯度代理**，因此不等价于官方 GNN 归因。
- 参考分布的 KDE 使用固定 RNG 的 NumPy 实现；对于 Conv2D 做通道级 KDE，不是作者所有空间节点的图分布。
- β 使用在 TRAIN 采样集上得到的同索引激活差均值，**不是数据集自带逐图匹配证明**。
- 双目标 TS 为 `(new clean - original clean) + (new adv - original adv)`；按当前修复任务允许 `--max-clean-drop 0.03`，默认单个动作和累计动作的 TS 必须改善。动作在 TRAIN 选定，TEST 不选任何动作。
- 不能把本移植的结果标注为严格复现的 **DeepProv (official)**；应标作 **DeepProv activation-intervention port (ours)** 或继续使用作者原版完成模型级复现。

## 安装（不覆盖旧结果）
把 ZIP 上传服务器 `/root/workspace/zzx/experiment1/NSrepair/`，然后：

```bash
cd /root/workspace/zzx/experiment1/NSrepair
unzip -o DeepProv_PaperMechanism_Keras_Port_20261030.zip -d cfrepair_adv
cd cfrepair_adv/deepprov_paper_port
python smoke_tf.py
```

`smoke_tf.py` 是关键：会检查零干预下与原模型完全一致、插入动作后输出确实变化、保存/重新加载后仍然有效。报错则**停止批量实验并提供报错日志**，不要继续生成五种子论文表。

## 单种子：MNIST ε=0.05；CIFAR-10 ε=0.1（按当前 E2 审计配置）

```bash
cd /root/workspace/zzx/experiment1/NSrepair/cfrepair_adv/deepprov_paper_port
python -u deepprov_train.py mnist 42 --epsilon 0.05 \
  --loc-sample 256 --eval-sample 768 --candidate-pool 32 --max-actions 24
python -u deepprov_test.py mnist 42 --epsilon 0.05
python -u deepprov_train.py cifar10 42 --epsilon 0.1 \
  --loc-sample 256 --eval-sample 768 --candidate-pool 32 --max-actions 24
python -u deepprov_test.py cifar10 42 --epsilon 0.1
```

当前 `--root` 默认 NSrepair 项目根目录，读取模型：

`/root/workspace/zzx/experiment1/NSrepair/model/mnist_adv.h5`

`/root/workspace/zzx/experiment1/NSrepair/model/cifar10_adv.h5`

不自动回退到 `mnist.h5` / `cifar10.h5`，防止错误初始模型。

TRAIN 的对抗数据预期在：

`data/adv/mnist/train/epsilon0.05.{txt,npy}` 以及 `data/adv/cifar10/train/epsilon0.1.{txt,npy}`，各目录配套 `label.{txt,npy}`。

TEST 在 `data/adv/{mnist,cifar10}/test/...`。MNIST 使用原始 [0,1] 规范；CIFAR-10 使用 `/255` 后减均值除标准差（与用户 `adv_test_2.py` 基本协议一致），读取器会报告实际处理路径和转换，不能把 CIFAR-10 的 ε=0.1 与历史 ε=0.05 记录混合统计。

## 运行五种子

```bash
bash run_5seeds.sh both
python summarize_deepprov_port.py
```

五种子：42、123、256、1024、2026。运行脚本按任务自动指定 MNIST 0.05、CIFAR-10 0.1；两个任务不能共用同一 epsilon。对于已有 `train_manifest.json` 或 `deepprov_port_test_results.json` 的种子默认跳过。需要重新运行应先单独备份相应种子结果，再手动加 `--overwrite`；勿自动覆盖旧的 `deepprov` 历史文件。

## 输出示例

`cfrepair_adv/loc_result/mnist/42/deepprov_paper_port_eps0.05/`

- `repaired_model.h5`：完整 Keras 模型，包含非训练 `DeepProvPaperActOn` 修复层；
- `train_manifest.json`：只从 TRAIN 确定候选 / 参考 / β / 动作 / 参数；模型、权重、TRAIN 文件的 SHA256；
- `deepprov_port_test_results.json`：固定动作的独立完整 TEST，10000 clean + 10000 adv；
- `deepprov_paper_port_summary/per_seed.csv` / `means.csv`：按五种子汇总，只有全部五种子 TRAIN 可行且 TEST 完成才标记统计完整。

测试加载完整模型时：

```python
from deepprov_runtime import DeepProvPaperActOn
model=tf.keras.models.load_model('repaired_model.h5',compile=False,
    custom_objects={'DeepProvPaperActOn':DeepProvPaperActOn})
```

不能用 `model.load_weights(...)` 去加载这个模型作为激活修复。

## 出错处理
- `[ZERO-MASK PARITY]` 不通过：说明 Keras 图克隆改变了原模型语义，必须停止；
- `[NO TRAIN FEASIBLE ACTION]`：没有通过预设 TRAIN Tradeoff 门槛的动作，报告为未找到可行修复，不得改用 TEST 选动作；
- `MODEL input mismatch` / `[TEST SOURCE]` 错：先核对 CIFAR-10 是否已经标准化，不能在运行脚本内静默更换输入；
- 本代码在交付环境中已通过纯 NumPy 算法测试及 Python 语法校验，**尚未在用户真实旧版 TensorFlow 环境执行完整 H5 模型测试**，所以务必先 `smoke_tf.py`。

## 学术口径
官方原论文：https://arxiv.org/pdf/2509.26562
官方源码：https://github.com/um-dsp/DeepProv
建议论文在 Baselines/Implementation Details 中把官方 DeepProv（PyTorch）和同任务 Keras 适配版区分，不能声称本移植已复现官方 IPG/GNN 全流程。

## 2026-10-30 纠错说明
- 最初压缩包的 `run_5seeds.sh` 错误地给 CIFAR-10 传入 0.05，现已修复为 0.1。
- `deepprov_train.py` / `deepprov_test.py` 在省略 `--epsilon` 时亦采用数据集专属默认值。
- `summarize_deepprov_port.py` 现在读取各自的目录：MNIST `eps0.05`，CIFAR-10 `eps0.1`，并检查 JSON 中的 epsilon，避免混算。
- 已有 CIFAR-10 `eps0.05` 文件如存在，请保留为历史实验，不要重命名或合并进 `eps0.1`。
- 正式投稿前确认 RQ2、RQ4 和 Baseline 全部对应相同的攻击强度与同一 TEST 协议；此前部分历史实验明确使用 CIFAR-10 0.05。
