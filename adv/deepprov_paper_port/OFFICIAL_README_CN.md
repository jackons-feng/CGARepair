# DeepProv 原论文和官方源码复现（请与 Keras 移植版区分）

论文：Firas Ben Hmida 等，*DeepProv: Behavioral Characterization and Repair of Neural Networks via Inference Provenance Graph Analysis*, ACSAC 2025。

- 论文：https://arxiv.org/pdf/2509.26562
- 官方代码：https://github.com/um-dsp/DeepProv
- 原版 PyTorch 动作：https://raw.githubusercontent.com/um-dsp/DeepProv/main/artifact/library/mod_models.py
- 参考激活 / 选点：https://raw.githubusercontent.com/um-dsp/DeepProv/main/artifact/library/actions_selections.py
- 原版流水线：https://raw.githubusercontent.com/um-dsp/DeepProv/main/Claims/mnist/run.sh

**官方原代码不在本压缩包中**。为了避免将自写的近似移植误认为官方代码，应从作者仓库自行克隆并标注 commit。

```bash
cd /root/workspace/zzx/experiment1/NSrepair
mkdir -p third_party_original
cd third_party_original
git lfs install
git clone https://github.com/um-dsp/DeepProv.git
cd DeepProv
git lfs pull
git rev-parse HEAD
```

作者 README 说明需要 Git LFS 才能拿全预训练模型、图及 pickle。`install.sh` 设置 Python 3.10、PyTorch、PyTorch Geometric、TensorFlow 等，且会下载 EMBER 大文件。**切勿在现有 `nsrepair` conda 环境中直接执行**；使用独立 `deepprov-env` 环境，先检查官方 `install.sh` 后决定是否安装全部数据。

官方 MNIST 全流程 (`Claims/mnist/run.sh`) 包含：
1. 生成正常 / 对抗 IPG；
2. 训练 GNN；
3. GNN 解释器获取节点归因；
4. `artifact/main.py` 做正常参考分布与运行时激活修复。

在官方依赖准备好之后，按官方工作目录和脚本参数执行：

```bash
cd /root/workspace/zzx/experiment1/NSrepair/third_party_original/DeepProv/Claims/mnist
bash run.sh --attack FGSM
```

CIFAR-10 官方 Claim 使用其预计算图和独立脚本 `Claims/cifar10/run.sh`，不是直接套用 MNIST 路线：

```bash
cd /root/workspace/zzx/experiment1/NSrepair/third_party_original/DeepProv/Claims/cifar10
bash run.sh
```

**不能将官方 PyTorch `models/mnist_1.pth` 与用户的 Keras `model/mnist_adv.h5` 当作同一模型。** 两种结果仅可用于源方法复现 / 定性对照，不能混算为用户的五种子 Baseline。

官方原版源码包含使用测试集参与部分动作分析的写法。用户同测试协议比较时，必须在 TRAIN 上固定动作，再用 TEST 独立评估，不能直接照抄其中 TEST 选择流程。参见另一个 `README_CN.md`。
