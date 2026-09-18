# Kaggle S6E9 — Predicting Electric Vehicle Purchases

[English](readme.md) | **简体中文**

预测 `Will_Buy_EV = Yes` 的概率，评价指标为 ROC AUC。
本项目采用模块化训练流程，用统一交叉验证比较 LightGBM、XGBoost、CatBoost，
结合数字位、多尺度分箱、频率和目标编码，保存 OOF 后再选择单模型或融合。

本地完整训练结果写入
`artifacts/experiments.csv`；公开榜成绩只有实际提交后才能确认。项目不会自动向 Kaggle 提交。

## 安装与数据

建议 Python 3.12，在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

输入文件：

```text
data/train.csv              # id + 13 个原始特征 + Will_Buy_EV
data/test.csv               # id + 13 个原始特征
data/sample_submission.csv  # Kaggle 提交格式
```

训练集 668,665 行，测试集 286,571 行。目标为 `No` / `Yes`；`id` 仅用于对齐，
不会进入模型。数据加载器检查 ID 唯一性、目标值和 train/test 特征列一致性。

## 保留的架构

```text
training/
  lightlgm_cv.py       # 原有拼写和启动命令保留
  lightgbm_cv.py       # 正确拼写的等价入口
  lightgbm_reference_cv.py # 单独保留的原版 10 折 LightGBM（175fba1）
  xgboost_cv.py
  catboost_cv.py
  experiment_suite.py # 顺序执行候选，再比较融合
  ensemble.py         # 对已有 OOF bundles 做融合
utils/
  data_loader.py      # EVDataLoader，仍返回原来的五项数据
  feature_engineering.py
  cv_runner.py        # 三个模型共用分折、编码、检查点和评估逻辑
  prediction_saver.py # PredictionSaver，保留版本号与实验日志
  ensemble.py         # ID 对齐、权重选择、融合诊断
predictions/          # 仅提交 CSV
artifacts/predictions/ # OOF NPZ、配置 JSON、融合报告
artifacts/            # 逐折检查点、运行日志和进度（不提交 Git）
tests/
```

保留独立模型入口、命令行调参、提前停止、OOF AUC、分折成绩与最佳迭代统计、
预测版本命名、实验记录。重复训练逻辑收敛到 `utils/cv_runner.py`，避免三个入口
在编码和验证方式上逐渐不一致。

## 直接运行

### 原有入口

```bash
python -m training.catboost_cv
python -m training.xgboost_cv
python -m training.lightlgm_cv
```

CatBoost / XGBoost 默认仍使用原始特征与原来的主要参数；加
`--feature-engineering` 使用保留的业务特征 `legacy`。
LightGBM 默认使用 `notebook` 特征配置、10 折和三重目标编码。

```bash
python -m training.catboost_cv --depth 4 --learning-rate 0.05 --iterations 5000
python -m training.xgboost_cv --feature-engineering
python -m training.lightgbm_cv --no-target-encoding
```

### 保留的原版 LightGBM

```bash
python -m training.lightgbm_reference_cv
```

此入口基于 `175fba1` 保留原版训练配置：10 折、学习率 0.005、
最多 100000 轮、500 轮提前停止、数字位/频率特征和三重目标编码。
仍调用共享的数据/特征/预测工具（对应特征公式未改），不会被 strong preset 替换。
它与旧版本一样只在全部折完成后保存提交 CSV 和实验记录，没有新版的逐折恢复/OOF bundle。
正在运行的旧进程不受新文件影响，原有预测文件也保留。

### 新的候选方案

```bash
python -m training.lightgbm_cv --preset strong
python -m training.xgboost_cv --preset strong
python -m training.catboost_cv --preset strong
```

`strong` 是候选配置，不代表已经证明最优：默认多尺度特征、
折内频率/三重目标编码、收入邻域编码、5 折。默认 CPU 和 4 个线程。

| 参数 | LightGBM strong | XGBoost strong | CatBoost strong |
|---|---:|---:|---:|
| 最大轮数 | 3500 | 2400 | 3500 |
| 学习率 | 0.02 | 0.03 | 0.05 |
| 深度 | 5 | 6 | 6 |
| 提前停止 patience | 150 | 150 | 200 |
| 列采样 | 0.30 | 0.55 | 0.80 |

CLI 显式参数覆盖 preset。`--model-seed` 只改变模型随机性；`--random-seed` 改变分折和编码随机性。

### 完整对照与融合

```bash
python -m training.experiment_suite --n-splits 5 --n-jobs 4
```

按顺序运行：

1. 多尺度 LightGBM。
2. 数字位/频率 LightGBM，学习率改为 0.02、最大 20000 轮、提前停止 300 轮。
3. 多尺度 XGBoost。
4. 多尺度 CatBoost。
5. 用四组 OOF 比较单模型、概率加权、排名加权，生成一个候选提交。

四组使用相同的样本、分折和 split seed。Suite 的 `notebook` 候选采用学习率
0.02，与单独保留的 0.005 配置不同。整个流程可能需要数小时。

```bash
# 只运行两个候选
python -m training.experiment_suite --candidates multiscale_lgb multiscale_xgb

# 小样本冒烟检查：不能当成完整竞赛成绩
python -m training.experiment_suite --sample-size 6000 --n-splits 2 --max-iterations 20
```

## 特征方案

| recipe | 内容 | 拟合位置 |
|---|---|---|
| `legacy` | 原有充电、通勤、收入、类别交互等业务特征 | 逐行计算 |
| `notebook` | 收入/充电交互、数字位 -4…3、one-hot、频率、常量/完全相关列过滤 | 无标签预处理；频率用 train+test |
| `multiscale` | 收入数字位/余数、多种宽度的收入与通勤分箱，保留原始类别 | 逐行计算；频率仅拟合外层训练折 |

`notebook` 方案保留原始浮点整除的数字位计算方式，不能随意改成四舍五入后再拆位。
多尺度方案则显式将通勤乘 10 并取整，作为另一个不同的候选。

**标签相关特征全部在外层训练折内拟合：**

- 三重 Target Encoding：smooth = auto / 10 / 100；训练行使用内层 5 折
  `fit_transform`，验证与测试仅 `transform`。`notebook` 默认编码七个数值列；
  multiscale 编码原始列和部分分箱键。
- `--te-scope all`：对非 multiscale 方案额外编码数字位和原始类别，作为可选实验。
- `--income-neighbors`：8192 / 16384 两个收入分辨率，计算中心、邻域、左右购买率、
  斜率、曲率和样本数；训练行同样交叉拟合，缺失值回退到训练先验。

`notebook` 的 train+test 频率是使用测试分布的竞赛预处理，不使用目标标签；
它不是严格的仅训练数据预处理。其常量/相关过滤也沿用外层分折前的方式。
multiscale 的频率、类别词表和标签统计都从外层训练部分拟合。

```bash
# 同参数逐项消融
python -m training.lightgbm_cv --preset strong --no-income-neighbors
python -m training.lightgbm_cv --preset strong --no-target-encoding --no-income-neighbors
python -m training.lightgbm_cv --feature-recipe legacy --no-target-encoding
```

切换模型或特征时应保持分折一致。调参、比较多个方案都会给成绩引入选择偏差，
不能把微小提升直接解释为稳定泛化提升。

## 输出、进度与恢复

每完成一个折就保存：

```text
artifacts/runs/<model>_<signature>/
  config.json
  progress.json
  fold_01.npz
  ...
  result.json           # 全部折完成后才有
```

签名包含输入文件内容哈希、训练配置、核心代码哈希和依赖版本。相同命令再次运行
默认读取已完成折；未完成的那一折从头训练。`--no-resume` 可以重新训练。
修改源文件（包括注释）、参数或依赖可能产生新签名，避免错误复用旧检查点。
检查点保存预测，不保存可部署模型。重跑完成的任务可能生成新的提交版本。

完整 CV 结束后，文件使用相同前缀，分别保存：


代码统一管理生成文件：提交 CSV 写入 `predictions/`，元数据和 OOF 写入 `artifacts/predictions/`，实验记录写入 `artifacts/experiments.csv`，检查点和日志写入 `artifacts/runs/` 或 `artifacts/suite/`。CatBoost 自带文件日志已禁用。正在运行的旧进程保留原输出设置，新默认路径从下次启动生效。

- `predictions/*.csv`：`id,Will_Buy_EV`，可提交 Kaggle。
- `artifacts/predictions/*.npz`：训练 ID、标签、OOF 预测、fold ID、测试 ID、测试预测，供融合使用。
- `artifacts/predictions/*.json`：配置、成绩和融合报告；`artifacts/experiments.csv` 追加兼容不同模型字段的记录。

OOF bundle 缺行、重复 ID、标签不一致、折不一致或概率非法都会被拒绝。
旧版本只保存提交 CSV，不能凭空补出 OOF，需要重新训练才可用于本地验证融合。
带 `sample-size` 的运行仅为样本实验，不能与全量 OOF 混合。

Suite 另外保存 `artifacts/suite/suite_status.json` 和每个候选的 `.log`。
新版模型入口和 Suite 支持 `--train-path`、`--test-path`、`--output-dir`、
`--artifact-dir`、`--experiment-file`、`--run-root`。原版保留入口不支持逐折恢复；详细参数用 `--help` 查看。

## 融合评估的边界

```bash
python -m training.ensemble artifacts/predictions/model_a.npz artifacts/predictions/model_b.npz
```

先按 ID 对齐，要求同一训练集、标签和分折；比较单模型、等权平均以及模型两两
25% / 50% / 75% 权重，分别尝试概率和排名。权重只在固定的一半 OOF 行上选择，
另一半用于报告，不根据该报告继续调权重。单模型也可能胜出。

**这仍是 OOF 诊断，不是完全独立的嵌套验证。** 基模型在其他折训练时可能见过
另一半行的标签；全量 OOF 分数也受融合选择影响。报告明确标注
`full_oof_auc_after_selection`，不将其宣传为无偏成绩。若要严格评估融合收益，
还需额外的完全隔离测试集或完整嵌套重训。

排名融合针对 AUC，输出不是校准后的购买概率。不引入只有测试提交文件、没有
可验证 OOF 的外部预测；不做基于公开榜反复调权重或伪标签。

## 测试与当前结果

```bash
python -m unittest discover -s tests -v
```

测试覆盖目标编码交叉拟合、未见值回退、邻域编码、ID/折对齐、参数覆盖和输出
校验。小规模端到端检查覆盖四个候选、逐折续跑、预测保存及融合。

历史完整本地基线（旧实验，参数/分折未必与新实验一致）：

| 实验 | OOF AUC |
|---|---:|
| CatBoost depth=4，原始特征 | 0.9419543 |
| CatBoost depth=4，原业务特征 | 0.9417906 |
| XGBoost，原业务特征 | 0.9416650 |
| LightGBM，原业务特征 | 0.9417209 |

新增完整训练的实际成绩见 `artifacts/experiments.csv` 和 Suite 的 `result.json`。

## 2026-09-17 本地验证与交接

本次已完成代码整合并通过 11 项测试、四候选的小规模端到端训练/融合/续跑验证。
全量训练按用户要求暂停，交由用户继续执行：

```bash
source .venv/bin/activate
python -m training.experiment_suite --n-splits 5 --n-jobs 4
```

上述命令与已启动的训练配置一致；仅在签名也一致时读取完整折检查点，未完成折重新训练。
无需删除 `artifacts/suite/`。代码或依赖发生变化时签名可能改变，旧折不会误用。

| 已完成的全量实验 | 外层折数 | OOF AUC | 提交版本 |
|---|---:|---:|---|
| 单独保留的原版数字位/频率 LightGBM | 10 | 0.94580 | v017 |
| 新增多尺度 + 邻域 + 三重 TE LightGBM | 5 | 0.94612535 | v018 |

两者折数不同，仅作记录，不能据此认定稳定提升。统一 5 折的 `notebook` 对照、
XGBoost、CatBoost 和最终融合尚未全部完成，不能宣称融合成绩已经提升。

现有两份完整提交都在本地 `predictions/` 中。OOF bundle 若已删除，
在配置、代码和依赖签名一致时，可从保留的完整折检查点重新生成。
原版脚本永久单独保存在 `training/lightgbm_reference_cv.py`。

已确认的公开榜成绩：v018 为 **0.94639**（用户于 2026-09-17 提供的 Kaggle 提交结果）。

## 致谢

本项目的 `notebook` 特征方案和单独保留的参考 LightGBM 配置，改编自 Rugved Bane 的 Kaggle Notebook：

["0.94590 LB — Stacking failed, this didn't"](https://www.kaggle.com/code/rugvedbane/0-94590-lb-stacking-failed-this-didn-t)

其中，以下方法受到该参考方案的启发：

- 数字位拆分
- 频率编码
- 三重目标编码
- 参考 LightGBM 配置

本仓库在该基线之上扩展了可复用的交叉验证流程、防止标签泄漏的折内特征拟合、
OOF 产物、检查点与断点续跑、多尺度特征、收入邻域编码、模型对比、
融合诊断和自动化测试。

## 基于 v18 的下一轮实验

v20 复现了 v18，可直接从检查点恢复缺失的 OOF，无需训练：

```bash
python -m training.recover_oof artifacts/suite/multiscale_lgb/lightgbm_0f835db60a41eb8c
```

恢复时校验数据哈希、折划分和原提交预测，仅支持已完成的全量实验。

```bash
# No model training: compare v20 (= v18) with small v22 weights.
python -m training.ensemble artifacts/predictions/prediction_v020_*.npz artifacts/predictions/prediction_v022_*.npz --challenger-weights 0.1 0.2 0.3

# Optional full training: same folds and features, different model seeds.
python -m training.lightgbm_cv --preset strong --n-splits 5 --random-seed 42 --model-seed 17 --n-jobs 4
python -m training.lightgbm_cv --preset strong --n-splits 5 --random-seed 42 --model-seed 2026 --n-jobs 4
```

第一条命令不训练，只比较基准与加入少量 v22 的融合。后两条会完整训练：保持折划分和特征一致，仅更改模型种子。提交前查看融合报告的 `audit_gain_over_baseline`；它只是 OOF 诊断，并非独立的嵌套验证，不保证涨分。种子实验生成新版本后，用对应 NPZ 的完整路径传给 `training.ensemble`，与基准比较。

## 扩大分箱目标编码范围

多尺度方案新增 `--te-scope bins`：对全部 7 个收入分箱和 4 个通勤分箱做三重目标编码，增加 24 列 TE。频率特征、内层交叉拟合、模型参数和外层划分不变。默认 `numeric` 保留原基线。这是待验证的实验，不代表已提升成绩。

```bash
python -m training.lightgbm_cv --preset strong --te-scope bins --n-jobs 4
```

## OOF review

以 v027 为基准复盘已有 OOF，不训练、不搜索新权重、不生成提交。报告包含逐折 AUC 差异、预测相关性，以及不兼容文件的跳过原因。重复使用的验证数据仅用于回顾诊断，不能据此宣称统计显著或独立验证提升。

```bash
python -m training.review_experiments
```

`artifacts/reviews/summary.csv`, `artifacts/reviews/folds.csv`, `artifacts/reviews/comparison.json`

独立折结构诊断，不修改主实验的 seed 42：

```bash
python -m training.diagnose_folds
```

[Fold diagnosis report](docs/fold_diagnosis.md). `--probe` runs 25 bounded diagnostic models across five CV seeds. Reports: `artifacts/fold_diagnostics/`.

## 可选：原始数据统计特征

使用 itzzomkar 发布的[原始 EV 数据](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety)（CC0、版本 1），新增 13 个平滑目标统计特征。编码器只读取外部标签，并先排除与比赛训练、测试集特征完全重叠的外部行。默认关闭，保留原来的五折、seed=42 和模型配置。目前尚未完成全量训练，不能认定会提分。

```bash
python -m training.prepare_original_data
python -m training.lightgbm_cv --preset strong --original-data --n-jobs 4
```

本机已准备好数据，可直接运行第二条。原始 CSV 放在 `data/external/`，审计报告放在 `artifacts/external_data/`，不会混入提交目录。完整核查和对照方案见[实验说明](docs/original_data_experiment.md)。
