# 纽约 HVFHV 补充实验代码

本文件夹按 R2 稿件 Table 7 保留的五种纽约实验方法整理：

1. 固定参数 XGBoost（XGB）
2. 未增广的 BO-XGB
3. SMOTE-BO-XGB，代码具体使用 SMOTENC 处理类别特征
4. CTGAN-BO-XGB
5. PCTGAN-BO-XGB

实验设置对应 2026 年 8 月开发数据、固定的 5,000 条 8 月测试数据，以及 205,374 条 7 月外部评估数据。生成器损失消融实验和 July 500 人工审阅工作簿不属于本版本；人工审阅数据的版本问题见 [MANUSCRIPT_ALIGNMENT.md](MANUSCRIPT_ALIGNMENT.md)。

## 模块划分

- `run.py`：统一运行入口，支持预测、数据划分、BO-XGB、GAN 训练与采样、SMOTENC。
- `preprocess.py`：从月度原始文件恢复冻结样本并校验构造特征。
- `reproduce_clustering.py`：复现冻结的 BMI2-DBSCAN 聚类配置。
- `code/data.py`：字段校验、纽约特征构造、OD 基线与派生变量。
- `code/behavior_features.py`：聚类复现使用的五维行为表示。
- `code/diagnostic_sampling.py`：候选标签与行为区间边界规则。
- `code/memory_dbscan.py`：内存受限的 DBSCAN 实现。
- `code/augmentation.py`：PCTGAN 约束变换与重构、SMOTENC 增广、随机种子设置。
- `code/joint_conditional_gan.py`：CTGAN/PCTGAN 共用的按类别条件化训练实现。
- `code/search.py`：BO-XGB 搜索空间定义。
- `data/`：冻结后的 8 月队列、7 月外部集、基线统计、标签映射和数据字典。
- `models/`：分类器、预处理器和聚类配置。
- `results/`：五随机种子汇总指标、逐种子指标和参数结果。

文件夹包含 seed 42–46 的 CTGAN 与 PCTGAN 训练检查点，可直接用来生成样本；完整月度原始 Parquet 和生成后的完整训练表不纳入 Git。原始文件可从 [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) 获取，放到 `data/raw/`，文件名为：

```text
fhvhv_tripdata_2026-07.parquet
fhvhv_tripdata_2026-08.parquet
```

运行预测：

```powershell
python -m venv .venv_eval
.venv_eval\Scripts\python -m pip install -r requirements-evaluation.txt
.venv_eval\Scripts\python run.py verify
.venv_eval\Scripts\python run.py predict --out runs/predictions
```

复现每个随机种子的分层 80/20 训练/BO 验证划分：

```powershell
.venv_eval\Scripts\python run.py split --out runs/splits
```

GAN 环境安装方式和完整重训命令见英文版 README。若直接复用随包提供的 GAN 检查点，可运行：

```powershell
.venv_gan\Scripts\python run.py sample --families ctgan pctgan --out runs/generated
```

完整重训与增广命令见英文版 README。所有输出目录都必须为空或尚不存在。

## 结果解释边界

8 月聚类标签和 7 月规则标签是订单模式候选标签，不是经过独立核实的欺诈事实。由于纽约数据没有原论文中的远程接驾附加费字段，第 4 类使用“请求至上车等待时间较长”作为代理。8 月取整先验反映的是候选样本中的类别比例，不代表纽约全体订单或欺诈行为的自然发生率。

当前没有替项目选择软件许可证；公开 GitHub 前还需确认衍生样本的再分发许可，并添加团队选择的许可证。
