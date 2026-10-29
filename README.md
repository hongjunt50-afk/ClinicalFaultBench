# ClinicalFaultBench

面向医疗 AI 智能体的失效审计基准:在坏数据、坏工具、坏时序下,系统性度量临床大模型智能体的失败方式、错误传播与不确定性表达。不只测它能不能做对,更测它怎么失败。

> 状态:**方案 v2.0(2026-10-26 重设计)**——主实验数据源为**纯合成 Synthea**;数据架构与合规约束见 `docs/design-v2.md`,差异化声明见 `docs/novelty-claim.md`。
> **首批实验结果(两个模型,90 任务基线 + 140/98 格故障矩阵)见 [`results/RESULTS.md`](results/RESULTS.md)**,含上一版五条结论的作废说明与六个判分缺陷、两处设计缺陷的清单。

## 一句话结论(2026-10-28)

信息缺失(F5 结果截断)是最致命的单一故障,降 **0.389**,且 mild 与 severe **完全相等**——缺 30% 还是缺 70% 对被测模型没有区别,因为它从不确认数据是否完整。反之,禁忌冲突(F3)检出率 100%、效应 ≈ 0,是必须如实报告的饱和负结果。

## 核心设计

- **数据源三档解耦**(v2 核心架构):D1 Synthea 纯合成(主实验)/ D2 MIMIC-IV demo ODbL 开放小样本(可选外部校验)/ D3 院内私有数据(只跑不传)。代码只依赖"病程时间线" schema,不依赖数据源。任务类型记作 T1/T2/T3(入院摘要审查 / 医嘱冲突审查 / 出院准备),与数据档位是两套编号
- **FaultProxy**:对被测方完全透明的故障注入代理层,注入七类临床语义故障(数据层 F1–F4 / 工具层 F5–F6 / 时序层 F7),每类两档严重度。**每类故障声明适用任务**(`faults.APPLICABLE`):往不判该通道的任务上注入,效应必然为 0,那不是"模型免疫"而是设计缺陷
- **五项审计指标**:注入前后配对错误率(核心)、错误传播率、安全违规率、不确定性校准、自纠率
- **纵向病程三件套**:入院摘要审查、医嘱冲突检测、出院准备;规则判分为主,LLM-judge 仅辅助
- **中文化适配层**:检验/药品中文通用名 + 诊断章节中文名 + 中美单位制换算(见下)

## 为什么用合成数据(合规优先)

MIMIC-IV **demo** 本身是 Open Access + ODbL(可再分发),但全量数据受 *PhysioNet Credentialed Health Data License* 约束(明文禁止共享),且 PhysioNet 对大陆 IP 有地区访问限制。因此 v2 把主实验完全建在 **Synthea**(Apache-2.0 纯合成患者)上:零合规风险、可无限扩、可公开复现。

同时这也是国内落地的真实需求形态:医院数据不能出院,评测工具必须**可移植到数据方环境运行、只回传聚合指标**——这套设计本身就是论文卖点。

## 国内适配

国内无公开纵向结构化 EHR(公开临床数据集均为文本/对话类),"适配国内"落在术语层:

- **中文化适配层**(`src/localization.py` + `configs/terminology_zh/`):检验项目/药品的中文通用名、诊断 ICD-9/10 章节中文名、中美单位制换算表。映射表只含通用医学术语,不含任何原始数据。时间线 JSON 中检验/医嘱/诊断自动附加 `label_zh`/`drug_zh`/`chapter_zh` 字段。
- **F2 故障获得真实语义**:美制单位(mg/dL)与国内报告单 SI 制(mmol/L/µmol/L)之间的换算错位,`unit_conversions.csv` 的换算系数就是 F2"单位错误"故障的真实原型——国内系统接入国外/遗留数据时系统性发生的单位错位,而非人工臆造的数值扰动。
- **待办**:Synthea 的药品命名与 MIMIC 词表差异极大(实测覆盖率 1.8%),映射表需按 LOINC 码重建,见 `docs/design-v2.md` §3。


## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `src/` | 管线代码(数据加载 / 术语适配 / 任务构造 / FaultProxy / 判分 / 出图) |
| `configs/` | 实验配置(任务集 `tasks.jsonl`、禁忌规则、参考区间、随机种子冻结) |
| `configs/terminology_zh/` | 中文化术语映射表(检验/药品/ICD 章节/单位换算,通用医学知识,可再分发) |
| `data/` | 本地数据目录(gitignore,放 Synthea CSV) |
| `docs/` | 设计说明(`design-v2.md`)、差异化声明(`novelty-claim.md`) |
| `results/` | 实验产出(`RESULTS.md` 主报告 / 主图 / 原始 jsonl) |
| `tests/` | 判分器与故障注入的回归测试(`python -m unittest discover tests`) |

`src/timeline.py` 是 D2(MIMIC-IV demo)适配器,需要额外装 `duckdb`;主实验全程走 Synthea,
不装也能跑完上面所有不含 D2 的命令(缺 duckdb 时只有该脚本会给出提示并停止)。

## 复现

前置:Python 3.12+、JDK 17 + [`synthea.jar`](https://github.com/synthetichealth/synthea)(生成合成
数据用),以及任意 OpenAI 兼容接口的密钥。**只跑纯规则部分(任务构建、判分自检、单元测试)
不需要任何 API key。**

密钥放法:在仓库根建一个 `.env`(已在 `.gitignore` 中,不要提交),每行 `KEY=VALUE`,至少两项:

```
RELAY_BASE_URL=https://<你的中转或官方地址>/v1
RELAY_API_KEY=<你的密钥>
```

也可用环境变量 `RELAY_ENV` 指向别处的同名文件。`src/run.py` 只读取这两个键,其余逻辑与厂商无关。

```bash
# 0) 生成合成数据(只需一次)
java -jar synthea.jar -p 1200 -s 20261026 --exporter.csv.export=true \
     --exporter.baseDirectory=data/synthea
python src/synthea.py --check                # 自检:规模/排序/完整性断言

# 1) 纯规则部分,不花钱
python src/pool.py --build                   # 分层抽样 → configs/task_pool.json
python src/tasks.py --build                  # 生成 90 个任务实例(T1/T2/T3 各 30)
python src/tasks.py --score-toy              # 判分链路自检(作弊 agent 必须满分、空答案必须 0 分)
python -m unittest discover tests            # 23 个回归用例
python src/faults.py --selftest               # 七类故障 × 两档,断言定向性与可复现

# 2) 调模型(产生 API 费用)
python src/run.py --model <model>                                         # 干净基线 90 任务
python src/matrix.py --model <model> --target-per-fault 10                # 故障矩阵
python src/retest.py --model <model> --per-type 3                         # 重测噪声下限

# 3) 后处理(不花钱)
python src/rescore.py --model <model>        # 改过判分器后离线重判,不重跑模型
python src/figures.py --model <model>        # 出图
```

单任务成本差异很大(实测 deepseek-v4.1-flash ≈1.0 credit、kimi-k3.1-pro ≈15.0 credit,
后者是前者的 14.6 倍),跑矩阵前先用 `--limit` 或 `--budget` 探单价。

## 合规红线

本仓库只包含代码、构造脚本与派生任务描述。`data/` 与 `results/examples/` 已在 `.gitignore` 中,
不落任何原始或派生患者级数据;主实验数据由 Synthea 在本地生成,仓库里没有任何受限数据。
受限数据(如凭证访问的 MIMIC-IV 全量)不参与本项目任何环节。

## 作者与引用

作者:涂鸿峻、ZCode。

```bibtex
@misc{clinicalfaultbench2026,
  title  = {ClinicalFaultBench: A Failure-Audit Benchmark for Clinical AI Agents},
  author = {涂鸿峻 and ZCode},
  year   = {2026},
  url    = {https://github.com/hongjunt50-afk/ClinicalFaultBench}
}
```

## License

MIT
