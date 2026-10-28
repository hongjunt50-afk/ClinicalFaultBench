# ClinicalFaultBench

面向医疗 AI 智能体的失效审计基准:在坏数据、坏工具、坏时序下,系统性度量临床大模型智能体的失败方式、错误传播与不确定性表达。不只测它能不能做对,更测它怎么失败。

> 状态:**方案 v2.0(2026-10-26 重设计)**——因 MIMIC-IV 全量的 DUA 与地区访问限制,主实验数据源改为**纯合成 Synthea**;详见 `docs/design-v2.md`。差异化声明见 `docs/novelty-claim.md`。
> **首批实验结果(deepseek-v4.1-flash,90 任务 + 140 格故障矩阵)见 [`results/RESULTS.md`](results/RESULTS.md)**,含上一版五条结论的作废说明与六个已修判分缺陷的清单。

## 一句话结论(2026-10-28)

信息缺失(F5 结果截断)是最致命的单一故障,降 **0.389**,且 mild 与 severe **完全相等**——缺 30% 还是缺 70% 对被测模型没有区别,因为它从不确认数据是否完整。反之,禁忌冲突(F3)检出率 100%、效应 ≈ 0,是必须如实报告的饱和负结果。

## 核心设计

- **数据源三档解耦**(v2 核心架构):T1 Synthea 纯合成(主实验)/ T2 MIMIC-IV demo ODbL 开放小样本(可选外部校验)/ T3 院内私有数据(只跑不传)。代码只依赖"病程时间线" schema,不依赖数据源
- **FaultProxy**:对被测方完全透明的故障注入代理层,注入七类临床语义故障(数据层 F1–F4 / 工具层 F5–F6 / 时序层 F7),每类两档严重度。**每类故障声明适用任务**(`faults.APPLICABLE`):往不判该通道的任务上注入,效应必然为 0,那不是"模型免疫"而是设计缺陷
- **五项审计指标**:注入前后配对错误率(核心)、错误传播率、安全违规率、不确定性校准、自纠率
- **纵向病程三件套**:入院摘要审查、医嘱冲突检测、出院准备;规则判分为主,LLM-judge 仅辅助
- **中文化适配层**:检验/药品中文通用名 + 诊断章节中文名 + 中美单位制换算(见下)

## 为什么用合成数据(合规优先)

MIMIC-IV **demo** 本身是 Open Access + ODbL(可再分发),但全量数据受 *PhysioNet Credentialed Health Data License* 约束(明文禁止共享),且 PhysioNet 对大陆 IP 有地区访问限制。因此 v2 把主实验完全建在 **Synthea**(Apache-2.0 纯合成患者)上:零合规风险、可无限扩、可公开复现。

同时这也是国内落地的真实需求形态:医院数据不能出院,评测工具必须**可移植到数据方环境运行、只回传聚合指标**——这套设计本身就是论文卖点。

## 国内适配与 v2 路线图(2026-10-25 拍板)

国内无公开纵向结构化 EHR(公开临床数据集均为文本/对话类),v1 数据源保持 MIMIC;"适配国内"落在术语层,数据源级泛化走 v2:

- **中文化适配层**(`src/localization.py` + `configs/terminology_zh/`):检验项目/药品的中文通用名、诊断 ICD-9/10 章节中文名、中美单位制换算表。映射表只含通用医学术语(不触 DUA);demo 数据实测覆盖率:检验事件加权 89.2%、药品医嘱加权 86.9%、诊断章节 100%,未覆盖项由 `python src/localization.py --coverage` 持续暴露。时间线 JSON 中检验/医嘱/诊断自动附加 `label_zh`/`drug_zh`/`chapter_zh` 字段。
- **F2 故障获得真实语义**:MIMIC 用美制单位(mg/dL),国内报告单用 SI 制(mmol/L/µmol/L)——`unit_conversions.csv` 的换算系数就是 F2"单位错误"故障的真实原型:国内系统接入国外/遗留数据时系统性发生的单位错位,而非人工臆造的数值扰动。
- **v2(数据源级泛化)**:FaultProxy 的注入函数是数据无关纯函数 `(clean, spec)→corrupted`,换数据只换加载器。等国内真实数据渠道(医院合作 + 伦理审批)走通后,在脱敏中文 EHR 上做泛化验证——直接回应论文最大 limitation(MIMIC 欧美 ICU 人群外推性)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `src/` | 管线代码(数据加载 / 术语适配 / 任务构造 / FaultProxy / 判分 / 出图) |
| `configs/` | 实验配置(任务集 `tasks.jsonl`、禁忌规则、参考区间、随机种子冻结) |
| `configs/terminology_zh/` | 中文化术语映射表(检验/药品/ICD 章节/单位换算,通用医学知识,可再分发) |
| `data/` | 本地数据目录(gitignore,放 Synthea CSV) |
| `docs/` | 项目介绍、差异化声明、v2 方案 |
| `results/` | 实验产出(`RESULTS.md` 主报告 / 主图 / 原始 jsonl) |
| `tests/` | 判分器与故障注入的回归测试(`python -m unittest discover tests`) |

## 复现

```bash
python src/synthea.py --build-timelines     # Synthea CSV → 病程时间线
python src/pool.py                          # 分层抽样出任务池
python src/tasks.py --build                 # 生成 90 个任务实例(T1/T2/T3 各 30)
python src/tasks.py --score-toy             # 判分链路自检(作弊 agent 必须满分)
python -m unittest discover tests           # 23 个回归用例
python src/run.py --model <model>           # 干净基线
python src/matrix.py --model <model> --target-per-fault 10   # 故障矩阵
python src/rescore.py                       # 改了判分器后离线重判,不花钱
python src/figures.py --model <model>       # 出图
```

## 合规红线

MIMIC 数据使用协议(DUA)禁止再分发原始数据。本仓库只包含代码、构造脚本与派生任务描述,`data/` 与 `results/examples/` 已在 `.gitignore` 中,不落任何原始或派生患者级数据。

## License

MIT
