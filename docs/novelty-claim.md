# ClinicalFaultBench · 差异化声明(novelty claim)

- 版本:v1(2026-10-29)——四篇并行精读完成;数据源已按 v2 架构改为 Synthea 合成患者,claim 措辞同步更正。
- 精读对象:MedAgentBench v1+v2 / AgentClinic / ReliabilityBench / ChaosLLM(要点见 §3,证据链见 §7)。
- M0.3 验收("能一句话回答与 ReliabilityBench 何异"):见 §2。

## 1. 三句 claim

**Claim 1|临床语义故障分类学**:首个面向临床 agent 的透明故障注入分类学——七类故障跨数据/工具/时序三层(数据层 F1 关键化验缺失、F2 单位错误、F3 矛盾记录、F4 陈旧数据;工具层 F5 接口超时/截断、F6 跨患者污染;时序层 F7 时间线漂移),每类两档严重度,设计标准是"看起来合理但错"。既有故障注入方法论(ChaosLLM、ReliabilityBench)只覆盖 API/接口层通用失效,不带任何临床语义。

**Claim 2|患者安全导向度量**:五项审计指标——注入前后配对错误率(核心)、错误传播率、安全违规率、不确定性校准(该拒答时的拒答率/过度自信率)、自纠率。首次把"错得多危险、敢不敢说不知道"做成可机器判定的基准维度;能力基准只给成功率/准确率。

**Claim 3|纵向病程任务**:首个建在纵向病程数据上的 agent 失效审计;三件套任务(入院摘要审查 / 住院医嘱冲突检测 / 出院准备)要求跨时间整合信息,区别于单次就诊短操作(MedAgentBench,≤3 步)与单 encounter 问诊(AgentClinic)。主实验用纯合成 Synthea 患者(可无限扩、完全公开可复现),MIMIC-IV demo 保留为可选外部校验。

## 2. 一句话差异(M0.3 验收答案)

**与 ReliabilityBench 何异**:它在合成的通用工具环境里注入接口层故障,测"agent 能不能恢复";我们在纵向病程上注入"看起来合理但错"的临床语义故障,测"会不会害人(安全违规)、错误怎么传播、敢不敢承认不知道(校准)"——临床语义故障、患者安全度量、纵向病程,三样它一样都没有。

## 3. 撞车判定(按预写标准执行)

**主 claim:不收窄,维持三件套。**

| 对象 | 它覆盖了什么 | 威胁 | 处置 |
| --- | --- | --- | --- |
| ReliabilityBench(arXiv 2601.06112) | 接口/基础设施层 9 类故障(超时/重置/限流/截断/schema 漂移/缓存陈旧/空响应),四域全通用(scheduling/travel/support/e-commerce),全合成 mock 环境,零医疗任务;度量=可靠性曲面 R(k,ε,λ) | 低——唯一重叠是"透明注入代理"这一方法论同型 | related work 显式切割;其 "StaleData" 是**缓存不一致**,与我们的 F4"陈旧检验值用于当前决策"名字撞、实质不同,必须写明 |
| ChaosLLM(ISSREW 2025,4 页工作坊短文) | 仅 4 类故障(unreachable/slow/hang/incorrect),第 4 类是领域无关值扰动(数值 delta/字符操作);5 个单步微任务;度量只有 TSR/幻觉率/超时率 | 低 | 引为 motivation 与架构先例(声明式 campaign、transformer 式输出扰动、逐运行子进程隔离);其"移植到 MCP 层"的 future work 恰是我们的落点 |
| AgentClinic(npj Digital Medicine 2026) | 拟真多 agent 问诊环境(patient/doctor/measurement/moderator);**无数据噪声注入**——注入的是 23 类认知偏差到 system prompt(persona 层,不动数据);v5 已用 MIMIC-IV 200 例(单诊断、单 encounter)+ NEJM 120 例 | 中——占了"临床 agent + 真实 EHR"叙事高地 | 必引;差异=它测"拟真环境下诊断准不准",我们测"故障下怎么失败";FaultProxy 设计为可叠加在同类环境之上 |
| MedAgentBench v1+v2(NEJM AI 2025 / PSB 2026) | FHIR 虚拟 EHR、100 患者 300 任务、多轮带工具(9 函数/8 轮上限);**无任何故障注入**,度量仅成功率(pass@1) | 低 | 引为能力饱和论据 + 任务形态/判分协议参照 |

**新发现的撞车候选(本次精读扫出,原方案未列)**:

- **arXiv 2610.21270「Escalation Reliability in Healthcare LLM Agents」——最强部分重叠,必引**。对医疗 agent 的输入做 missing/stale/conflicting/late 四轴扰动并测校准(ECE/Brier)。与我们差异:全合成手写种子、单一升级决策任务(非纵向三件套)、无工具环境注入、无跨患者污染/单位错误/工具层故障、无错误传播率与安全违规率。它占掉"输入扰动+校准"小角,但 squeeze 不到我们的组合;claim 措辞保持"透明代理层 × 七类含工具层与跨患者污染 × 纵向病程三件套 × 注入前后配对安全五指标"。
- 其余扫描(医疗多智能体审计 arXiv 2510.10185、agent 审计医疗模型 arXiv 2507.05755、MedBench/HealthBench 等):均为能力评测、安全评测或纯观测式研究,无故障注入。**未发现直接撞车**。

## 4. 关键事实修正(影响对外口径)

1. **"顶尖模型 98%"措辞不准,必须修**:98.0% 是 PSB 2026 v2 中 **GPT-4.1 + 记忆脚手架**的工程成绩(无记忆 91.0%);裸跑最优仍是 Claude 3.5 Sonnet v2 的 **69.67%**(v1)。项目介绍书与 KB 卡位里"顶尖模型成功率已达 98%"应改为"带记忆脚手架的工程最优 98%、裸跑 69.67%"。这反而强化论点:连 98% 的成功也是建立在干净输入上的。
2. **do-nothing 攻击即卖点**:OpenReview 已有批评指出 v2 上什么都不做的 agent 可得 42%。我们的规则判分下 do-nothing = 0 分,主动写进论文。
3. **AgentClinic 正确 arXiv 号是 2405.07960**(2405.07160 是数学论文);正式引用用 npj Digital Medicine 2026, vol.9:499。

## 5. 留给设计的三个决定(从精读直接来)

- FaultProxy 拦截点选在**工具/数据接口边界**(decorator 包 tool.run(),被测方零改码;ChaosLLM 同款手法),故障落成声明式 spec + 输出 transformer;MCP 层是其 future work、我们的差异化落点。
- 判分协议要像 MedAgentBench 一样交代清楚(参考答案构造+规则校验),M1.3"判分器人工核对一致率 ≥95%"验收就是为此设的。
- AgentClinic 的 23 类偏差注入可作为 v2 的正交轴(偏差 × 数据故障),v1 不做,related work 点一句即可。

## 6. 国内适配与 F2 的语义来源(2026-10-25 定,2026-10-26 随 v2 调整)

国内无公开纵向结构化 EHR(公开临床数据集均为文本/对话类),"适配国内"落在**术语层**:新增中文化适配层(`src/localization.py` + `configs/terminology_zh/`),检验项目/药品中文通用名 + ICD 章节中文名 + 中美单位制换算。数据源级泛化(真实国内 EHR + 伦理审批)列入 v2 路线。

**对 claim 的实质增强**:美制单位(mg/dL)与国内报告单 SI 制(mmol/L/µmol/L)之间的换算错位,使 F2"单位错误"从人工臆造的数值扰动升级为**国内医院数据集成中系统性真实发生的错误模式**(`configs/terminology_zh/unit_conversions.csv` 即 F2 的语义数据源)。related work 里 2610.21270 的四轴扰动不含单位维度,此处差异再添一层。

注:v2 把主实验数据源换成 Synthea 后,术语映射表须由 MIMIC 词表重建为按 LOINC 码索引(实测 Synthea 药品名对旧词表覆盖率仅 1.8%),详见 `docs/design-v2.md` §4。

## 7. 来源

- MedAgentBench v1:https://arxiv.org/abs/2501.14654 · https://ai.nejm.org/doi/full/10.1056/AIdbp2500144
- MedAgentBench v2(PSB 2026):https://psb.stanford.edu/psb-online/proceedings/psb26/chen_eric.pdf · https://pubmed.ncbi.nlm.nih.gov/41758153 · do-nothing 批评:https://openreview.net/forum?id=j92gdh0VRH
- AgentClinic:https://arxiv.org/abs/2405.07960 · npj Digital Medicine:https://www.nature.com/articles/s41746-026-02674-7 · https://www.agentclinic.com
- ReliabilityBench:https://arxiv.org/abs/2601.06112
- ChaosLLM:https://akiannillo.github.io/misc/publications/ISSRE2025_Iannillo.pdf
- 新撞车候选:https://arxiv.org/abs/2610.21270
