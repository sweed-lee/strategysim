# StrategySim

消费策略仿真示例。随包数据全部为人工构造，用一名虚构顾客、一家商店、一份商品和一张优惠券演示从曝光到结算的流程。

| 模块 | 位置 | 可调整的内容 |
| --- | --- | --- |
| 用户 Agent | `strategysim/agents/` | 用户画像、访问内记忆、激励感知、商品选择和结算决策 |
| 领域知识 | `strategysim/agents/knowledge/` | 八个纯文本块，描述预算、门槛、价格与选择。阶段与知识块的对应关系见 `agents/prompting.py` |
| 阶段提示词 | `strategysim/agents/prompts/` | 模型可见的信息、任务和输出字段 |
| 交易环境 | `strategysim/envs/` | 商店、菜单、优惠券、购物车及订单执行 |
| 实验编排 | `strategysim/orchestrator/` | 策略载入、访问调度、交易观测、连续订单权重与结果汇总 |

## 运行示例

使用具备 AgentSociety2、pandas、pydantic 和 json-repair 的 Python 环境，从解压后的根目录执行。模型凭据、模型名和服务地址由运行者通过 AgentSociety2 环境变量提供，如 `AGENTSOCIETY_LLM_API_KEY`、`AGENTSOCIETY_LLM_MODEL`、`AGENTSOCIETY_LLM_API_BASE`。

```bash
python -m strategysim.run_independent --config config.json --output-dir outputs/demo
```

`config.json` 提供虚构画像，`data/` 提供访问、商店、菜单、优惠券、历史与基准购买率。不同策略可共用访问和商店文件，通过 `env_modules[0].kwargs.coupon_dataset_json` 指向各自的优惠券文件。可用 `bash run_strategies.sh config-a.json config-b.json` 并行运行多份配置，脚本为每份配置建立独立的结果目录。单次运行复用已有输出目录时会尝试续跑。

## 示例规则

券按输入文件中的有效期、门槛和面额结算，支付后移除已用券。连续意愿调整量范围为 −1 到 1，保留访问的订单贡献为基准购买率乘以一加调整量。汇总提供订单贡献、实付与用券统计。这里的参数仅用于演示代码接口。

在配置顶层设置 `"include_domain_knowledge": false` 可关闭知识块。模型消息日志默认关闭，可用 `"log_model_messages": true` 开启。

离线单元测试可在根目录运行：
```bash
python -m unittest discover -s strategysim/tests
```
