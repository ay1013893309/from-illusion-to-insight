# CPDP Change-Aware

`cpdp_change_aware` 是一个独立的小项目，用来实现你提出的设想：

1. 用源项目中大量已标注的“代码变更对”学习跨项目通用的缺陷模式。
2. 在扩展上下文里加入 AST/结构特征，弱化具体函数名差异，突出跨项目共享的变更模式。
3. 用多智能体辩论框架完成最终判定。
4. 每次预测时检索最相似的源项目变更对，作为 few-shot 迁移样本。

## 任务定义

训练和知识库来源：

- 源项目变更对：`old_code + diff + new_code + new_label`

预测对象：

- 目标项目变更对：`old_code + diff + new_code`

说明：

- 这个 MVP 仍然是 change-aware 设定，所以目标项目输入默认也是“变更对”。
- 项目内不做深度微调；“训练”体现在构建源项目变更知识库、检索器和跨项目 few-shot 推理流程上。

## 项目结构

- `src/cpdp_change_aware/data/`：变更对读取与从版本 CSV 构造样本
- `src/cpdp_change_aware/features/`：diff、AST/结构特征、扩展上下文
- `src/cpdp_change_aware/retrieval/`：基于 TF-IDF 的相似变更检索器
- `src/cpdp_change_aware/debate/`：多智能体辩论系统
- `src/cpdp_change_aware/prompts/`：跨项目提示词模板
- `examples/`：toy 数据

## 输入格式

推荐使用 `jsonl`，每行一个变更对：

```json
{
  "pair_id": "camel-2.9-3.0-org/foo/A.java",
  "project": "camel",
  "file_path": "org/foo/A.java",
  "language": "java",
  "old_version": "2.9.0",
  "new_version": "3.0.0",
  "old_code": "class A { ... }",
  "new_code": "class A { ... }",
  "old_label": 0,
  "new_label": 1
}
```

其中：

- `old_label` 可选
- `new_label` 对训练集是必需的，对待预测集可为空

## 安装

```bash
cd cpdp_change_aware
python -m pip install -e .
```

## 典型流程

### 1. 从两个版本 CSV 构造变更对

如果你已经有和原仓库兼容的 `File/Bug/SRC` CSV，可以直接生成 jsonl：

```bash
cpdp-change-aware prepare-pairs ^
  --project camel ^
  --old-csv ..\src\sdp\Dataset\File-level\camel-2.9.0_ground-truth-files_dataset.csv ^
  --new-csv ..\src\sdp\Dataset\File-level\camel-3.0.0_ground-truth-files_dataset.csv ^
  --output data\camel_2.9.0_3.0.0.jsonl
```

### 2. 用源项目变更对构建检索知识库

```bash
cpdp-change-aware fit ^
  --sources examples\toy_source.jsonl ^
  --artifact-path artifacts\toy_index.pkl
```

### 2.1 批量从原仓库 File-level CSV 生成 jsonl

```bash
cpdp-change-aware prepare-bulk ^
  --dataset-dir ..\src\sdp\Dataset\File-level ^
  --output-dir data\file_level_pairs
```

输出规则：

- 每个项目一个子目录
- 每对相邻版本一个 `jsonl`
- 根目录额外生成 `manifest.csv`

### 3. 对目标项目变更对做跨项目预测

```bash
cpdp-change-aware predict ^
  --artifact-path artifacts\toy_index.pkl ^
  --target examples\toy_target.jsonl ^
  --output predictions.csv ^
  --backend heuristic ^
  --top-k 3
```

### 4. 启用 LLM 多智能体辩论

```bash
cpdp-change-aware predict ^
  --artifact-path artifacts\toy_index.pkl ^
  --target examples\toy_target.jsonl ^
  --output predictions_llm.csv ^
  --backend llm ^
  --model gpt-5-mini ^
  --top-k 5
```

## 设计说明

### 1. 变更感知特征

每个变更对都会抽取：

- unified diff
- 行级变更统计
- AST/结构摘要
- 结构变化增量
- 受控长度的旧代码/新代码片段

### 2. 跨项目 few-shot 检索

检索器不会只看原始代码文本，还会混合：

- diff 文本
- AST/结构摘要
- 变更统计

这样更容易学到“条件分支变复杂”“异常处理路径变化”“状态更新逻辑变更”之类的通用模式。

### 3. 多智能体辩论

- `Analyzer`：解释变更语义和通用风险模式
- `Proposer`：论证该变更可能引入缺陷
- `Skeptic`：从重构、无害修改、命名调整等角度反驳
- `Judge`：综合 few-shot 样本和双方观点给出最终标签

### 4. 后续增强方向

- 把 TF-IDF 检索器替换成 CodeBERT / UniXcoder 向量检索
- 加入真实 AST parser，例如 JavaParser/tree-sitter
- 对 `00/01/10/11` 转移子集分别建模
- 加入少量 target adaptation
