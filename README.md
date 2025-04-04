<div align="center">

# Awesome Process Reward Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

</div>



## 🔗 Table of Contents

- [Awesome Process Reward Models](#awesome-process-reward-models)
  - [🔗 Table of Contents](#-table-of-contents)
  - [📐 PRMs for Mathematical Tasks](#-prms-for-mathematical-tasks)
  - [💻 PRMs for Other Tasks](#-prms-for-other-tasks)
  - [🔍 Other Process-Supervised Models](#-other-process-supervised-models)
  - [🌇 Multimodal PRMs](#-multimodal-prms)
  - [📊 Benchmarks](#-benchmarks)
  - [💪 Contributing](#-contributing)
  - [📝 Citation](#-citation)



## 📐 PRMs for Mathematical Tasks

- (**GenPRM**) GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning [[arXiv 2025.04](https://arxiv.org/abs/2504.00891)] [[Code](https://github.com/RyanLiu112/GenPRM)] [[Website](https://ryanliu112.github.io/GenPRM)] [[Model](https://huggingface.co/collections/GenPRM/genprm-67ee4936234ba5dd16bb9943)] [[Data](https://huggingface.co/collections/GenPRM/genprm-67ee4936234ba5dd16bb9943)]

- (**RetrievalPRM**) Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning [[arXiv 2025.02](https://arxiv.org/abs/2502.14361)] [[Code](https://anonymous.4open.science/r/RetrievalPRM-1C77)] [[Model](https://huggingface.co/gebro13/RetrievalPRM)] [[Data](https://huggingface.co/datasets/gebro13/RetrievalPRM_Dataset)]

- (**Multilingual PRM**) Demystifying Multilingual Chain-of-Thought in Process Reward Modeling [[arXiv 2025.02](https://arxiv.org/abs/2502.14361)] [[Code](https://github.com/weixuan-wang123/Multilingual-PRM)]

- (**Universal PRM**) AURORA:Automated Training Framework of Universal Process Reward Models via Ensemble Prompting and Reverse Verification [[arXiv 2025.02](https://arxiv.org/abs/2502.11520)] [[Website](https://auroraprm.github.io)] [[Model](https://huggingface.co/infly/Universal-PRM-7B)]

- (**PURE PRM**) Stop Gamma Decay: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning [[Blog](https://tungsten-ink-510.notion.site/Stop-Gamma-Decay-Min-Form-Credit-Assignment-Is-All-Process-Reward-Model-Needs-for-Reasoning-19fcb6ed0184804eb07fd310b38af155)] [[Code](https://github.com/CJReinforce/PURE)] [[Model](https://huggingface.co/jinachris/PURE-PRM-7B)] [[Data](https://huggingface.co/datasets/HuggingFaceH4/prm800k-trl-dedup)]

- (**CFPRM**) Coarse-to-Fine Process Reward Modeling for Mathematical Reasoning [[arXiv 2025.01](https://arxiv.org/abs/2501.13622)]

- (**Qwen2.5-Math PRM**) The Lessons of Developing Process Reward Models in Mathematical Reasoning [[arXiv 2025.01](https://arxiv.org/abs/2501.07301)] [[Website](https://qwenlm.github.io/blog/qwen2.5-math-prm)] [[Model](https://huggingface.co/collections/Qwen/qwen25-math-66eaa240a1b7d5ee65f1da3e)]

- (**PPM**) rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking [[arXiv 2025.01](https://arxiv.org/abs/2501.04519)] [[Code](https://github.com/microsoft/rStar)]

- (**ER-PRM**) Entropy-Regularized Process Reward Model [[arXiv 2024.12](https://arxiv.org/abs/2412.11006)] [[Code](https://github.com/hanningzhang/ER-PRM)] [[Website](https://hanningzhang.github.io/math-prm/)] [[Model](https://huggingface.co/HanningZhang/Llama3.1-Math-PRM)] [[Data](https://huggingface.co/datasets/HanningZhang/ER-PRM-Data)]

- (**Implicit PRM**) Free Process Rewards without Process Labels [[arXiv 2024.12](https://arxiv.org/abs/2412.01981)] [[Code](https://github.com/PRIME-RL/ImplicitPRM)] [[Model](https://huggingface.co/collections/Windy0822/implicitprm-675033e6b3719046c13e2e48)] [[Data](https://huggingface.co/datasets/Windy0822/ultrainteract_math_rollout)]

- (**Skywork PRM**) Skywork-o1 Open Series [[Model](https://huggingface.co/collections/Skywork/skywork-o1-open-67453df58e12f6c3934738d0)]

- (**RLHFlow PRM**) An Implementation of Generative PRM [[Code](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm)] [[Model](https://huggingface.co/collections/RLHFlow/rlhflow-math-process-reward-model-6725a42fc8808e12aa1cb144)] [[Data](https://huggingface.co/collections/RLHFlow/rlhflow-math-process-reward-model-6725a42fc8808e12aa1cb144)]

- (**PQM**) Process Reward Model with Q-Value Rankings [[ICLR 2025](https://openreview.net/forum?id=wQEdh2cgEk)] [[arXiv 2024.10](https://arxiv.org/abs/2410.11287)] [[Code](https://github.com/WindyLee0822/Process_Q_Model)] [[Model](https://huggingface.co/Windy0822/PQM)]

- (**Math-psa**) OpenR: An Open Source Framework for Advanced Reasoning with Large Language Models [[arXiv 2024.10](https://arxiv.org/abs/2410.09671)] [[Code](https://github.com/openreasoner/openr)] [[Website](https://openreasoner.github.io)] [[Model](https://huggingface.co/openreasoner/Math-psa)] [[Data](https://huggingface.co/datasets/openreasoner/MATH-APS)]

- (**PAV**) Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning [[ICLR 2025](https://openreview.net/forum?id=A6Y7AqlzLW)] [[arXiv 2024.10](https://arxiv.org/abs/2410.08146)]

- FG-PRM: Fine-grained Hallucination Detection and Mitigation in Language Model Mathematical Reasoning

- Advancing Process Verification for Large Language Models via Tree-Based Preference Learning 

- (**OmegaPRM**) Improve Mathematical Reasoning in Language Models by Automated Process Supervision [[arXiv 2024.06](https://arxiv.org/abs/2406.06592)] [[Code (Third Party)](https://github.com/openreasoner/openr/tree/main/data/omegaPRM_v2)]

- AlphaMath Almost Zero: process Supervision without process 

- (**Math-Shepherd**) Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations [[ACL 2024](https://aclanthology.org/2024.acl-long.510)] [[arXiv 2023.12](https://arxiv.org/abs/2312.08935)] [[Model](https://huggingface.co/peiyi9979/math-shepherd-mistral-7b-prm)] [[Data](https://huggingface.co/datasets/peiyi9979/Math-Shepherd)]

- Let's reward step by step: Step-Level reward model as the Navigators for Reasoning [[arXiv 2023.10](https://arxiv.org/abs/2310.10080)]

- Let's Verify Step by Step [[ICLR 2024](https://openreview.net/forum?id=v8L0pN6EOi)] [[arXiv 2023.05](https://arxiv.org/abs/2305.20050)] [[Data](https://github.com/openai/prm800k)] [[Blog](https://openai.com/research/improving-mathematical-reasoning-with-process-supervision)]

- Solving math word problems with process- and outcome-based feedback [[arXiv 2022.11](https://arxiv.org/abs/2211.14275)]



## 💻 PRMs for Other Tasks

- (**MT-RewardTree**) MT-RewardTree: A Comprehensive Framework for Advancing LLM-Based Machine Translation via Reward Modeling [[arXiv 2025.03](https://arxiv.org/abs/2503.12123)] [[Code](https://github.com/sabijun/MT-RewardTree)] [[Website](https://sabijun.github.io/MT_RewardTreePage)] [[Model](https://huggingface.co/collections/sabijun/mt-rewardtree-models-67cac935143f75dfae6f0938)] [[Data](https://huggingface.co/collections/sabijun/mt-rewardtree-dataset-67cacadc0dcbc92c02428948)]

- (**ASPRM**) AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence [[arXiv 2025.02](https://arxiv.org/abs/2502.13943)] [[Code](https://github.com/Lux0926/ASPRM)] [[Model](https://huggingface.co/Lux0926)] [[Data](https://huggingface.co/Lux0926)]

- (**AgentPRM**) Process Reward Models for LLM Agents: Practical Framework and Directions [[arXiv 2025.02](https://arxiv.org/abs/2502.10325)] [[Code](https://github.com/sanjibanc/agent_prm)]

- (**VersaPRM**) VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data [[arXiv 2025.02](https://arxiv.org/abs/2502.06737)] [[Code](https://github.com/UW-Madison-Lee-Lab/VersaPRM)] [[Model](https://huggingface.co/collections/UW-Madison-Lee-Lab/versaprm-67a7eb34049b2a1bd3055f6e)] [[Data](https://huggingface.co/datasets/UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled)]

- (**MedS$^3$**) MedS$^3$: Towards Medical Small Language Models with Self-Evolved Slow Thinking [[arXiv 2025.01](https://arxiv.org/abs/2501.12051)] [[Code](https://github.com/pixas/MedSSS)] [[Model](https://huggingface.co/pixas/MedSSS_PRM)] [[Data](https://huggingface.co/datasets/pixas/MedSSS-data)]

- (**OpenPRM**) OpenPRM: Building Open-domain Process-based Reward Models with Preference Trees [[ICLR 2025](https://openreview.net/forum?id=fGIqGfmgkW)]

- (**o1-Coder**) o1-Coder: an o1 Replication for Coding [[arXiv 2024.12](https://arxiv.org/abs/2412.00154)] [[Code](https://github.com/ADaM-BJTU/O1-CODER)]

- Process Supervision-Guided Policy Optimization for Code Generation [[arXiv 2024.10](https://arxiv.org/abs/2410.17621)]



## 🔍 Other Process-Supervised Models

- (**Dyve**) Dyve: Thinking Fast and Slow for Dynamic Process Verification [[arXiv 2025.02](https://arxiv.org/abs/2502.11157)] [[Code](https://github.com/staymylove/Dyve)] [[Model](https://huggingface.co/Jianyuan1/deepseek-r1-14b-cot-math-reasoning-full)] [[Data](https://huggingface.co/datasets/Jianyuan1/cot-data)]



## 🌇 Multimodal PRMs

- (**VisualPRM**) VisualPRM: An Effective Process Reward Model for Multimodal Reasoning [[arXiv 2025.03](https://arxiv.org/abs/2503.10291)] [[Website](https://internvl.github.io/blog/2025-03-13-VisualPRM)] [[Model](https://huggingface.co/OpenGVLab/VisualPRM-8B)] [[Data](https://huggingface.co/datasets/OpenGVLab/VisualPRM400K)]



## 📊 Benchmarks

- (**MPBench**) MPBench: A Comprehensive Multimodal Reasoning Benchmark for Process Errors Identification [[arXiv 2025.03](https://arxiv.org/abs/2503.12505)] [[Code](https://github.com/xu5zhao/MPBench)] [[Website](https://mpbench.github.io)] [[Data](https://huggingface.co/datasets/xuzhaopan/MPBench)]

- (**PRMBench**) PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models [[arXiv 2025.01](https://arxiv.org/abs/2501.03124)] [[Code](https://github.com/ssmisya/PRMBench)] [[Website](https://prmbench.github.io)] [[Data](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)]

- (**ProcessBench**) ProcessBench: Identifying Process Errors in Mathematical Reasoning [[arXiv 2024.12](https://arxiv.org/abs/2412.06559)] [[Code](https://github.com/QwenLM/ProcessBench)] [[Model](https://huggingface.co/Qwen/Qwen2.5-Math-7B-PRM800K)] [[Data](https://huggingface.co/datasets/Qwen/ProcessBench)]



## 💪 Contributing

If you find a paper that should be included but is missing, feel free to create an issue or submit a pull request. Please use the following format to contribute:

```bash
- (**Method Name**) Title [[Journal/Conference](Link)] [[arXiv Year.Month](Link)] [[Code](Link)] [[Website](Link)] [[Model](Link)] [[Data](Link)]
```

## 📝 Citation

If you find this work helpful, please consider citing the repository:

```bibtex
@misc{Awesome-Process-Reward-Models,
    title        = {Awesome Process Reward Models},
    author       = {Runze Liu and Jian Zhao and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    howpublished = {\url{https://github.com/RyanLiu112/Awesome-Process-Reward-Models}},
    note         = {GitHub repository},
    year         = {2025}
}
```

Out recent work on PRM test-time scaling:

```bibtex
@article{zhao2025genprm,
    title   = {GenPRM: Scaling Test-Time Compute of Process Reward Models via Generative Reasoning},
    author  = {Jian Zhao and Runze Liu and Kaiyan Zhang and Zhimu Zhou and Junqi Gao and Dong Li and Jiafei Lyu and Zhouyi Qian and Biqing Qi and Xiu Li and Bowen Zhou},
    journal = {arXiv preprint arXiv:2504.00891},
    year    = {2025}
}
```

Our recent work on LLM test-time scaling with PRMs:

```bibtex
@article{liu2025can,
    title   = {Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling},
    author  = {Runze Liu and Junqi Gao and Jian Zhao and Kaiyan Zhang and Xiu Li and Biqing Qi and Wanli Ouyang and Bowen Zhou},
    journal = {arXiv preprint arXiv:2502.06703},
    year    = {2025}
}
```
