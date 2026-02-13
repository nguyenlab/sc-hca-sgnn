# Relevant Papers for LLM-Based Vulnerability Detection (2024-2025)

This document tracks relevant research papers using small open-source LLMs (7B-14B) for smart contract vulnerability detection, serving as a reference for baselines in this project.

## 1. Efficient Adaptation of Large Language Models for Smart Contract Vulnerability Detection (2025)
*   **URL:** https://doi.org/10.1145/3727582.3728688
*   **Models:** StarCoder-7B, LLAMA 3-8B, CodeLlama-7B
*   **Methodology:** Uses **LoRA (Low-Rank Adaptation)** and **Parameter-Efficient Fine-Tuning (PEFT)**. Adapts general code models using lightweight adapters instead of full-parameter training. Uses a dedicated classification head.
*   **Relevance:** High. This paper demonstrates the most resource-efficient way to implement a strong baseline. The PEFT/LoRA approach is ideal for our `llm-based` implementation to avoid high GPU costs while outperforming zero-shot methods.

## 2. Detection Made Easy: Potentials of Large Language Models for Solidity Vulnerabilities (Sept 2024)
*   **URL:** https://arxiv.org/abs/2409.10574
*   **Models:** CodeLlama-7B-Instruct, CodeLlama-13B-Instruct
*   **Methodology:** Comparative study of **Zero-shot prompting** vs. **Fine-tuning**.
*   **Key Findings:** Zero-shot prompting struggles with complex Logic/Reentrancy bugs. Fine-tuning 7B models yields a massive performance jump, beating traditional static analysis tools.
*   **Relevance:** Justifies the need for both a simple Zero-shot baseline (for checking prompt sensitivity) and a Fine-tuned baseline (for actual competitive performance against GNNs).

## 3. Smart-LLaMA: Two-Stage Post-Training of Large Language Models for Smart Contract Vulnerability Detection and Explanation (Nov 2024)
*   **URL:** https://arxiv.org/abs/2411.06221
*   **Models:** LLaMA-2-7B / CodeLlama-7B
*   **Methodology:** Two-stage training: (1) Unsupervised pre-training on raw Solidity corpus to learn syntax. (2) Instruction tuning on labeled vulnerability datasets.
*   **Relevance:** Highlights the importance of domain adaptation. For this project, we can approximate this by skipping Stage 1 (too expensive) and focusing on Stage 2 (Instruction Tuning) on our `data/sc-source` dataset.

## 4. FELLMVP: An Ensemble LLM Framework for Classifying Smart Contract Vulnerabilities (2024)
*   **URL:** https://doi.org/10.1109/Blockchain62396.2024.00021
*   **Models:** CodeLlama-7B
*   **Methodology:** Uses the LLM as a **Feature Extractor** (Embedding Model). Extracts vector representations of code and feeds them into classic classifiers or ensemble models.
*   **Relevance:** Conceptual "SOTA" baseline. Comparing **LLM Embeddings** (from this approach) vs **GNN/AST Embeddings** (our project's core) is the most direct apples-to-apples comparison of "Semantic vs Structural" feature learning.

## 5. Large Language Model-Powered Smart Contract Vulnerability Detection: New Perspectives (2024)
*   **URL:** https://arxiv.org/abs/2310.01152
*   **Technique:** **Auditor-Critic (Adversarial) Framework**.
*   **Methodology:** Decomposes detection into two personas:
    1.  **Auditor:** Scans code to flag potential vulnerabilities (High Recall).
    2.  **Critic:** Reviews Auditor's findings to filter false positives (High Precision).
*   **Relevance:** Vital for small 7B models which are prone to hallucinations. Running a second "verification" pass can significantly improve precision without retraining.

## 6. LLM4Vuln: A Knowledge-Enhanced Framework for Smart Contract Vulnerability Detection (ISSTA 2024)
*   **URL:** https://arxiv.org/abs/2401.16185
*   **Technique:** **RAG / Knowledge-Enhanced Prompting**.
*   **Methodology:** Injects vulnerability definitions, patterns, and audit checklists into the prompt context (Retrieval-Augmented Generation) instead of relying solely on the model's pre-trained weights.
*   **Relevance:** Bridges the knowledge gap for small models. Instead of fine-tuning a model to "know" reentrancy, we provide the definition in the prompt.


## 7. Assisting Static Analysis with LLMs: A Hybrid Approach (2024)
*   **URL:** https://doi.org/10.1145/3611643.3613078
*   **Technique:** **Tool-Integrated / Hybrid Prompting**.
*   **Methodology:** Uses static tools (like Slither) to identify candidate lines, then uses the LLM as a "False Positive Filter" to reason about specific flagged code.
*   **Relevance:** Solves the context window limit for small models. Instead of analyzing the whole contract, the model only analyzes functions flagged by static tools.

## Recommendations for Implementation

Based on these papers, the following baselines are recommended for the `llm-based/` directory:

1.  **Modern Model Selection:** While 2024 papers used CodeLlama, we should assume **Qwen2.5-Coder-7B-Instruct** or **DeepSeek-Coder-6.7B** for 2025/2026 contexts, as they significantly outperform CodeLlama at the same size.
2.  **Proposed Baselines:**
    *   **Zero-Shot:** `llm-based/evaluate_zeroshot.py` (Using Qwen2.5-7B)
    *   **Chain-of-Thought (CoT):** `llm-based/evaluate_cot.py` (Decomposed reasoning steps)
    *   **LoRA Fine-Tuning:** `llm-based/train_lora.py` (PEFT adaptation on training set)
    *   **Embedding Comparison:** `llm-based/extract_embeddings.py` (Last layer states as features)
