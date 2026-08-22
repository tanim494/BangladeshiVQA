# Computational Environment and Fine-Tuning Parameters

This document records the SmolVLM2 fine-tuning setup reported in the BangladeshiVQA paper
(Appendix B). The corresponding notebook is
[`smolvlm2_finetuning.ipynb`](smolvlm2_finetuning.ipynb), and the resulting LoRA adapter
ships in [`smolvlm2_lora_adapter/`](smolvlm2_lora_adapter/).

## Environment

| Item | Value |
|---|---|
| Hardware | Dual NVIDIA T4 GPUs |
| Framework | Hugging Face Transformers + PEFT |
| Precision | BFloat16 mixed precision |
| Training type | Supervised fine-tuning |

## SmolVLM2-2.2B (fine-tuned small VLM)

Base model: `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, fine-tuned on the 5,628-pair
training split.

| Hyperparameter | Value |
|---|---|
| Quantization | 4-bit NormalFloat (NF4) |
| Adaptation | LoRA |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention projection matrices) |
| Optimizer | 8-bit Paged AdamW |
| Learning rate | 2e-4 |
| Per-device batch size | 1 |
| Gradient accumulation steps | 16 |
| Epochs | 3 |
| Image resolution | 384 × 384 (no image splitting) |
| Logging interval | every 5 steps |

The resulting checkpoint quantifies the in-domain Bangla adaptation effect on a small
model with weak prior visual-culture coverage of Bangladesh. The fine-tuned model is
evaluated against the original base SmolVLM2 so results reflect the impact of
fine-tuning rather than the model's underlying capacity.

## Other models in the paper

The paper additionally reports **Bangla-BERT cross-modal baselines** (BanglaBERT + ViT /
ResNet-50 — closed-vocabulary classifiers over the 1,524 unique answers) and
zero-/few-shot results for proprietary and open-source LVLMs. Only the SmolVLM2
fine-tuning code and adapter are released in this repository; see the paper for the
baseline architectures and the full result tables.
