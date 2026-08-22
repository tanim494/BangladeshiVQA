# SmolVLM2-2.2B LoRA adapter (fine-tuned on BangladeshiVQA)

This is the **final LoRA adapter** produced by
[`../smolvlm2_finetuning.ipynb`](../smolvlm2_finetuning.ipynb) — the SmolVLM2-2.2B
fine-tuning run reported in the paper. It is a small (~18 MB) adapter, **not** a full model;
load it on top of the base model.

| | |
|---|---|
| Base model | [`HuggingFaceTB/SmolVLM2-2.2B-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) |
| Method | LoRA (PEFT), 4-bit NF4 base |
| Rank / α / dropout | r = 8 · α = 16 · dropout = 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Task | Bangla open-ended VQA |
| Training data | BangladeshiVQA train split (5,628 QA pairs) |

Full hyperparameters are in [`../finetuning_config.md`](../finetuning_config.md).

## Usage

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

BASE = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
ADAPTER = "code/training/smolvlm2_lora_adapter"  # this directory

processor = AutoProcessor.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)   # attach the fine-tuned adapter
model.eval()

image = Image.open("data/sample_images/CS_0001.jpg").convert("RGB")
question = "কয়টি সিএনজি অটোরিকশা দেখা যাচ্ছে?"
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=32)
print(processor.decode(out[0], skip_special_tokens=True))
```

## Files

- `adapter_config.json` — PEFT/LoRA configuration.
- `adapter_model.safetensors` — the trained LoRA weights.

## License

Released under [CC BY-NC 4.0](../../../LICENSE) (non-commercial academic use), consistent with
the BangladeshiVQA dataset it was fine-tuned on. The base model retains its own license.
