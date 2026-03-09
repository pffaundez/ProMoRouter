from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import torch

# Model and tokenizer
model_id = "mistralai/Mistral-7B-Instruct-v0.3"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

# Quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,  # INT4
    group_size=128,  # Common group size for GPTQ
    desc_act=False,  # Disable act-order for simplicity
)

# Load model for quantization
model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config,
    torch_dtype=torch.float16,
    device_map="auto"  # Use GPU if available
)

# Calibration data (load a small dataset, e.g., from a text file)
with open("calibration_data.txt", "r") as f:
    calibration_data = f.read().splitlines()

# Quantize
model.quantize(calibration_data, batch_size=4, use_triton=True)

# Save quantized model
save_path = "./quantized_mistral_7b_gptq"
model.save_quantized(save_path)
tokenizer.save_pretrained(save_path)

print(f"Quantized model saved to {save_path}")