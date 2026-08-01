import os
import json
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

INFERENCE_TEMPLATE = """You are an expert reviewer analyzing changes in PEP (Python Enhancement Proposal) specifications.
Below is the general context of the PEP document (metadata and versions lists), followed by a specific diff to analyze.
Determine the reason type and reason text for the diff.

### Document Context:
{document_context}

### Diff to Analyze:
{diff}

### Response:
```json
{{
  "reason_type":"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    return parser.parse_args()


def get_hf_cache_dir():
    return (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
        or os.environ.get("HF_CACHE_DIR")
        or os.environ.get("HF_HOME")
    )


def detect_backend():
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if has_cuda:
        return "cuda"
    if has_mps:
        return "mps"
    return "cpu"


def parse_reason_type(response_text):
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            json_str = response_text[start:end+1]
        else:
            return None
    try:
        data = json.loads(json_str)
        return data.get("reason_type")
    except:
        return None

def main():
    args = parse_args()
    cache_dir = get_hf_cache_dir()
    if cache_dir:
        print(f"Using Hugging Face cache directory: {cache_dir}")

    tokenizer_kwargs = {"trust_remote_code": True}
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tokenizer_kwargs)

    backend = detect_backend()
    use_4bit = backend == "cuda"
    print(f"Detected backend: {backend}")

    model_kwargs = {
        "trust_remote_code": True,
    }

    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": quantization_config,
                "torch_dtype": torch.float16,
            }
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if backend == "mps" else torch.float32

    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, **model_kwargs
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    if not use_4bit:
        model.to(backend)
    model.eval()

    model_device = next(model.parameters()).device

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    correct = 0
    total = len(test_data)
    
    print(f"Evaluating {args.adapter_dir} on {total} samples...")
    for idx, item in enumerate(test_data):
        prompt = INFERENCE_TEMPLATE.format(
            document_context=item['document_context'],
            diff=item['diff_str']
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, pad_token_id=tokenizer.eos_token_id)
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        predicted_type = parse_reason_type('{"reason_type":' + response)
        
        if predicted_type == item['reason_type']:
            correct += 1
            
        print(f"[{idx+1}/{total}] True: {item['reason_type']} | Pred: {predicted_type}")
        
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.2f}%)")

if __name__ == "__main__":
    main()
