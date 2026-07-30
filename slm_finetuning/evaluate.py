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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", quantization_config=quantization_config, torch_dtype=torch.float16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

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
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
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
