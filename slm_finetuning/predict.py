import os
import json
import re
import math
import argparse
import torch
import copy
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

def parse_json_response(response_text):
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            json_str = response_text[start:end+1]
        else:
            return None, None
    try:
        data = json.loads(json_str)
        return data.get("reason_type"), data.get("reason_text")
    except:
        return None, None


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "dataset_reviewers", "Tali", "outputs")
    data_dir = os.path.normpath(data_dir)

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

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(data_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
        general_props = {k: v for k, v in content.items() if k != 'diffs'}
        if 'versions' in general_props:
            general_props['versions'] = copy.deepcopy(general_props['versions'])
            for v_id, v_data in general_props['versions'].items():
                v_data.pop('content', None)
                
        document_context = json.dumps(general_props, indent=2, ensure_ascii=False)
        updated = False
        
        for diff_obj in content.get('diffs', []):
            if not diff_obj.get('reason_type') or not diff_obj.get('reason_text'):
                diff_inputs = {k: v for k, v in diff_obj.items() if k not in ['reason_type', 'reason_text']}
                diff_str = json.dumps(diff_inputs, indent=2, ensure_ascii=False)
                
                prompt = INFERENCE_TEMPLATE.format(document_context=document_context, diff=diff_str)
                inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
                
                with torch.no_grad():
                    # We pass output_scores=True to get the logprobs (certainty) of the generation
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.1,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Calculate Certainty Score from the logprobs
                transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
                avg_logprob = transition_scores[0].mean().item()
                certainty_score = round(math.exp(avg_logprob), 4) # Converts logprob to probability (0.0 to 1.0)
                
                full_json_str = '{"reason_type":' + response
                reason_type, reason_text = parse_json_response(full_json_str)
                
                if reason_type and reason_text:
                    diff_obj['reason_type'] = reason_type
                    diff_obj['reason_text'] = reason_text
                    diff_obj['certainty'] = certainty_score
                    updated = True
                    print(f"[{filename}] Updated Type: {reason_type} | Certainty: {certainty_score:.2f}")

        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, indent=2)
            print(f"Saved {filename}")

if __name__ == "__main__":
    main()
