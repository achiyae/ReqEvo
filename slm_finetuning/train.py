import os
import json
import random
import copy
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()

PROMPT_TEMPLATE = """You are an expert reviewer analyzing changes in PEP (Python Enhancement Proposal) specifications.
Below is the general context of the PEP document (metadata and versions lists), followed by a specific diff to analyze.
Determine the reason type and reason text for the diff.

### Document Context:
{document_context}

### Diff to Analyze:
{diff}

### Response:
```json
{{
  "reason_type": "{reason_type}",
  "reason_text": "{reason_text}"
}}
```"""

def load_raw_data(data_dir):
    data = []
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
        
        for diff_obj in content.get('diffs', []):
            if diff_obj.get('reason_type') and diff_obj.get('reason_text'):
                diff_inputs = {k: v for k, v in diff_obj.items() if k not in ['reason_type', 'reason_text']}
                diff_str = json.dumps(diff_inputs, indent=2, ensure_ascii=False)
                
                text = PROMPT_TEMPLATE.format(
                    document_context=document_context,
                    diff=diff_str,
                    reason_type=diff_obj['reason_type'],
                    reason_text=diff_obj['reason_text'].replace('"', '\\"')
                )
                
                data.append({
                    "text": text,
                    "document_context": document_context,
                    "diff_str": diff_str,
                    "reason_type": diff_obj['reason_type']
                })
    return data

def main():
    args = parse_args()
    model_name_safe = args.model_id.split('/')[-1]
    output_dir = f"./lora_adapter_{model_name_safe}"
    
    print(f"Loading data...")
    data_dir = r"C:\Repositories\ReqEvo\dataset_reviewers\Tali\outputs"
    raw_data = load_raw_data(data_dir)
    
    # Shuffle and split 90% Train / 10% Test
    random.seed(42)
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.9)
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]
    
    # Save the test set for the evaluate.py script
    test_file = f"test_set_{model_name_safe}.json"
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    print(f"Train size: {len(train_data)} | Test size: {len(test_data)} (saved to {test_file})")

    # Create HuggingFace datasets
    train_ds = Dataset.from_list([{"text": d["text"]} for d in train_data])
    test_ds = Dataset.from_list([{"text": d["text"]} for d in test_data])

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model_id} in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    # "all-linear" ensures PEFT targets all linear layers regardless of architecture (works for both Phi and Qwen)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # SFTTrainer will automatically call get_peft_model since we pass peft_config to it

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Reduced to 1 for 6GB VRAM compatibility
        gradient_accumulation_steps=8, # Kept effective batch size at 8
        optim="paged_adamw_32bit",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        max_length=2048,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Done! Model saved to {output_dir}")

if __name__ == "__main__":
    main()
