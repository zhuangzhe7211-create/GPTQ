"""Three model API operations to learn before changing quantization code."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen2.5-0.5B')
    p.add_argument('--device', default='cpu')
    p.add_argument('--allow-download', action='store_true')
    a = p.parse_args()
    # 1. Tokenizer makes IDs; from_pretrained loads BOTH architecture and weights.
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=not a.allow_download, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=not a.allow_download,
                    trust_remote_code=False, dtype=torch.bfloat16 if a.device.startswith('cuda') else torch.float32)
    model = model.to(a.device).eval()
    # 2. Find a linear layer exactly as you would in your own nn.Module.
    target = model.get_submodule('model.layers.1.self_attn.o_proj')
    print('layers:', model.config.num_hidden_layers, 'target weight:', tuple(target.weight.shape))
    # 3. Forward gives per-position vocabulary logits, not generated sentences.
    inputs = tok('The purpose of quantization is', return_tensors='pt').to(a.device)
    with torch.no_grad():
        logits = model(**inputs).logits
        output = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    print('input shape:', tuple(inputs['input_ids'].shape), 'logits shape:', tuple(logits.shape))
    print(tok.decode(output[0], skip_special_tokens=True))


if __name__ == '__main__':
    main()
