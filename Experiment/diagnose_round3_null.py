"""Investigate FP32 row-batch sensitivity without changing experiment results."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import quant_core
from run_pilot import capture, read_texts, sequences

torch.manual_seed(2)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
config = json.loads(Path('results/round3_l12_s2/config.json').read_text())['args']
tokenizer = AutoTokenizer.from_pretrained(config['model'], revision=config['revision'], local_files_only=True)
batches = sequences(read_texts(config['calibration']), tokenizer, 64, 256)
model = AutoModelForCausalLM.from_pretrained(config['model'], revision=config['revision'], local_files_only=True,
            dtype=torch.float32, attn_implementation='eager').to('cuda').eval().requires_grad_(False)
model.config.use_cache = False
target = model.get_submodule(config['target'])
original = target.weight.detach().clone()
xf, _, grad = capture(model, target, batches, 'cuda', gradients=True)
with torch.no_grad():
    for block in model.model.layers[:12]:
        for module in block.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.copy_(quant_core.RowQuantizer(module.weight, 4).quantize(module.weight.float()))
xq, _, _ = capture(model, target, batches, 'cuda', gradients=False)
xf, xq, grad = xf.cuda(), xq.cuda(), grad.cuda()
weights = quant_core.make_weights(grad, 4, 0.)
assert all(torch.equal(score, torch.ones_like(score)) for _, score in weights)
h1, c1 = quant_core.statistics(xf, xq, weights[0][1])
h2, c2 = quant_core.statistics(xf, xq, weights[-1][1])
assert torch.equal(h1, h2) and torch.equal(c1, c2)

traces = []
BaseQuantizer = quant_core.RowQuantizer


class RecordingQuantizer(BaseQuantizer):
    def __init__(self, weight, bits):
        super().__init__(weight, bits)
        self.inputs = []
        traces.append(self.inputs)

    def quantize(self, weight):
        self.inputs.append((weight/self.scale).detach().cpu().flatten())
        return super().quantize(weight)


quant_core.RowQuantizer = RecordingQuantizer
one = torch.ones(xq.shape[1], device='cuda')
baseline = quant_core.official_rescomp(original, xf, xq, one)
grouped = torch.cat([quant_core.official_rescomp(original[rows], xf, xq, score) for rows, score in weights])
quant_core.RowQuantizer = BaseQuantizer
different = baseline != grouped
indices = different.nonzero()
assert len(indices), 'Failure did not reproduce'
row, col = indices[indices[:, 1].argmin()].tolist()
full_trace = torch.stack(traces[0], dim=1)
group_trace = torch.cat([torch.stack(values, dim=1) for values in traces[1:]])
left, right = float(full_trace[row, col]), float(group_trace[row, col])
result = {'identical_all_one_scores': True, 'identical_H_C': True,
          'different_weight_count': int(different.sum()), 'total_weights': different.numel(),
          'first_differing_column': col, 'row': row,
          'full_row_batch_pre_round_grid_value': left, 'group_row_batch_pre_round_grid_value': right,
          'pre_round_difference': left-right,
          'conclusion': 'Same statistics, different row-batch FP32 updates cross a half-integer rounding boundary'}
Path('results/round3_null_diagnosis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
