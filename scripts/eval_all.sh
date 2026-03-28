#!/bin/bash
# Run all evaluations on a trained model
# Usage: ./scripts/eval_all.sh checkpoints/small/final.pt

CHECKPOINT=${1:-checkpoints/small/final.pt}

echo "=== Perplexity ==="
python -c "
from memoria.model.config import small_config
from memoria.model.memoria_model import MemoriaModel
from memoria.eval.perplexity import evaluate_perplexity
import torch

config = small_config()
model = MemoriaModel(config)
ckpt = torch.load('$CHECKPOINT', map_location='cuda')
model.load_state_dict(ckpt['model_state'], strict=False)
model.state.load_state_cognitive(ckpt['cognitive_state'])
model = model.cuda()

result = evaluate_perplexity(model, num_batches=50)
print(result)
"

echo "=== Belief Tracking ==="
python -c "
from memoria.model.config import small_config
from memoria.model.memoria_model import MemoriaModel
from memoria.eval.belief_tracking import evaluate_belief_tracking
import torch

config = small_config()
model = MemoriaModel(config)
ckpt = torch.load('$CHECKPOINT', map_location='cuda')
model.load_state_dict(ckpt['model_state'], strict=False)
model.state.load_state_cognitive(ckpt['cognitive_state'])
model = model.cuda()

result = evaluate_belief_tracking(model)
print(result)
"

echo "=== Improvement Curve ==="
python -c "
from memoria.model.config import small_config
from memoria.model.memoria_model import MemoriaModel
from memoria.eval.improvement import evaluate_improvement_curve, save_improvement_results
import torch

config = small_config()
model = MemoriaModel(config)
ckpt = torch.load('$CHECKPOINT', map_location='cuda')
model.load_state_dict(ckpt['model_state'], strict=False)
model.state.load_state_cognitive(ckpt['cognitive_state'])
model = model.cuda()

result = evaluate_improvement_curve(model, total_interactions=200, eval_every=25)
save_improvement_results(result, 'results/improvement_curve.json')
print(f\"Accuracies: {result['accuracies']}\")
"
