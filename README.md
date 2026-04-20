## Pruning Runs

```bash
uv run -m examples.pruning_sweep --example modulus --fractions 0.02,0.05,0.10 --out-dir /tmp/neurotop_next/modulus_sweep --checkpoint outputs/modulus/grokking_checkpoint.pt --prune-cycles 6 --finetune-epochs 10 --train-epochs 100 --max-samples 256 --modulus 113 --seed 0 --max-accuracy-drop 0.02 --min-final-sparsity 0.10
```

```bash
uv run -m examples.cifar.ex2_pruning --checkpoint outputs/cifar/training_checkpoint.pt --train-epochs 40 --prune-cycles 3 --finetune-epochs 5 --max-prune-fraction 0.075 --max-samples 128 --batch-size 256 --max-accuracy-drop 0.02 --min-final-sparsity 0.05 --out-dir /tmp/neurotop_next/cifar_cap075
```

```bash
uv run -m examples.final_experiments --example bert --out-dir /tmp/neurotop_final_bert_best_2pt --methods rf,random --seeds 0 --fractions 0.04 --prune-cycles 4 --finetune-epochs 2 --train-epochs 3 --subset-size 5000 --max-samples 512 --batch-size 32 --models-dir ./trained_models_serious --dataset cola --model-name bert-base-uncased --max-accuracy-drop 0.02 --min-final-sparsity 0.05
```
