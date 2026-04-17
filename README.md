## Pruning Runs

```bash
uv run -m examples.pruning_sweep --example modulus --fractions 0.02,0.05,0.10 --out-dir /tmp/neurotop_next/modulus_sweep --checkpoint outputs/modulus/grokking_checkpoint.pt --prune-cycles 6 --finetune-epochs 10 --train-epochs 100 --max-samples 256 --modulus 113 --seed 0 --max-accuracy-drop 0.02 --min-final-sparsity 0.10
```

```bash
uv run -m examples.cifar.ex2_pruning --checkpoint outputs/cifar/training_checkpoint.pt --train-epochs 40 --prune-cycles 3 --finetune-epochs 5 --max-prune-fraction 0.075 --max-samples 128 --batch-size 256 --max-accuracy-drop 0.02 --min-final-sparsity 0.05 --out-dir /tmp/neurotop_next/cifar_cap075
```

```bash
uv run -m examples.pruning_sweep --example bert --dataset cola --model-name bert-base-uncased --fractions 0.02,0.05 --out-dir /tmp/neurotop_next/bert_cola_sweep --prune-cycles 2 --finetune-epochs 1 --train-epochs 1 --subset-size 1000 --max-samples 128 --batch-size 16 --models-dir ./trained_models --seed 0 --max-accuracy-drop 0.03 --min-final-sparsity 0.02
```
