# LingBot full-state recovery experiment

Authorized recovery fix on frank/eighteen-preparation. Saves FP32 master model, AdamW, LR scheduler, optimizer step, per-rank RNG and DataLoader epoch/cursor. Replays seeded workers before restoring training RNG. Exact worker replay requires load_worker >= 1 (production uses 4). Completion marker is written last; missing state or changed config fails closed. No automatic deletion.

Validation: Python syntax and git diff checks pass. GPU FSDP2 test pending; small-model production-method test covers worker randomness, accumulation and epoch crossing. Formal training has not started.
`nSingle-GPU FSDP2 experiment passed at c92ba00: save update2, resume4, exact batches/loss/master weights/AdamW/LR/progress across epoch boundary. Distributed experiment added; pending execution.
