# LingBot full-state recovery experiment

Authorized recovery fix on frank/eighteen-preparation. Saves FP32 master model, AdamW, LR scheduler, optimizer step, per-rank RNG and DataLoader epoch/cursor. Replays seeded workers before restoring training RNG. Exact worker replay requires load_worker >= 1 (production uses 4). Completion marker is written last; missing state or changed config fails closed. No automatic deletion.

Validation: Python syntax and git diff checks pass. GPU FSDP2 test pending; small-model production-method test covers worker randomness, accumulation and epoch crossing. Formal training has not started.
`nSingle-GPU FSDP2 experiment passed at c92ba00: save update2, resume4, exact batches/loss/master weights/AdamW/LR/progress across epoch boundary. Distributed experiment added; pending execution.
Final two-GPU FSDP2 result at6276e52: six optimizer updates, checkpoint2, resume4, accumulation2, two random workers, epoch crossing; batches/loss/FP32 weights/AdamW/LR/step/epoch exact. DLC dlc1wwk5jfxbk3br completed LingBot portion successfully; later OpenWAM test in same job failed, so overall job Failed. LingBot result.json is separate and explicit. No formal LingBot training launched.

用户确认原版bf16历史权重全部保留、完整恢复状态仅最新一份。FP32 master现随training_state.pt保存/恢复；新complete.json落盘后才移除旧training_state.pt及完成标记，历史bf16导出保持。既有小模型测试扩展到第二次保存清理、最新FP32与optimizer恢复。尚待远端定向验收。
