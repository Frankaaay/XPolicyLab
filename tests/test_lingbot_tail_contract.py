"""CPU-only checks of production sampling and action tail alignment."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parents[1] / "policy/LingBot_VA"


def production_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


class TailContractTest(unittest.TestCase):
    def test_all_short_lengths_keep_actions_and_sampled_frames(self):
        sample = production_function(ROOT / "process_data.py", "sample_frame_ids", {"VAE_TEMPORAL_RATE": 4})
        process = production_function(
            ROOT / "lingbot_va/wan_va/dataset/lerobot_latent_dataset.py", "_action_post_process",
            {"np": np, "torch": torch, "rearrange": rearrange},
        )
        reader = SimpleNamespace(config=SimpleNamespace(env_type="none", inverse_used_action_channel_ids=list(range(30))),
                                 q01=np.zeros(30), q99=np.ones(30))
        for n in range(1, 514):
            with self.subTest(length=n):
                ids = sample(n, 2)
                self.assertEqual([i for i in ids if i < n], list(range(0, n, 2)))
                self.assertEqual((len(ids) - 1) % 4, 0)
                self.assertTrue(np.all(np.diff(ids) == 2))
                actions, mask = process(reader, 0, n, ids, np.full((n, 30), .5))
                self.assertEqual(actions.shape[2], 8)
                self.assertEqual(int(mask.sum()), n * 30)
                self.assertFalse(mask[:, 0].any())
                self.assertTrue(torch.all(actions[~mask] == 0))

    def test_invalid_sample_requests(self):
        sample = production_function(ROOT / "process_data.py", "sample_frame_ids", {"VAE_TEMPORAL_RATE": 4})
        for count, stride in [(0, 2), (5, 0), (-1, 2)]:
            with self.assertRaises(ValueError):
                sample(count, stride)


if __name__ == "__main__":
    unittest.main()
