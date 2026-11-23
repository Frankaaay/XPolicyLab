"""CPU-only checks of production sampling and action tail alignment."""
import ast
import hashlib
import json
import os
import tempfile
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
    def test_cache_resume_requires_intact_outputs_and_same_inputs(self):
        scope = dict(Path=Path, hashlib=hashlib, json=json, os=os)
        for name in ['file_sha256', 'atomic_json', 'validate_episode_receipt']:
            production_function(ROOT / 'process_data.py', name, scope)
        out = Path(tempfile.mkdtemp(prefix='lingbot-resume-test-'))
        marker = out / 'episode.json'
        inputs = {'source.mp4': [123, 456]}
        validate = scope['validate_episode_receipt']
        self.assertIsNone(validate(out, marker, inputs))
        artifact = out / 'latent.pth'
        artifact.write_bytes(b'complete latent')
        receipt = dict(inputs=inputs, outputs={'latent.pth': scope['file_sha256'](artifact)})
        scope['atomic_json'](marker, receipt)
        self.assertEqual(validate(out, marker, inputs), receipt)
        with self.assertRaisesRegex(RuntimeError, 'Source episode changed'):
            validate(out, marker, {'source.mp4': [124, 456]})
        artifact.write_bytes(b'interrupted/corrupt latent')
        with self.assertRaisesRegex(RuntimeError, 'missing or corrupt'):
            validate(out, marker, inputs)
        artifact.unlink()
        with self.assertRaisesRegex(RuntimeError, 'missing or corrupt'):
            validate(out, marker, inputs)
        marker.unlink()
        out.rmdir()

    def test_mono_stream_matches_native_posterior_and_resets(self):
        from fastwam.models.wan22.stereotok.vae2_2 import WanVAE_
        model = WanVAE_(dim=8, dec_dim=8, z_dim=48, dim_mult=[1, 2, 4, 4],
                        num_res_blocks=1, temperal_downsample=[False, True, True])
        model.fusion = torch.nn.Module()
        model.fusion.downsamples = torch.nn.Identity()
        model.eval().requires_grad_(False)
        path = ROOT / 'lingbot_va/wan_va/modules/utils.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if getattr(n, 'name', '') in ('patchify', 'WanVAEStreamingWrapper')]
        scope = {'torch': torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        codec = SimpleNamespace(model=model, requires_stereo=False,
                                binding={'format': 'stereotok-student-v3'})
        wrapper = scope['WanVAEStreamingWrapper'](codec)
        pixels = torch.rand(2, 3, 9, 32, 32) * 2 - 1
        with torch.no_grad():
            mean, logvar = model.encode_posterior(pixels)
        for _ in range(2):
            wrapper.clear_cache()
            encoded = torch.cat([wrapper.encode_chunk(pixels[:, :, :1]),
                                 wrapper.encode_chunk(pixels[:, :, 1:5]),
                                 wrapper.encode_chunk(pixels[:, :, 5:9])], dim=2)
            torch.testing.assert_close(encoded, torch.cat((mean, logvar), dim=1))
        wrapper.clear_cache()
        grouped = torch.cat([wrapper.encode_chunk(pixels[:, :, :1]),
                             wrapper.encode_chunk(pixels[:, :, 1:])], dim=2)
        torch.testing.assert_close(grouped, encoded, rtol=0, atol=0)
        wrapper.clear_cache()
        torch.testing.assert_close(wrapper.encode_chunk(pixels), encoded, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'four-frame'):
            wrapper.encode_chunk(pixels[:, :, :5])
        with self.assertRaisesRegex(ValueError, 'right eyes'):
            wrapper.encode_chunk(pixels[:, :, 1:5], pixels[:, :, 1:5])
        with self.assertRaisesRegex(ValueError, 'Reset'):
            wrapper.encode_chunk(pixels[:1, :, 1:5])

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
