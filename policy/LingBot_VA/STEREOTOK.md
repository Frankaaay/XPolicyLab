# LingBot-VA with the StereoTok tokenizer

This branch lets LingBot-VA use the StereoTok stereo tokenizer in place of the Wan2.2
VAE (`lingbot_va/wan_va/modules/stereotok.py`). StereoTok keeps the Wan2.2 latent
shape, so the LingBot-VA transformer and its token layout are unchanged. The left and
right views of each camera are encoded jointly into one latent per camera.

## Dependencies

The StereoTok codec is imported from the FastWAM StereoTok branch
(`fastwam.models.wan22.stereotok_vae`); install it into the LingBot-VA environment:

```bash
pip install -e <FastWAM checkout on its StereoTok branch>
```

## Latent cache

LingBot-VA trains on pre-encoded latents, so the cache must be built with the same
tokenizer and input mode that training and deployment use:

```bash
LINGBOT_STEREOTOK_CHECKPOINT=/path/to/stereo.pth \
LINGBOT_RIGHT_CAMERA_MAP=right-camera-map.json \
bash process_data.sh ...
# or, for RoboTwin / LIBERO native HDF5:
python process_native_hdf5.py --stereotok-checkpoint /path/to/stereo.pth ...
```

The right-camera map names the true right eye of every configured left camera. Use
`--stereotok-input-mode mono-left` for the monocular path. The dataset refuses a cache
whose codec binding differs from the configured tokenizer.

## Training and deployment

`lingbot_va/wan_va/configs/shared_config.py` reads `LINGBOT_STEREOTOK_CHECKPOINT` and
`LINGBOT_STEREOTOK_INPUT_MODE` (`stereo` or `mono-left`); set them for training and for
`wan_va_server.py`. The RoboTwin recipe: 8 GPUs, batch 1 per GPU, 50k updates, seed 42;
inference with 25 video and 50 action denoising steps, guidance 5 / 1.

## Evaluation

RoboTwin clean/randomized evaluation uses the frozen shared plans and the evaluation entry
of the FastWAM StereoTok branch (`scripts/c2r/evaluate.py`, `model="lingbot"`), which drives
the running server through `lingbot_va/evaluation/robotwin/shared_plan_policy.SharedPlanPolicy`
(`host`, `port`, `stereo`, `video_guidance_scale`, `action_guidance_scale`). Stereo LIBERO
inputs are rendered with `utils/render_libero_stereo.py` and exported with
`utils/export_libero_stereo_lerobot.py`.
