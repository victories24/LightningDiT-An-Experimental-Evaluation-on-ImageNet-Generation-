# =============================================================================
# 子进程：只生成一段样本并保存为 npz，进程退出后 GPU 显存会被系统回收。
# 用法（在 LightningDiT 项目根目录下）：
#   python run_chunk_generate.py --base_path D:/DATA/Chen_Jin/LightningDiT-main --num 500 --chunk_out D:/DATA/Chen_Jin/LightningDiT-main/chunks/chunk_000.npz
# =============================================================================

import os
import sys
import argparse
import numpy as np
import torch
import yaml

# 必须在 import 模型前设置
os.environ["TORCH_COMPILE_DISABLE"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="LightningDiT 项目根目录")
    parser.add_argument("--num", type=int, required=True, help="本进程要生成的图片数量")
    parser.add_argument("--chunk_out", type=str, required=True, help="本段 npz 输出路径")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)
    num_images = args.num
    chunk_out = os.path.abspath(args.chunk_out)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 在项目根目录下运行，保证 tokenizer/configs 等相对路径正确
    os.chdir(base_path)
    if base_path not in sys.path:
        sys.path.insert(0, base_path)
    config_path = os.path.join(base_path, "configs/reproductions/lightningdit_xl_vavae_f16d32_800ep_cfg.yaml")
    vae_path = os.path.join(base_path, "pretrained/vavae-imagenet256-f16d32-dinov2.pt")
    ckpt_path = os.path.join(base_path, "pretrained/lightningdit-xl-imagenet256-800ep.pt")
    latent_stats_path = os.path.join(base_path, "pretrained/latents_stats.pt")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["sample"]["cfg_scale"] = 6.7

    # 导入并加载模型（每个子进程独立加载，退出时一起释放）
    from tokenizer.vavae import VA_VAE
    from models.lightningdit import LightningDiT_models
    from transport import create_transport, Sampler

    # VAE
    vae = VA_VAE(f'tokenizer/configs/{config["vae"]["model_name"]}.yaml')
    ckpt = torch.load(vae_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    vae_state = {k: v for k, v in state.items() if k.startswith(("encoder.", "decoder.", "quant_conv.", "post_quant_conv."))}
    vae.model.load_state_dict(vae_state, strict=False)
    vae.model = vae.model.to(device).eval()

    # Latent stats
    stats = torch.load(latent_stats_path, map_location="cpu")
    latent_mean = stats["mean"].to(device)
    latent_std = stats["std"].to(device)
    latent_multiplier = config["data"].get("latent_multiplier", 1.0)

    # DiT
    latent_size = config["data"]["image_size"] // config["vae"]["downsample_ratio"]
    model = LightningDiT_models[config["model"]["model_type"]](
        input_size=latent_size,
        num_classes=config["data"]["num_classes"],
        use_qknorm=config["model"]["use_qknorm"],
        use_swiglu=config["model"]["use_swiglu"],
        use_rope=config["model"]["use_rope"],
        use_rmsnorm=config["model"]["use_rmsnorm"],
        wo_shift=config["model"]["wo_shift"],
        in_channels=config["model"]["in_chans"],
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "ema" in ckpt:
        ckpt = ckpt["ema"]
    model.load_state_dict(ckpt)
    model = model.to(device).eval()

    # Sampler
    transport = create_transport(
        config["transport"]["path_type"],
        config["transport"]["prediction"],
        config["transport"]["loss_weight"],
        config["transport"]["train_eps"],
        config["transport"]["sample_eps"],
        use_cosine_loss=config["transport"]["use_cosine_loss"],
        use_lognorm=config["transport"]["use_lognorm"],
    )
    sampler = Sampler(transport)

    # 本进程内用 batch_size=1 逐张生成，减少显存
    num_classes = config["data"]["num_classes"]
    cfg_scale = config["sample"]["cfg_scale"]
    cfg_interval_start = config["sample"]["cfg_interval_start"]
    timestep_shift = config["sample"]["timestep_shift"]
    num_steps = config["sample"]["num_sampling_steps"]

    sample_fn = sampler.sample_ode(
        sampling_method=config["sample"]["sampling_method"],
        num_steps=num_steps,
        atol=config["sample"]["atol"],
        rtol=config["sample"]["rtol"],
        reverse=config["sample"]["reverse"],
        timestep_shift=timestep_shift,
    )
    using_cfg = cfg_scale > 1.0

    list_imgs = []
    for n in range(num_images):
        labels = torch.randint(0, num_classes, (1,), device=device)
        z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000], device=device)
            y = torch.cat([labels, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=labels)
            model_fn = model.forward

        with torch.no_grad():
            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
        samples = (samples * latent_std) / latent_multiplier + latent_mean
        img = vae.decode_to_images(samples)
        img = img[0]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        list_imgs.append(img.astype(np.uint8))

    arr = np.stack(list_imgs, axis=0)
    os.makedirs(os.path.dirname(chunk_out) or ".", exist_ok=True)
    np.savez_compressed(chunk_out, arr_0=arr)
    print(f"Saved chunk: {chunk_out} shape={arr.shape}")


if __name__ == "__main__":
    main()
