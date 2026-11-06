"""SD 1.5 + ControlNet Canny + стандартный SD 1.5 IP-Adapter."""
import os


def load_pipeline(c):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetPipeline
    from transformers import CLIPVisionModelWithProjection

    if not torch.cuda.is_available():
        raise RuntimeError("Run baseline inference on a CUDA GPU in Colab")
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    models = c["models"]
    controlnet = ControlNetModel.from_pretrained(**model_args(models["controlnet"]), torch_dtype=torch.float16, use_safetensors=True)
    ip = models["ip_adapter"]
    # Загружаем encoder явно: его revision закреплён независимо от поведения loader.
    encoder = CLIPVisionModelWithProjection.from_pretrained(
        ip["repo"], subfolder=ip["image_encoder_subfolder"], revision=ip["revision"], torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        **model_args(models["sd"]), controlnet=controlnet, image_encoder=encoder,
        torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(ip["repo"], subfolder=ip["subfolder"], weight_name=ip["weight_name"],
                         revision=ip["revision"], image_encoder_folder=None)
    pipe.set_ip_adapter_scale(c["ip_adapter_scale"])
    pipe.enable_vae_slicing()
    if c["cpu_offload"]:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def model_args(model):
    return {"pretrained_model_name_or_path": model["repo"], "revision": model["revision"]}
