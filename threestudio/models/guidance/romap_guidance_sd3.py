import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base_sd3 import PromptProcessorOutput_sd3 as PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *
from diffusers import StableDiffusion3Pipeline
import copy
import math
import matplotlib.pyplot as plt

import numpy as np
import rembg
from tqdm import tqdm

global idx
idx = 0

global block_idx
block_idx = 0
import os
from typing import Optional, Union, List, Dict, Any
from jaxtyping import Float
from torch import Tensor


from diffusers.models.attention_processor import Attention
from typing import Optional


class RoMaPAttnProcessorWithHook:
    """
    Custom Joint Attention Processor.
    Uses an externally supplied sdpa (scaled dot-product attention) object, which ensures that hook functions are called during attention computation.
    """

    def __init__(self, sdpa_object):
        # 1. Accepts an sdpa object (with hooks) as an argument during initialization.
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("This processor requires PyTorch 2.0 or newer.")
        self.sdpa = sdpa_object

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Implements joint-attention logic using the externally provided sdpa object (not the built-in).
        This guarantees that any registered hook is triggered within the sdpa call.
        """
        residual = hidden_states

        # If the input is 4D, reformat to sequence (B, N, C) for attention
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        # Same for encoder_hidden_states (context)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # Project queries, keys, and values
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # Concatenate main and encoder-projected QKV for multi-source attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        # Split into heads for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 2. Call the externally supplied sdpa (scaled dot-product attention) module.
        #    This ensures that hooks attached to sdpa will be executed.
        hidden_states, _ = self.sdpa(
            query, key, value, dropout_p=0.0, is_causal=False
        )

        # Recombine heads, restore batch/sequence shape
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split main and context outputs
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :residual.shape[1]],
            hidden_states[:, residual.shape[1]:],
        )

        # Final output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Restore 4D shape if the input was originally 4D
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states

def seed_everything(seed):
    """
    Set seeds for Python, Numpy, and PyTorch (including CUDA) to ensure reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


class SDPA(nn.Module):
    def __init__(self, hook_fn, last):
        super(SDPA, self).__init__()  # Ensure super class is initialized
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.hook_fn = hook_fn
        self.last = last

    def forward(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype).to(query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        
        attn_weight = query@key.transpose(-1,-2)*scale_factor
        attn_weight +=attn_bias.to(attn_weight.device)
        
        if self.last:
            q_attn = query @ query.transpose(-2,-1) *scale_factor
            k_attn = key @ key.transpose(-2,-1) *scale_factor
            temp_attn_weight = torch.softmax(q_attn, dim=-1) + torch.softmax(k_attn, dim=-1)
            output_attn_weight = temp_attn_weight
        else:
            output_attn_weight = attn_weight 
        
        
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        
        new_output_attn_weight = self.hook_fn(None, None, (value, output_attn_weight))
        return attn_weight @ value, output_attn_weight
@threestudio.register("romap-sd3")
class RectifiedFlowGuidance(BaseModule):
    """
    Implements the Rectified Flow Guidance module using Stable Diffusion 3.
    Integrates custom attention processors and hooks for detailed attention map extraction.
    """
    @dataclass
    class Config(BaseModule.Config):
        pretrained_model_name_or_path: str = ""
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 7.5
        grad_clip: Optional[Any] = field(default_factory=lambda:[0,2.0,8.0,1000])
        half_precision_weights: bool = True
        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        seed = 0
        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Rectified Flow ...")

        self.weights_dtype = torch.float16 if self.cfg.half_precision_weights else torch.float32

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline

        # Load Stable Diffusion 3 pipeline (with no safety checker/tokenizer)
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            tokenizer=None,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        ).to(self.device)

        # Inject custom attention hooks into all relevant attention modules
        for name, module in pipe.transformer.named_modules():
            if name.split('.')[-1] == 'attn':
                # If the attention block has a layer id '23', mark it as 'last'
                if name.split('.')[-2] == '23':
                    module.sdpa = SDPA(self.hook_fn, last=True)
                else:
                    module.sdpa = SDPA(self.hook_fn, last=False)

        # Build processors for each attention layer using the injected sdpa
        attn_processors = {}
        for name, module in pipe.transformer.named_modules():
            if name.endswith("attn"):
                if hasattr(module, 'sdpa'):
                    # Create and store a custom processor instance for this layer
                    custom_processor_instance = RoMaPAttnProcessorWithHook(sdpa_object=module.sdpa)
                    key = f"{name}.processor"
                    attn_processors[key] = custom_processor_instance
                else:
                    print(f"Warning: Module {name} ends with 'attn' but has no 'sdpa' attribute.")
        if attn_processors:
            pipe.transformer.set_attn_processor(attn_processors)
            print(f"Custom processor with hooks successfully set for {len(attn_processors)} 'attn' blocks.")
        else:
            print("No 'attn' attention processors found to apply the custom processor.")

        self.submodules = SubModules(pipe=pipe)

        # Optional memory-efficient attention settings
        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info("PyTorch 2.0 uses memory efficient attention by default.")
            elif not is_xformers_available():
                threestudio.warn("xformers is not available, memory efficient attention is not enabled.")
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        # Remove unused text encoders to reduce memory usage
        del self.pipe.text_encoder
        del self.pipe.text_encoder_2
        del self.pipe.text_encoder_3
        cleanup()

        # Freeze VAE and transformer model parameters (no gradient updates)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        # Camera embedding setup (placeholder: dimension hardcoded)
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        )

        # Scheduler setup
        self.scheduler = self.pipe.scheduler
        self.scheduler_sample = self.pipe.scheduler
        self.timesteps = self.scheduler_sample.timesteps
        self.grad_clip_val: Optional[float] = None
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()
        self.noise_scheduler_copy = copy.deepcopy(self.pipe.scheduler)
        threestudio.info(f"Loaded Rectified Flow!")

        # Range of attention blocks to extract attention maps from
        self.block_list = range(0, 23)
        self.batch_size = 8

        # For storing extracted attention/segmentation maps and selection indices
        self.attention_maps = None
        self.segmentation_maps = None
        self.target_index = None

    def set_path(self, get_save_path):
        """Set the function used to generate save paths for outputs."""
        self.get_save_path = get_save_path

    def set_obj_list(self, obj_list):
        """Set the list of segmentation/object prompt groups."""
        self.obj_list = obj_list
    
    # The hook receives attention module, its input, and output.
    # It accumulates per-object attention for the chosen block range and stores it on the class.
    def hook_fn(self, module, input, output):
        if self.is_attn:
            global idx
            global block_idx
            attn_weight = output[1]  # e.g.: torch.Size([1, 24, 3185, 3185])
            # Block map reset per new block
            if block_idx != idx // 24:
                block_idx = idx // 24
                self.attention_maps = None

            if idx % 24 in self.block_list:
                batch_size, channel, _, _ = attn_weight.shape
                if self.attention_maps is None:
                    self.attention_maps = np.zeros((len(self.obj_list), batch_size, 32, 32))
                for i, obj_groups in enumerate(self.obj_list):
                    group_weights = torch.zeros((batch_size, channel, 1024)).to("cuda")
                    for obj_group in obj_groups:
                        # Add up the attention over all tokens in group
                        group_weights += attn_weight[:, :, 1024 + obj_group, :1024]
                    group_mean = group_weights.mean(1).reshape(-1, 32, 32) / len(obj_groups)
                    self.attention_maps[i, ...] += group_mean.detach().cpu().numpy()
        idx += 1
        return None

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        """
        Set the min/max denoising step percent (relative to total scheduler steps).
        """
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        """
        Returns the underlying Stable Diffusion pipeline (SD3).
        """
        return self.submodules.pipe

    @property
    def transformer(self):
        """
        Returns the transformer module (usually MMDiT) from the pipeline.
        """
        return self.submodules.pipe.transformer

    @property
    def vae(self):
        """
        Returns the VAE decoder from the pipeline.
        """
        return self.submodules.pipe.vae

    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility.
        """
        self.seed = seed

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        """
        Performs classifier-free guidance sampling and denoising loop for SD3/SD3M[6].
        """
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        # Prepare Gaussian noise latents for the initial step
        latents = pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            self.weights_dtype,
            device,
            generator,
        )

        for i, t in enumerate(timesteps):
            # Expand the latents for classifier-free guidance (uncond + cond)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual using the UNet with/without class embedding
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                ).sample

            # Split into text and unconditional branches and apply classifier-free guidance
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # Compute the previous latents via the scheduler's step
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        # Decode final latents with the VAE
        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # Final conversion to [B, H, W, 3]
        images = images.permute(0, 2, 3, 1).float()
        return images

    def sample(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        """
        High-level SD3/SD3M text-to-image sampling based on view-dependent embeddings.
        """
        # Get view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        # Use a seed for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            cross_attention_kwargs=kwargs.get('cross_attention_kwargs', None),
            generator=generator,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_transformer(
        self,
        transformer,
        latents,
        t,
        prompt_embeds,
        pooled_prompt_embeds
    ) -> Float[Tensor, "..."]:
        """
        Forward pass through the transformer model using properly cast inputs,
        consistent with SD3/SD3M design and precision requirements[5][6].
        """
        input_dtype = latents.dtype
        return transformer(
            hidden_states=latents.to(self.weights_dtype),
            timestep=t.to(self.weights_dtype),
            encoder_hidden_states=prompt_embeds.to(self.weights_dtype),
            pooled_projections=pooled_prompt_embeds.to(self.weights_dtype),
            return_dict=False,
        )[0].to(input_dtype)

       
    @torch.cuda.amp.autocast(enabled=False)
    def forward_custom_transformer(
        self,
        transformer,
        latents,
        t,
        prompt_embeds,
        pooled_prompt_embeds,
        edit_prompt_embeds=None,
        edit_pooled_prompt_embeds=None,
        uncond_prompt_embeds=None,
        uncond_pooled_prompt_embeds=None
    ) -> Float[Tensor, "..."]:
        output_dtype = latents.dtype
       
        
        # input_dtype = latents.dtype
        input_dtype = self.weights_dtype
        # temp_transformer = self.submodules.pipe.transformer
        # self.submodules.pipe.transformer = self.custom_transformer
       
        if edit_prompt_embeds is not None:
            result = transformer(
                self.submodules.pipe.transformer.to(input_dtype),
                hidden_states=latents.to(input_dtype),
                timestep=t.to(input_dtype),
                encoder_hidden_states=prompt_embeds.to(input_dtype),
                pooled_projections = pooled_prompt_embeds.to(input_dtype),
                edit_encoder_hidden_states = edit_prompt_embeds.to(input_dtype),
                edit_pooled_projections = edit_pooled_prompt_embeds.to(input_dtype),
                uncond_encoder_hidden_states = uncond_prompt_embeds.to(input_dtype),
                uncond_pooled_projections = uncond_pooled_prompt_embeds.to(input_dtype),
                return_dict=False,
            )[0].to(output_dtype)
        else:
            result = transformer(
                self.submodules.pipe.transformer.to(input_dtype),
                hidden_states=latents.to(input_dtype),
                timestep=t.to(input_dtype),
                encoder_hidden_states=prompt_embeds.to(input_dtype),
                pooled_projections = pooled_prompt_embeds.to(input_dtype),
                edit_encoder_hidden_states = None,
                edit_pooled_projections = None,
                uncond_encoder_hidden_states = None,
                uncond_pooled_projections = None,
                return_dict=False,
            )[0].to(output_dtype)
        # self.submodules.pipe.transformer = temp_transformer
        return result

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = torch.clamp(imgs, min=0, max=1)
        imgs = self.pipe.image_processor.preprocess(imgs)
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample()
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        #! interpolate 
        B, C, H, W = latents.shape
        new_h = H + H%2
        new_w = W + W%2
        latents = F.interpolate(
            latents, (new_h, new_w), mode="bilinear", align_corners=False
        )
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        image_height: int = 512,
        image_width: int = 512,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        
        image = (image * 0.5 + 0.5).clamp(0, 1)
        
        image = F.interpolate(
            image, (image_height, image_width), mode="bilinear", align_corners=False
        )
        return image.to(input_dtype)

        
       

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding
    def get_sigmas(self,timesteps, n_dim=4, dtype=torch.float16, scheduler=None):
        
        if scheduler is None:
            scheduler = self.noise_scheduler_copy
        
        sigmas = scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    
    def compute_exact_inversion(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        """
        Performs iRFDS (inverse Rectified Flow Distillation Sampling) inversion.
        This inverts latents back to noise using rectified flow, as used for editing/inversion tasks.
        """
        index = 0
        B = latents.shape[0]

        # Split conditional/unconditional text and pooled embeddings
        (text_embeddings, uncond_text_embeddings) = text_embeddings_vd[0].squeeze().chunk(2)
        (text_embeddings_pooled, uncond_text_embeddings_pooled) = text_embeddings_vd[1].squeeze().chunk(2)
        prompt_embeds = torch.cat([text_embeddings, uncond_text_embeddings])
        pooled_prompt_embeds = torch.cat([text_embeddings_pooled, uncond_text_embeddings_pooled])

        with torch.no_grad():
            step = 5  # The number of inversion steps

            # Get the specific timesteps for inversion steps
            indices = torch.tensor(
                range(self.min_step, self.max_step + 1, int((self.max_step - self.min_step) / step))
            )
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=self.device)
            timesteps, _ = torch.sort(timesteps, descending=False)
            sigmas = self.get_sigmas(
                timesteps, n_dim=latents.ndim, dtype=latents.dtype
            ).to("cuda")

            # Initialize noisy latents and starting random noise
            latents_noisy = latents.clone()
            y1 = torch.randn_like(latents_noisy)

            for t in range(step):
                # Prepare a tensor filled with the current timestep value for all items in the batch
                curr_t = torch.full(
                    (latents_noisy.size(0),), timesteps[t], device=self.device
                )
                self.attention_maps = None

                # Get unconditional noise prediction from the transformer
                uncon_noise_pred = self.forward_transformer(
                    self.transformer,
                    latents_noisy,
                    curr_t,
                    text_embeddings,
                    text_embeddings_pooled,
                )
                # Calculate conditional noise prediction for rectified flow update
                con_noise_pred = (y1 - latents_noisy) / (1 - (timesteps[t] / 1000))

                # Compute vector field adjustment (i.e., controlled flow)
                noise_pred = uncon_noise_pred + 0.5 * (con_noise_pred - uncon_noise_pred)

                # Update latents using the vector field step
                latents_noisy = latents_noisy + noise_pred * (sigmas[t + 1] - sigmas[t])
                # Optional: Stop after 4th iteration for computational reasons
                if t == 3:
                    break

    def compute_grad_romap(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        """
        Computes the gradient for RFDS-Rev using classifier-free guidance (CFG).
        This combines unconditional and conditional predictions, following state-of-the-art practices for text-conditional diffusion models[2][4][8][9].
        """
        B = latents.shape[0]
        # Split the text embeddings into conditional and unconditional branches
        (text_embeddings, uncond_text_embeddings) = text_embeddings_vd[0].squeeze().chunk(2)
        (text_embeddings_pooled, uncond_text_embeddings_pooled) = text_embeddings_vd[1].squeeze().chunk(2)
        prompt_embeds = torch.cat([text_embeddings, uncond_text_embeddings])
        pooled_prompt_embeds = torch.cat([text_embeddings_pooled, uncond_text_embeddings_pooled])

        with torch.no_grad():
            # Sample real Gaussian noise for the batch
            real_noise = torch.randn_like(latents)
            
            # Randomly select time indices for flow matching (per batch element)
            indices = torch.randint(self.min_step, self.max_step + 1, (B,))
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=self.device)

            # Compute flow-matching sigmas (noise weights per time step)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            
            # Forward process: add noise according to flow matching schedule
            latents_noisy = sigmas * real_noise + (1.0 - sigmas) * latents
            
            # ---- iRFDS step: predict velocity under unconditional branch (CFG off) ----
            self.is_cfg = False
            velocity = self.forward_transformer(
                self.transformer,
                latents_noisy,
                timesteps,
                uncond_text_embeddings,
                uncond_text_embeddings_pooled,
            )
            # Compute step size for integration along the flow
            stepsize = 1 - sigmas
            # Compute noise for the RFDS step, combined from velocity and target
            noise = real_noise + stepsize * (velocity + latents - real_noise)
            
            # ---- RFDS (CFG) step: run conditional/unconditional (CFG on) ----
            latents_noisy = sigmas * noise + (1.0 - sigmas) * latents
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            self.is_cfg = True
            velocity = self.forward_transformer(
                self.transformer,
                latent_model_input,
                torch.cat([timesteps] * 2),
                prompt_embeds,
                pooled_prompt_embeds,
            )

        # Split output into conditional (text) and unconditional (null) branches
        velocity_text, velocity_null = velocity.chunk(2)
        # Random weighting for batch element (for importance weighting)
        u = torch.normal(mean=0, std=1, size=(B,), device=self.device)
        weighting = torch.nn.functional.sigmoid(u)

        # Classifier-free guidance: combine unconditional and conditional predictions
        model_pred = velocity_null + self.cfg.guidance_scale * (velocity_text - velocity_null)
        # See [2][4][8][9] for details on this formula

        return model_pred, noise, weighting

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        """
        Converts an RGB image tensor to the model's latent representation,
        using VAE encoding unless 'rgb_as_latents' is set (then performs resize/interpolation).
        """
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            latents = self.encode_images(rgb_BCHW)
        return latents
    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        is_attn=False,
        attn_mask=None,
        global_step=None,
        edit_obj_list=None,
        orig_image=None,
        edit_prompt_utils=None,
        is_spo=False,
        **kwargs,
    ):
        """
        Main forward method for model inference and loss calculation.
        - Handles text/image processing, camera conditioning, and segmentation-guided attention.
        """

        # Store task/context flags
        self.is_attn = is_attn
        self.attn_mask = attn_mask
        self.edit_obj_list = edit_obj_list
        self.global_step = global_step
        self.edit_prompt_utils = edit_prompt_utils

        batch_size = rgb.shape[0]

        # Reorder channels for PyTorch: [B, H, W, C] -> [B, C, H, W]
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        # Convert image to latents using VAE, or just resize if rgb_as_latents is True
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        # Compute original image's latents if provided (for inversion/editing tasks)
        if orig_image is not None:
            orig_rgb_BCHW = orig_image.permute(0, 3, 1, 2)
            self.orig_latents = self.get_latents(orig_rgb_BCHW, rgb_as_latents=rgb_as_latents).detach().clone().requires_grad_(False)
        else: 
            self.orig_latents = None

        # Obtain view-dependent text embeddings (camera-aware prompt conditioning)
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances,
            view_dependent_prompting=True,
        )

        # If using edit-specific prompt for editing region or mask, also get those embeddings
        if self.edit_prompt_utils is not None: 
            self.edit_text_embeddings_vd = self.edit_prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances,
                view_dependent_prompting=True,
            )

        # If the edit mask/object list is given (mask segment-specific flows)
        if self.edit_obj_list is not None:
            B, C, H, W = latents.shape            
            self.reshaped_attn_mask = self.attn_mask.repeat(1, C, 1, 1)
            self.reshaped_attn_mask = F.interpolate(
                self.reshaped_attn_mask.to(torch.float32),
                size=(H, W), mode='bilinear', align_corners=False
            )

        # Set camera conditioning type (extrinsic matrix or MVP matrix)
        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        # --- Segmentation (Attention) Inversion Mode ---
        if self.is_attn and edit_obj_list is None:
            # If only attention/segmentation is requested, run iRFDS inversion
            self.compute_exact_inversion(latents, text_embeddings_vd, camera_condition)
            return {
                "attention": self.attention_maps,
            }

        # --- RFDS Loss Calculation Mode ---
        else: 
            grad, noise, noise_pred = self.compute_grad_romap(
                latents, text_embeddings_vd, camera_condition
            )
            grad = torch.nan_to_num(grad)
            target = grad.detach()

            # Segmentation-aware (masked) loss: only compute loss over selected objects/masks
            if self.edit_obj_list is not None:
                reshaped_attn_mask = self.reshaped_attn_mask > 0.5
                loss_rfds = F.mse_loss(
                    noise[reshaped_attn_mask] - latents[reshaped_attn_mask],
                    target[reshaped_attn_mask],
                    reduction='mean'
                ) / batch_size
            else:
                # Normal unmasked loss: mean squared error over all points
                loss_rfds = F.mse_loss(
                    noise - latents,
                    target,
                    reduction='mean'
                ) / batch_size

            return {
                "loss_rfds": loss_rfds,
                "grad_norm": noise_pred.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
                # "attention": self.attention_maps,
            }

           

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated inc
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
        
  
    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        
        import inspect
        r"""
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
                `num_inference_steps` and `sigmas` must be `None`.
            sigmas (`List[float]`, *optional*):
                Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
                `num_inference_steps` and `timesteps` must be `None`.

        Returns:
            `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """
        if timesteps is not None and sigmas is not None:
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

