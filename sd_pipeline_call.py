from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
from torch.nn import functional as F
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from utils.attention import MyCustomAttnProcessor, AttentionStore
from utils.sd_utils import parse_module_name, make_dims_compatible, prompt2idx

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CompASSPipeline(StableDiffusionPipeline):
    """
    Initialize the CompASSPipeline, inheriting from StableDiffusionPipeline.
    """
    def get_resolution_defaults(self):
        """
        Compute and store default model parameters.
        """
        self.default_output_resolution = self.unet.config.sample_size * self.vae_scale_factor
        self.unet_depth = len(self.unet.config.block_out_channels) - 1
        self.total_downsample_factor = 2**self.unet_depth * self.vae_scale_factor

    def register_attention_control(self):
        down_exp = 0  # Initialize resolution factor
        max_exp = down_exp
        for name, module in self.unet.named_modules():
            if hasattr(module, "is_cross_attention"):
                if module.is_cross_attention or name.startswith("mid"):
                    attn_type = "cross" if module.is_cross_attention else "self"
                    place_in_unet, level, instance = parse_module_name(name)
                    layer_key = (attn_type, place_in_unet, level, instance)
                    # Set custom processor 
                    module.set_processor(MyCustomAttnProcessor(self.attention_store, layer_key))
                    # Log metadata information
                    self.attnstore.layer_metadata[attn_type][layer_key] = (2**down_exp, name)
                    
            # Track resolution through downsampling/upsampling modules
            elif "sample" in name.split(".")[-1]:
                down_exp = down_exp + 1 if "down" in name else down_exp - 1
                max_exp = max(max_exp, down_exp)

       # Log a warning if the calculated UNet depth doesn't match the expected depth
        if max_exp != self.unet_depth:
            logger.warning(f"Calculated UNet depth ({max_exp}) does not match expected depth ({self.unet_depth}). Logic is not compatible with the model architecture.")

        for attn_type, metadata in self.attnstore.layer_metadata.items():
            for layer_key, (res_factor, name) in metadata.items():
                if layer_key[1] == "mid":  # Ensure it's a midblock
                    self.attnstore.layer_metadata[attn_type][layer_key] = (2**max_exp, name)
                logger.info(f"Registered {attn_type} attention for layer key {layer_key} with downsample factor {2**down_exp}")

    @torch.no_grad() ## Use this for now while we're just extracting
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        The call function to the pipeline for generation.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. CUSTOM Default height and width to unet
        self.get_resolution_defaults()
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        width, height = make_dims_compatible(width, height, self.total_downsample_factor, min_dim=self.default_output_resolution)

        # Pass height and width to the denoising loop
        cross_attention_kwargs = cross_attention_kwargs or {}
        cross_attention_kwargs.update({"img_height": height, "img_width": width})

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt, self.do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # CUSTOM. Register ATTENTION CONTROL 
        self.attention_store = AttentionStore()
        self.register_attention_control()

        # PREPARE TOKEN INDEX TO MATCH BATCH LOGIC 
        eot_indices = prompt2idx(self.tokenizer, prompt)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                ######### CUSTOM LOGIC HERE ################
                # with torch.enable_grad(): Use this when we implement actual guidances
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds, cross_attention_kwargs=self.cross_attention_kwargs,
                ).sample

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                ####################### THIS IS WHERE YOU NEED TO ADD ATTENTION GUIDANCE #############################

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if XLA_AVAILABLE:
                    xm.mark_step()

        # Output
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        do_denormalize = [True] * image.shape[0] if has_nsfw_concept is None else [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)