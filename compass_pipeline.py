from typing import Any, Callable, Dict, List, Optional, Union
import torch
from PIL import Image
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from attention import MyCustomAttnProcessor, AttentionStore
from utils.sd_utils import parse_module_name, scale_resolution_to_factor, prompt2idx

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

    # def image2latent(self, image):
#     """
#     Prepare latents from an image or random noise.
#     """
#     image = image.to(device=self.device, dtype=self.dtype)
#     latents = self.vae.encode(image).latent_dist.mean * self.vae.config.scaling_factor
    
#     return latents

    def get_model_compatible_resolution(self, height, width):
        """
        Correct resolution
        """
        factor  = self.total_downsample_factor
        min_dim = self.default_output_resolution
        new_width, new_height = scale_resolution_to_factor(width, height, factor, min_dim=min_dim)
        
        return new_height, new_width

    def register_attention_control(self):
        for name, module in self.unet.named_modules():
            if hasattr(module, "is_cross_attention"):
                if module.is_cross_attention or name.startswith("mid"):
                    attn_type = "cross" if module.is_cross_attention else "self"
                    place_in_unet, level, instance = parse_module_name(name)
                    layer_key = f"{attn_type}_{place_in_unet}_{level}_{instance}"

                    # Log metadata information
                    self.attention_store.layer_metadata[attn_type][layer_key] = [name]

                    # Set custom processor 
                    logger.info(f"Registering custom {attn_type}-attention control for layer key {layer_key}")
                    module.set_processor(MyCustomAttnProcessor(self.attention_store, layer_key))
                    
                    
    def denoising_step(self, latents, t, prompt_embeds, requires_grad=True):
        # with torch.enable_grad() if requires_grad else torch.no_grad():
        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input, 
            t, 
            encoder_hidden_states=prompt_embeds, 
            cross_attention_kwargs=self.cross_attention_kwargs,
        ).sample

        # perform guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        return noise_pred

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
        height, width = self.get_model_compatible_resolution(height, width)

        self._guidance_scale = guidance_scale
        self._cross_attention_kwargs = cross_attention_kwargs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds, callback_on_step_end_tensor_inputs)

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
        self._num_timesteps = len(timesteps)

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

        # Remove gradients from unet
        for name, param in self.unet.named_parameters():
            param.requires_grad = False

        # PREPARE TOKEN INDEX TO MATCH BATCH LOGIC 
        eot_indices = prompt2idx(self.tokenizer, prompt)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                ######### CUSTOM LOGIC HERE ################ 
                latents = latents.requires_grad_(True) # Must track to gradients here
                noise_pred = self.denoising_step(latents, t, prompt_embeds, requires_grad=True)
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                self.unet.zero_grad()

                # replaece with actuall loss functoin
                # loss = compute_ca_loss(attn_map_integrated_mid, attn_map_integrated_up, bboxes=bboxes,
                #                         object_positions=object_positions) * cfg.inference.loss_scale

                grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

                latents = latents - grad_cond * self.scheduler.sigmas[i] ** 2
                latents = latents - step_size * grad_cond
               
                torch.cuda.empty_cache()
                    
                ####################################################

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
        
        with torch.no_grad():
            # Postprocess final outputs
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