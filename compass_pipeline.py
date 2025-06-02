from typing import Any, Callable, Dict, List, Optional, Union
from types import MethodType
import torch
from PIL import Image
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from utils.attn_utils import MyCustomAttnProcessor, AttentionStore, aggregate_padding_tokens
from utils.sd_utils import parse_module_name, prompt2idx, scale_resolution_to_multiple
from composition import centroids_to_kde, divergence_loss, generate_grid, compute_centroids

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
    
    def set_resolution_defaults(self):
        self.default_output_resolution = self.unet.config.sample_size * self.vae_scale_factor
        self.unet_depth = len(self.unet.config.block_out_channels) - 1
        self.total_downscale_factor = 2**self.unet_depth * self.vae_scale_factor

    def get_model_compatible_resolution(self, height=None, width=None, scale_to_default=True):
        if not height or not width:
            return self.default_output_resolution, self.default_output_resolution
        # Optionally scale minimum dimension to default output resolution
        min_dim = self.default_output_resolution if scale_to_default else None
        height, width = scale_resolution_to_multiple(height, width, self.total_downscale_factor, min_dim)
        return height, width
    
    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name, processor in self.unet.attn_processors.items():
            if "attn2" in name:
                module_name = name.replace(".processor", "")
                place_in_unet, level, instance = parse_module_name(module_name)
                layer_key = f"cross_{place_in_unet}_{level}_{instance}"
                self.attn_store.layer_metadata[layer_key] = name
                attn_procs[name] = MyCustomAttnProcessor(attention_store = self.attn_store, layer_key = layer_key)
                cross_att_count += 1
            else:
                attn_procs[name] = processor
        
        self.unet.set_attn_processor(attn_procs)
        self.attn_store.num_att_layers = cross_att_count
        self.attn_store.register_keys()

    """
    Adapted from https://github.com/huggingface/diffusers/blob/13e48492f0aca759dda5056481d32b641af0450f/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py
    """
    @torch.no_grad()
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
        target_map = None,
        loss_res = 16,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        run_compass: bool = False,
        **kwargs,
    ):
        """
        The call function to the pipeline for generation.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        ################################ CUSTOM LOGIC ########################################
        # 0 Additional check on height and width
        self.set_resolution_defaults()        
        height, width = self.get_model_compatible_resolution(height, width)
        #########################################################################################
        
        self._guidance_scale = guidance_scale
        self._cross_attention_kwargs = cross_attention_kwargs
    
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, 
            height, 
            width, 
            negative_prompt, 
            prompt_embeds, 
            negative_prompt_embeds, 
            callback_on_step_end_tensor_inputs
        )

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
            prompt, 
            device, 
            num_images_per_prompt, 
            self.do_classifier_free_guidance, 
            negative_prompt, 
            prompt_embeds=prompt_embeds, 
            negative_prompt_embeds=negative_prompt_embeds,
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
            batch_size * num_images_per_prompt, 
            num_channels_latents, 
            height, 
            width, 
            prompt_embeds.dtype, 
            device, 
            generator, 
            latents,
        )
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ############################# CUSTOM LOGIC ####################################
        # Extract EoT token indices from prompt tokens
        eot_indices = prompt2idx(self.tokenizer, prompt, eot_only=True)
        self.eot_tensor = torch.Tensor(eot_indices).repeat_interleave(num_images_per_prompt)
        if self.do_classifier_free_guidance:
            self.eot_tensor = torch.cat([torch.ones(batch_size * num_images_per_prompt),self.eot_tensor])
        self.eot_tensor = self.eot_tensor.to(device)

        # Register attention control
        _, _, latent_height, latent_width = latents.shape
        self.attn_store = AttentionStore(latent_height, latent_width, device, save_global_store=False)
        self.register_attention_control()

        loss_height = loss_res
        loss_width = latent_width / latent_height * loss_res
        grid = generate_grid(loss_height, loss_width, centered=True, grid_aspect="equal").to(device) # Shape [H, W, 2]
        target_map = target_map.to(device)
        ###############################################################################

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            
            for i, t in enumerate(timesteps):

                with torch.enable_grad():

                    latents = latents.requires_grad_(True)

                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input, 
                        t, 
                        encoder_hidden_states=prompt_embeds, 
                        cross_attention_kwargs=self.cross_attention_kwargs
                    ).sample
                    
                    self.unet.zero_grad()
                    
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    ######### CUSTOM LOGIC HERE ################ 
                    if run_compass:
                        # replaece with actuall loss function
                        # print(f"Number of centroids at timestep {i}: {len(self.attn_store.centroids)} with shape: {self.attn_store.centroids[0].shape}")
                        centroids = torch.stack([self.attn_store.get_eot_centroid(
                                                    self.attn_store.attention_store[key][-1]
                                                    ) for key in self.attn_store.attention_store.keys()
                                                ], dim=1)  # or your preferred source

                        saliency_pred = centroids_to_kde(centroids, grid, sigma=0.03)
                        loss = divergence_loss(saliency_pred, target_map)

                        print("latents requires_grad:", latents.requires_grad)
                        print("loss requires_grad:", loss.requires_grad)
                        print("latents in graph:", torch.autograd.grad(loss, [latents], retain_graph=True, allow_unused=True)[0] is not None)
                        grad_cond = torch.autograd.grad(loss, [latents], retain_graph=True)[0]
                        # loss.backward()
                        
                
                    ####################################################
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    noise_pred += self.eta * self.scheduler.sigmas[i] * grad_cond

                    
                    if i == len(timesteps) - 1 or (i + 1) % self.scheduler.order == 0:
                        progress_bar.update()

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

                    torch.cuda.empty_cache()

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