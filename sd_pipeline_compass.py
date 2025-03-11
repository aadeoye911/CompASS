import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.transforms import Normalize, ToTensor, Compose
from utils.attn_utils import AttentionStore
from utils.sd_utils import resize_image, extract_attention_info

class CompASSPipeline(StableDiffusionPipeline):
    """
    Initialize the CompASSPipeline, inheriting from StableDiffusionPipeline.
    """
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):

        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

        # Load resolution defaults
        self.default_output_resolution = None # Default output resolution for model
        self.unet_depth = None
        self.total_downsample_factor = None # Total VAE+UNet downscaling
        self.get_resolution_defaults()

        # Additional components for attention tracking
        self.attnstore = AttentionStore()
        self.setup_hooks()
        
        self.diffused_images = []
        self.height = None
        self.width = None
        self.empty_embeds = None
        self.get_empty_embeddings()


    def get_resolution_defaults(self):
        """
        Compute and store default model parameters.
        """
        self.default_output_resolution = self.unet.config.sample_size * self.vae_scale_factor
        self.unet_depth = len(self.unet.config.block_out_channels) - 1
        self.total_downsample_factor = 2**self.unet_depth * self.vae_scale_factor


    def setup_hooks(self):
        """
        Set up hooks on all cross-attention layers.
        """
        self.hooks = []
        down_exp = 0  # Initialize resolution factor
        max_exp = down_exp
        for name, module in self.unet.named_modules():

            # Place hook on attention modules
            if "Attention" in type(module).__name__:
                attn_type = "cross" if module.is_cross_attention else "self"
                place_in_unet, level, instance = extract_attention_info(name)
                layer_key = (attn_type, place_in_unet, level, instance)
                self.attnstore.attn_metadata[attn_type][layer_key] = (2**down_exp, name)
                self.hooks.append(module.register_forward_hook(self._hook_fn(layer_key)))

            # Track resolution through downsampling/upsampling modules
            elif "sample" in name.split(".")[-1]:
                down_exp = down_exp + 1 if "down" in name else down_exp - 1
                max_exp = max(max_exp, down_exp)

        # Ensure consistency between up/down resolutions
        assert max_exp == self.unet_depth, "Invalid UNet depth calculation"
        
        # Reassign resolution factor for midblocks
        for attn_type, layer_metadata in self.attnstore.attn_metadata.items():
            for layer_key, (res_factor, name) in layer_metadata.items():
                if layer_key[1] == "mid":  # Ensure it's a midblock
                    self.attnstore.attn_metadata[attn_type][layer_key] = (2**max_exp, name)

        print(f"Number of hooks initialised: {len(self.hooks)}")
    
    
    def _hook_fn(self, layer_key):
        """
        Hook function to capture attention scores.
        """
        def hook(module, input, output):
            try:
                print(f"Processsing attention with key: {layer_key} with input of length {len(input)}")
                if len(input) > 1:
                    print(input[0].shape, input[1].shape)
                query = module.to_q(input[0])
                key = module.to_k(self.empty_embeds[0] if layer_key[0] == "cross" else input[0])
                attn_probs = (module.get_attention_scores(query, key)).detach().cpu()
                self.attnstore.store(attn_probs, layer_key)
            except Exception as e:
                print(f"Error processing attention scores for layer {layer_key}: {e}")
        return hook
    

    def clear_hooks(self):
        """
        Remove all hooks from the model.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


    def reset(self):
        """
        Reset stored latents, images, and attention maps.
        """
        self.attnstore.reset()
        self.diffused_images.clear()


    def get_empty_embeddings(self, prompt="", batch_size=1):
        """
        Tokenize the prompt and get text embeddings.
        """
        self.empty_embeds = self.encode_prompt(prompt, self.device, batch_size, False)
        print(f"Initiatilized empty embeddings where ({self.empty_embeds[0].shape}, {self.empty_embeds[1]})")


    def preprocess_image(self, image, min_dim=None, factor=None):
        """
        Convert PIL image into a torch.Tensor with model-compatible dimensions.
        """
        min_dim = min_dim if min_dim is not None else self.default_output_resolution
        factor = factor if factor is not None else self.total_downsample_factor
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = resize_image(image, min_dim, factor)
        transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

        return transform(image).unsqueeze(0).to(self.dtype)
    

    def image2latent(self, image, timesteps, num_images_per_prompt=None, seed=42):
        """
        Prepare latents from an image or random noise.
        """
        image = image.to(device=self.device, dtype=self.dtype)
        batch_size = len(timesteps)
        if image.shape[0] < batch_size:
            if batch_size % image.shape[0] == 0:
                image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
            else: 
                raise ValueError(f"Cannot duplicate `image` of batch size {image.shape[0]} to batch_size {batch_size} ")
            
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = self.vae.encode(image).latent_dist.mean * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape, generator=generator, device=self.device)
        
        # Apply timestep-dependent noise across batch channel
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        return noisy_latents

    # def init_latent(size, generator=None, dtype=torch.float32):
    #     """
    #     Generate random noise latent tensor for Stable Diffusion.
    #     """
    #     if isinstance(generator, list):
    #         if len(generator) > batch_size:
    #             print(f'generator longer than batch size. truncationg list to match batch')
    #             generator = generator[:batch_size]

    #     return torch.randn((batch_size, num_channels, height, width), generator=generator, dtype=dtype)


    def extract_attention_maps(self, image, timesteps, seed=42):
        batch_size = len(timesteps)
        image = image.to(self.device)
        timesteps.to(self.device)
        latents = self.image2latent(image, timesteps)

        if self.empty_embeds[0].shape[0] != batch_size:
            # prompt_embeds, no_embeds = self.empty_embeds
            # prompt_embeds = torch.cat([prompt_embeds] * batch_size, dim=0)
            self.get_empty_embeddings(batch_size=batch_size)

        latents = self.image2latent(image, timesteps, seed)
        with torch.no_grad():
            unet_output = self.unet(latents, timesteps, encoder_hidden_states=self.empty_embeds[0], return_dict=True)
            noise_pred = unet_output["sample"]
            latents = self.scheduler.step(noise_pred, timesteps, latents)["prev_sample"]
            if self.device == "cuda":
                torch.cuda.empty_cache()
