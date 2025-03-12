import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.transforms import Normalize, ToTensor, Compose
from utils.attn_utils import AttentionStore
from utils.sd_utils import resize_image, extract_attention_metadata

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
        self.latent_height = None
        self.latent_width = None
        self.prompt_embeds = self.get_empty_embeddings()

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
                place_in_unet, level, instance = extract_attention_metadata(name)
                layer_key = (attn_type, place_in_unet, level, instance)
                self.attnstore.layer_metadata[attn_type][layer_key] = (2**down_exp, name)
                self.hooks.append(module.register_forward_hook(self._hook_fn(layer_key)))

            # Track resolution through downsampling/upsampling modules
            elif "sample" in name.split(".")[-1]:
                down_exp = down_exp + 1 if "down" in name else down_exp - 1
                max_exp = max(max_exp, down_exp)

        # Ensure consistency between up/down resolutions
        assert max_exp == self.unet_depth, "Invalid UNet depth calculation"
        
        # Reassign resolution factor for midblocks
        for attn_type, metadata in self.attnstore.layer_metadata.items():
            for layer_key, (res_factor, name) in metadata.items():
                if layer_key[1] == "mid":  # Ensure it's a midblock
                    self.attnstore.layer_metadata[attn_type][layer_key] = (2**max_exp, name)

        print(f"Number of hooks initialised: {len(self.hooks)}")
    
    def _hook_fn(self, layer_key):
        """
        Hook function to capture attention scores.
        """
        def hook(module, input, output):
            try:
                query = module.to_q(input[0])
                key = module.to_k(self.prompt_embeds if layer_key[0] == "cross" else input[0])
                attn_probs = (module.get_attention_scores(query, key)).detach().cpu()
                self.attnstore.store(attn_probs, layer_key, self.latent_height, self.latent_width)
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
        empty_prompt = self.encode_prompt(prompt, self.device, batch_size, False)
        
        return empty_prompt[0]

    def preprocess_image(self, image, min_dim=None, factor=None):
        """
        Convert PIL image into a torch.Tensor with model-compatible dimensions.
        """
        min_dim = min_dim if min_dim is not None else self.default_output_resolution
        factor = factor if factor is not None else self.total_downsample_factor
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = resize_image(image, min_dim, factor)
        transform = Compose([ToTensor(), Normalize([0.5], [0.5])])

        return transform(image).unsqueeze(0).to(self.dtype)

    def image2latent(self, image, timesteps, num_images_per_prompt=1, seed=42):
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
            
        # generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = self.vae.encode(image).latent_dist.mean * self.vae.config.scaling_factor
        noise = torch.randn(latents.shape, device=self.device)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        return noisy_latents

    def extract_attention_maps(self, image, timesteps, batch_size=1, num_images_per_prompt=1, seed=42):
        batch_size = len(timesteps)
        timesteps.to(self.device)
        latents = self.image2latent(image, timesteps, seed)
        self.latent_height, self.latent_width = latents.shape[2:]

        if self.prompt_embeds.shape[0] != batch_size:
            self.prompt_embeds = torch.cat([self.prompt_embeds] * batch_size, dim=0)

        with torch.no_grad():
            unet_output = self.unet(latents, timesteps, encoder_hidden_states=self.prompt_embeds, return_dict=True)
            # noise_pred = unet_output["sample"]
            # latents = self.scheduler.step(noise_pred, timesteps, latents)["prev_sample"]
            # if self.device == "cuda":
            #     torch.cuda.empty_cache()
