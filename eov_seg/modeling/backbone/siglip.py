"""
SigLIP-2 Backbone for EOV-Seg
Adapted from CLIP backbone implementation
"""
import torch
import torch.nn.functional as F
import math
from detectron2.utils import comm
from detectron2.modeling import Backbone, ShapeSpec
from .build import BACKBONE_REGISTRY

try:
    from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@BACKBONE_REGISTRY.register()
class SigLIP(Backbone):
    """
    SigLIP-2 Vision-Language Backbone for Open-Vocabulary Segmentation

    Supports SigLIP-2 ViT models from HuggingFace transformers:
    - google/siglip-base-patch16-256
    - google/siglip-base-patch16-384
    - google/siglip-base-patch16-512
    """

    def __init__(self, cfg, input_shape=None, backbone_type="vit"):
        super().__init__()

        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers library is required for SigLIP backbone. "
                "Install with: pip install transformers"
            )

        self.backbone_type = backbone_type

        # Get model name from config
        model_name = cfg.MODEL.OV_HEAD.SIGLIP_MODEL_NAME

        # Download on local rank 0 first to avoid race conditions
        if comm.get_local_rank() == 0:
            AutoModel.from_pretrained(model_name)
            AutoTokenizer.from_pretrained(model_name)
        comm.synchronize()

        self.model_name = model_name

        # Load SigLIP model and tokenizer
        self.siglip_model = AutoModel.from_pretrained(model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # Parse model configuration
        model_name_lower = model_name.lower()

        # SigLIP uses ViT architecture
        self.model_type = 'vit'

        # Get vision model config
        vision_config = self.siglip_model.vision_model.config
        self.output_channels = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        self.output_strides = self.patch_size

        # Set up output feature specifications for detectron2
        self._out_feature_strides = {
            "layer": self.output_strides,
        }
        self._out_feature_channels = {
            "layer": self.output_channels,
        }

        # Freeze model and set to eval mode
        self.eval()
        self.freeze_everything()

    def freeze_everything(self):
        """Freeze all parameters in the SigLIP model"""
        for param in self.siglip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = False):
        """
        Encode text using SigLIP text encoder

        Args:
            text: Tokenized text input
            normalize: Whether to L2-normalize the output features

        Returns:
            Text embeddings of shape [batch_size, hidden_size]
        """
        outputs = self.siglip_model.text_model(
            input_ids=text['input_ids'],
            attention_mask=text.get('attention_mask', None)
        )

        # Get pooled output (CLS token equivalent)
        text_embeds = outputs.pooler_output

        if normalize:
            text_embeds = F.normalize(text_embeds, dim=-1)

        return text_embeds

    def tokenize_text(self, text):
        """
        Tokenize text using SigLIP tokenizer

        Args:
            text: List of strings or single string

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.text_tokenizer(
            text,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )

    def extract_features(self, x):
        """
        Extract visual features from input images

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Feature tensor of shape [B, hidden_size, H', W']
        """
        return self.extract_features_vit(x)

    def extract_features_vit(self, x):
        """
        Extract features using SigLIP ViT encoder

        Args:
            x: Input images of shape [B, C, H, W]

        Returns:
            Spatial features of shape [B, C, H', W'] where H' = H/patch_size, W' = W/patch_size
        """
        # Get input dimensions
        B, C, H, W = x.shape

        # Patch embedding
        vision_model = self.siglip_model.vision_model
        embeddings = vision_model.embeddings(x)

        # embeddings shape: [B, num_patches, hidden_size]
        # num_patches = (H / patch_size) * (W / patch_size)

        # Pass through encoder
        encoder_outputs = vision_model.encoder(
            embeddings,
            output_hidden_states=True,
            return_dict=True
        )

        # Get the last hidden state
        last_hidden_state = encoder_outputs.last_hidden_state  # [B, num_patches, hidden_size]

        # Apply layer norm
        last_hidden_state = vision_model.post_layernorm(last_hidden_state)

        # Reshape back to spatial format
        # Calculate spatial dimensions
        H_out = H // self.patch_size
        W_out = W // self.patch_size

        # Reshape: [B, num_patches, C] -> [B, H', W', C] -> [B, C, H', W']
        features = last_hidden_state.view(B, H_out, W_out, self.output_channels)
        features = features.permute(0, 3, 1, 2).contiguous()

        return features

    def get_text_classifier(self, text_list, device):
        """
        Generate text classifier weights for a list of class names

        Args:
            text_list: List of text descriptions (class names with templates)
            device: Target device for tensors

        Returns:
            Text features of shape [num_classes, hidden_size]
        """
        self.eval()
        with torch.no_grad():
            # Tokenize text
            text_tokens = self.tokenize_text(text_list)
            text_tokens = {k: v.to(device) for k, v in text_tokens.items()}

            # Encode text (un-normalized for compatibility with CLIP pipeline)
            text_features = self.encode_text(text_tokens, normalize=False)

            return text_features

    def forward(self, x):
        """
        Forward pass through SigLIP backbone

        Args:
            x: Input images

        Returns:
            Dictionary with extracted features
        """
        self.eval()
        with torch.no_grad():
            return self.extract_features(x)

    @property
    def dim_latent(self):
        """Get the dimension of the latent embedding space"""
        # Check if projection_dim exists, otherwise use hidden_size
        if hasattr(self.siglip_model.config, 'projection_dim'):
            return self.siglip_model.config.projection_dim
        elif hasattr(self.siglip_model.vision_model.config, 'projection_dim'):
            return self.siglip_model.vision_model.config.projection_dim
        else:
            # Fallback to hidden size
            return self.siglip_model.vision_model.config.hidden_size

    def output_shape(self):
        """
        Return the output shape specification for detectron2

        Returns:
            Dictionary mapping feature names to ShapeSpec
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in ["layer"]
        }

    @property
    def size_divisibility(self):
        """
        Required divisibility of input size

        Returns -1 to indicate flexible input sizes
        """
        return -1
