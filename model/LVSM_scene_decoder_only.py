# Copyright (c) 2025 Haian Jin. Created for the LVSM project (ICLR 2025).


import os
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import traceback
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import camera_utils, data_utils 
from model.transformer import QK_Norm_SelfAttentionBlock, QK_Norm_CrossAttentionBlock, QK_Norm_SelfCrossAttentionBlock, QK_Norm_FFNBlock, init_weights
from model.loss import LossComputer
from model.encoder import preprocess_raw_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model.repa_pe import PEEncoder
import core.vision_encoder.pe as pe
from torchvision.transforms import Normalize
from model.repa_config import repa_map
import random

class Images2LatentScene(nn.Module):
    def __init__(self, config, logger=None):
        super().__init__()
        self.config = config
        self.process_data = data_utils.ProcessData(config)
        self.logger = logger

        # Initialize both input tokenizers, and output de-tokenizer
        self._init_tokenizers()
        
        # Initialize transformer blocks
        self._init_transformer()
        
        # Initialize loss computer
        self.loss_computer = LossComputer(config)

        # Initialize REPA
        if self.config.training.enable_repa:
            self._init_repa()
        
        self.num_registers = self.config.model.num_registers
        self.dim = self.config.model.transformer.d
        if self.num_registers == 0:
            self.register_tokens = None
        else:
            self.register_input = nn.Parameter(
                torch.randn(1, self.num_registers, self.dim)
            )
            self.register_input.requires_grad = True
            self.register_output = nn.Parameter(
                torch.randn(1, self.num_registers, self.dim)
            )
            self.register_output.requires_grad = True
            
    def _init_repa(self):
        if 'dino' in self.config.model.image_tokenizer.type:
            z_dim = 768
        elif 'pe' in self.config.model.image_tokenizer.type:
            z_dim = 768
        else:
            raise NotImplementedError(f"Unknown image tokenizer type: {self.config.model.image_tokenizer.type}")
        self.repa_label = {'input': {}, 'target': {}}
        self.repa_x = {'input': {}, 'target': {}}
        self.repa_projector = {'input': {}, 'target': {}}
        self.repa_config = repa_map[self.config.model.repa_config]
        self.repa_projector = nn.ModuleDict({
            'input': nn.ModuleDict(),
            'target': nn.ModuleDict()
        })
        for repa_type in ['input', 'target']:
            for key, value in self.repa_config[repa_type].items():
                projector_dict = nn.ModuleDict()
                self.repa_label[repa_type][key] = None
                for idx in value:
                    self.repa_x[repa_type][idx] = None
                    projector_dict[str(idx)] = self._create_repa_projector(self.config.model.transformer.d, self.config.model.projector_dim, z_dim, self.config.model.repa_projector_type)
                self.repa_projector[repa_type][str(key)] = projector_dict

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(
                in_channels * (patch_size**2),
                d_model,
                bias=False,
            ),
        )
        tokenizer.apply(init_weights)
        return tokenizer
    
    def _create_repa_projector(self, d_model, dim, z_dim, type):
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)
        use_flex_attention = config.attention_arch == 'flex'
        if type == 'linear2':    
            projector = nn.Sequential(
                nn.Linear(d_model, dim),
                nn.SiLU(),
                nn.Linear(dim, z_dim),
            )
        elif type == 'linear3':
            projector = nn.Sequential(
                nn.Linear(d_model, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, z_dim),
            )
        elif type == 'attention':
            projector = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                QK_Norm_SelfAttentionBlock(
                    d_model, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ),
                nn.Linear(d_model, z_dim),
            )
        else:
            raise ValueError(f"Unknown repa projector type: {type}")
        return projector

    def create_attention_mask(self, v_input, v_target, n_patches, device, batch_size, mask_strategy, view_min, view_max):
        """
        创建attention mask来屏蔽一些视角
        
        Args:
            v_input: 输入视角数量
            v_target: 目标视角数量  
            n_patches: 每个视角的patch数量
            device: 设备
            batch_size: batch大小
            mask_ratio: 屏蔽比例 (0.0-1.0)
            mask_strategy: 屏蔽策略 ('random', 'first', 'last', 'middle')
            
        Returns:
            attn_mask: [b, v_target, n_patches, v_input * n_patches] 的attention mask
        """
        
        # 创建mask矩阵 [b, n_patches, v_input * n_patches]
        attn_mask = torch.zeros(batch_size, v_target * n_patches, v_input * n_patches, device=device)
        
        # 随机屏蔽一些视角
        num_masked_views = view_max - random.randint(view_min, view_max)
        if mask_strategy == 'individual':
            if num_masked_views > 0:
                for b_idx in range(batch_size):
                    # 随机选择要屏蔽的视角
                    masked_view_indices = torch.randperm(v_input)[:num_masked_views]
                    for view_idx in masked_view_indices:
                        start_patch = view_idx * n_patches
                        end_patch = (view_idx + 1) * n_patches
                        attn_mask[b_idx, :, start_patch:end_patch] = float('-inf')
        elif mask_strategy == 'unified':
            if num_masked_views > 0:
                masked_view_indices = torch.randperm(v_input)[:num_masked_views]
                for view_idx in masked_view_indices:
                    start_patch = view_idx * n_patches
                    end_patch = (view_idx + 1) * n_patches
                    attn_mask[:, :, start_patch:end_patch] = float('-inf')
        return attn_mask

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.logger.info(f'create tokenizer with {self.config.model.image_tokenizer.type}')
        in_channels = self.config.model.image_tokenizer.in_channels
        if not self.config.model.concat_rgb:
            in_channels -= 3
        self.rgbp_tokenizer = self._create_tokenizer(
            in_channels = in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        encoder_dim = 0
        if self.config.model.image_tokenizer.type == 'pecore':
            model_type = 'PE-Core-L14-336'
            encoder = pe.VisionTransformer.from_config(model_type, pretrained=True)
            # encoder = encoder.to(self.device)
            self.image_encoder = encoder
            encoder_dim = 1024
        elif self.config.model.image_tokenizer.type == 'pes':
            model_type = 'PE-Spatial-B16-512'
            encoder = PEEncoder.from_config(model_type, pretrained=True)
            # encoder = encoder.to(self.device)
            self.image_encoder = encoder
            encoder_dim = 768
        elif self.config.model.image_tokenizer.type == 'dinov2':
            import timm
            encoder_type = self.config.model.image_tokenizer.type
            if self.config.model.image_tokenizer.source == 'local':
                encoder = torch.hub.load('/home/cowa/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14', source='local')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            del encoder.head
            patch_resolution = self.config.model.image_tokenizer.image_size // self.config.model.image_tokenizer.patch_size
            # TODO if needed for PE-Core/Spatial
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            # encoder = encoder.to(self.device)
            self.image_encoder = encoder
            encoder_dim = 768
        elif self.config.model.image_tokenizer.type == 'dinov3':
            import timm
            encoder = torch.hub.load(
                repo_or_dir='dinov3',
                model='dinov3_vitb16',
                source='local',
                pretrained=False
            )
            state_dict = torch.load(f"pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", map_location="cpu")
            encoder.load_state_dict(state_dict)
            # patch_resolution = self.config.model.image_tokenizer.image_size // self.config.model.image_tokenizer.patch_size
            # encoder.rope_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
            #     encoder.rope_embed.data, [patch_resolution, patch_resolution],
            # )
            encoder.head = torch.nn.Identity()
            # encoder = encoder.to(self.device)
            self.image_encoder = encoder
            encoder_dim = 768
        elif self.config.model.image_tokenizer.type == 'none':
            self.image_encoder = None
        else:
            raise NotImplementedError('unknown enocder type')

        if self.image_encoder is None:
            self.align_projector = None
        else:
            freeze_encoder = self.config.model.get("freeze_image_encoder", True)
            if freeze_encoder:
                self.logger.info('freeze parameters when loading image encoder')
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                self.image_encoder.eval()
            if not self.config.training.enable_repa:
                d_model = self.config.model.transformer.d
                projector = nn.Sequential(
                    nn.Linear(
                        d_model + encoder_dim,
                        d_model,
                            bias=False,
                        ),
                    )
                projector.apply(init_weights)
                self.align_projector = projector
        
        # Target pose tokenizer
        self.target_pose_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.target_pose_tokenizer.in_channels,
            patch_size = self.config.model.target_pose_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        
        # Image token decoder (decode image tokens into pixels)
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.config.model.transformer.d, elementwise_affine=False),
            nn.Linear(
                self.config.model.transformer.d,
                (self.config.model.target_pose_tokenizer.patch_size**2) * 3,
                bias=False,
            ),
            nn.Sigmoid()
        )
        self.image_token_decoder.apply(init_weights)


    def _init_transformer(self):
        """Initialize transformer blocks"""
        config = self.config.model.transformer
        use_qk_norm = config.get("use_qk_norm", False)
        use_flex_attention = config.attention_arch == 'flex'

        # Create transformer blocks
        if config.mode == 'self':
            self.logger.info(f'init transformer with self-attention only')
            self.self_attn_blocks = nn.ModuleList([
                QK_Norm_SelfAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(config.n_layer)
            ])
            self.cross_attn_blocks = None
            self.self_cross_blocks = None
        elif config.mode == 'cross':
            self.logger.info(f'init transformer with cross-attention only')
            self.cross_attn_blocks = nn.ModuleList([
                QK_Norm_CrossAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(config.n_layer)
            ])
            self.self_attn_blocks = None
            self.self_cross_blocks = None
        elif config.mode == 'alternate':
            self.logger.info(f'init transformer with alternating self-cross attention')
            self.self_cross_blocks = nn.ModuleList([
                QK_Norm_SelfCrossAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(config.n_layer // 2)
            ])
            self.self_attn_blocks = None
            self.cross_attn_blocks = None
        else:
            self.logger.info(f'init transformer with both self- and cross-attention')
            n_layer = config.n_layer // 2
            self.self_attn_blocks = nn.ModuleList([
                QK_Norm_SelfAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(n_layer)
            ])
            self.cross_attn_blocks = nn.ModuleList([
                QK_Norm_CrossAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(n_layer)
            ])
            self.self_cross_blocks = None

        if self.config.model.transformer.input_mode == 'embed':
            self.logger.info("use embed input self attention")
            self.input_self_attn_blocks = nn.ModuleList([
                QK_Norm_SelfAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(config.n_layer // 2)
            ])
        elif self.config.model.transformer.input_mode == 'encdec':
            self.logger.info("use encdec input self attention")
            self.input_self_attn_blocks = nn.ModuleList([
                QK_Norm_SelfAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(config.n_layer // 2)
            ])
        elif self.config.model.transformer.input_mode == 'ffn':
            self.logger.info(f'init transformer with ffn only')
            self.input_self_attn_blocks = nn.ModuleList([
                QK_Norm_FFNBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(config.n_layer // 2)
            ])
        # Apply special initialization if configured
        if config.get("special_init", False):
            # Initialize self-attention blocks
            if self.self_attn_blocks is not None:
                for idx, block in enumerate(self.self_attn_blocks):
                    if config.depth_init:
                        weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    else:
                        weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                    block.apply(lambda module: init_weights(module, weight_init_std))
            
            # Initialize cross-attention blocks
            if self.cross_attn_blocks is not None:
                for idx, block in enumerate(self.cross_attn_blocks):
                    if config.depth_init:
                        weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    else:
                        weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                    block.apply(lambda module: init_weights(module, weight_init_std))
            
            # Initialize self-cross blocks
            if self.self_cross_blocks is not None:
                for idx, block in enumerate(self.self_cross_blocks):
                    if config.depth_init:
                        weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                    else:
                        weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                    block.apply(lambda module: init_weights(module, weight_init_std))
        else:
            # Standard initialization
            if self.self_attn_blocks is not None:
                for block in self.self_attn_blocks:
                    block.apply(init_weights)
            if self.cross_attn_blocks is not None:
                for block in self.cross_attn_blocks:
                    block.apply(init_weights)
            if self.self_cross_blocks is not None:
                for block in self.self_cross_blocks:
                    block.apply(init_weights)
                
        self.transformer_input_layernorm = nn.LayerNorm(config.d, elementwise_affine=False)

        if self.config.model.extra_enc == 'attn':
            self.extra_enc = nn.ModuleList([
                QK_Norm_SelfAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm, use_flex_attention=use_flex_attention
                ) for _ in range(self.config.model.enc_layer)
            ])
        else:
            self.extra_enc = None
        
        if self.extra_enc is not None:
            for idx, block in enumerate(self.extra_enc):
                if config.depth_init:
                    weight_init_std = 0.02 / (2 * (idx + 1)) ** 0.5
                else:
                    weight_init_std = 0.02 / (2 * config.n_layer) ** 0.5
                block.apply(lambda module: init_weights(module, weight_init_std))



    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()

    def forward_features(self, x, output_layer=None, repa_type=None, masks=None):
        x = self.image_encoder.prepare_tokens_with_masks(x, masks)
        if output_layer is None:
            output_layer = len(self.image_encoder.blocks)

        for idx, blk in enumerate(self.image_encoder.blocks):
            if idx >= output_layer:
                break
            x = blk(x)
            if repa_type is not None:
                if idx + 1 in self.repa_label[repa_type].keys():
                    self.repa_label[repa_type][idx + 1] = x[:, 1:, :]

        x_norm = self.image_encoder.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.image_encoder.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.image_encoder.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward_features_v3(self, x_list, masks_list, output_layer=None, repa_type=None):
        x = []
        rope = []
        if output_layer is None:
            output_layer = len(self.image_encoder.blocks)
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.image_encoder.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for idx, blk in enumerate(self.image_encoder.blocks):
            if self.image_encoder.rope_embed is not None:
                rope_sincos = [self.image_encoder.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            if idx >= output_layer:
                break
            x = blk(x, rope_sincos)
            if repa_type is not None:
                if idx + 1 in self.repa_label[repa_type].keys():
                    self.repa_label[repa_type][idx + 1] = x[0][:, 5:, :]
            # x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.image_encoder.untie_cls_and_patch_norms or self.image_encoder.untie_global_and_local_cls_norm:
                if self.image_encoder.untie_global_and_local_cls_norm and self.image_encoder.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.image_encoder.local_cls_norm(x[:, : self.image_encoder.n_storage_tokens + 1])
                elif self.image_encoder.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.image_encoder.cls_norm(x[:, : self.image_encoder.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.image_encoder.norm(x[:, : self.image_encoder.n_storage_tokens + 1])
                x_norm_patch = self.image_encoder.norm(x[:, self.image_encoder.n_storage_tokens + 1 :])
            else:
                x_norm = self.image_encoder.norm(x)
                x_norm_cls_reg = x_norm[:, : self.image_encoder.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.image_encoder.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output
    
    def get_image_feature(self, image): # TODO distilled 
        with torch.no_grad():
            enc_type = self.config.model.image_tokenizer.type
            inter_mode = 'bicubic'
            x = rearrange(image, "b v c h w -> (b v) c h w")
            
            use_patch_interpolation = self.config.model.image_tokenizer.get("use_patch_interpolation", False)
            if 'dino' in enc_type:
                if use_patch_interpolation:
                    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
                    x = torch.nn.functional.interpolate(x, 336, mode=inter_mode)
                else:
                    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
                    x = torch.nn.functional.interpolate(x, 448, mode=inter_mode)
            elif 'pe' in enc_type:
                resolution = 336 if 'core' in enc_type else 448
                if use_patch_interpolation:
                    x = torch.nn.functional.interpolate(x, resolution, mode=inter_mode, align_corners=False)
                    x = (x - 0.5) / 0.5
                else:
                    x = torch.nn.functional.interpolate(x, 448, mode=inter_mode, align_corners=False)
                    x = (x - 0.5) / 0.5
            
            if self.config.model.image_tokenizer.output_layer is None:
                x = self.image_encoder.forward_features(x)
            elif 'pe' in enc_type:
                x = self.image_encoder.forward_features(x, layer_idx=self.config.model.image_tokenizer.output_layer)
            elif 'dinov2' in enc_type:
                x = self.forward_features(x, self.config.model.image_tokenizer.output_layer)
            elif 'dinov3' in enc_type:
                x = self.forward_features_v3([x], [None], self.config.model.image_tokenizer.output_layer)

            if 'dinov2' in enc_type: 
                x = x['x_norm_patchtokens']
            elif 'dinov3' in enc_type: 
                x = x[0]['x_norm_patchtokens']
            if 'pe' in enc_type: 
                x = x[:, 1:, :]
            
            if use_patch_interpolation:
                if 'pe' in enc_type:
                    pass
                current_grid_size = int(x.shape[1] ** 0.5)
                target_grid_size = 32
                x = rearrange(x, "b (h w) d -> b d h w", h=current_grid_size, w=current_grid_size)
                
                x = torch.nn.functional.interpolate(
                    x, 
                    size=(target_grid_size, target_grid_size), 
                    mode='bilinear', 
                    align_corners=False
                )
                x = rearrange(x, "b d h w -> b (h w) d")
            return x

    def get_repa_feature(self, image, repa_type): # TODO distilled 
        with torch.no_grad():
            enc_type = self.config.model.image_tokenizer.type
            inter_mode = 'bicubic'
            x = rearrange(image, "b v c h w -> (b v) c h w")
            
            use_patch_interpolation = self.config.model.image_tokenizer.get("use_patch_interpolation", False)
            if 'dino' in enc_type:
                if use_patch_interpolation:
                    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
                    x = torch.nn.functional.interpolate(x, 336, mode=inter_mode)
                else:
                    x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
                    x = torch.nn.functional.interpolate(x, 448, mode=inter_mode)
            elif 'dinov3' in enc_type:
                x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
                x = torch.nn.functional.interpolate(x, 512, mode='bicubic')
            elif 'pe' in enc_type:
                resolution = 336 if 'core' in enc_type else 512
                if use_patch_interpolation:
                    x = torch.nn.functional.interpolate(x, resolution, mode=inter_mode, align_corners=False)
                    x = (x - 0.5) / 0.5
                else:
                    x = torch.nn.functional.interpolate(x, 448, mode=inter_mode, align_corners=False)
                    x = (x - 0.5) / 0.5
            else:
                raise ValueError(f"Invalid image tokenizer type: {enc_type}")
            
            if 'dinov2' in enc_type: 
                x = self.forward_features(x, None, repa_type=repa_type)
                x = x['x_norm_patchtokens']
            elif 'dinov3' in enc_type: 
                x = self.forward_features_v3([x], [None], None, repa_type=repa_type)[0]
                x = x['x_norm_patchtokens']
            elif 'pe' in enc_type:
                x = self.image_encoder.forward_features(x, repa_type=repa_type, repa_label=self.repa_label[repa_type])
                x = x[:, 1:, :]
            
            if use_patch_interpolation:
                if 'pe' in enc_type:
                    pass
                current_grid_size = int(x.shape[1] ** 0.5)
                target_grid_size = 32
                for idx in self.repa_label[repa_type].keys():
                    x = self.repa_label[repa_type][idx]
                    x = rearrange(x, "b (h w) d -> b d h w", h=current_grid_size, w=current_grid_size)
                    x = torch.nn.functional.interpolate(
                        x, 
                        size=(target_grid_size, target_grid_size), 
                        mode='bicubic', 
                        align_corners=False
                    )
                    self.repa_label[repa_type][idx] = rearrange(x, "b d h w -> b (h w) d")

    
    def pass_layers(self, input_tokens, target_tokens, registers, token_shape, gradient_checkpoint=False, checkpoint_every=1, attn_mask=None, extract_features=False):
        """
        concat_tokens: [B, n_input + n_target, D]
        input_patch: int, input tokens数量
        attn_mask: Optional attention mask for cross attention
        extract_features: 是否提取每一层的特征用于可视化
        """
        # 获取token形状信息
        v_input, v_target, n_patches = token_shape
        bv, _, d = input_tokens.shape
        b = bv // v_input
        if self.num_registers > 0:
            input_tokens = torch.cat([registers, input_tokens], dim=1)
            register_output = self.register_output.expand(b * v_target, -1, -1)
            target_tokens = torch.cat([register_output, target_tokens], dim=1)
        
        # 用于存储每一层的特征
        layer_features = {}
        
        if self.config.model.transformer.input_mode == 'encdec':
            for idx, block in enumerate(self.input_self_attn_blocks):
                if self.config.model.transformer.input_scope == 'local':
                    input_tokens = input_tokens.view(b * v_input, n_patches, d)
                    input_tokens = block(input_tokens)
                    if extract_features:
                        layer_features[f'input_self_attn_{idx}'] = input_tokens.clone().detach()
                    input_tokens = input_tokens.view(b, v_input * n_patches, d)
                elif self.config.model.transformer.input_scope == 'global':
                    input_tokens = block(input_tokens)
                    if extract_features:
                        layer_features[f'input_self_attn_{idx}'] = input_tokens.clone().detach()
                else:
                    raise ValueError(f"Invalid input scope: {self.config.model.transformer.input_scope}")
        
        # 24-layer Cross-Attention
        if self.cross_attn_blocks is not None:
            target_tokens = target_tokens.view(b, v_target * (n_patches + self.num_registers), d)
            for idx in range(len(self.cross_attn_blocks) // 2):
                if self.config.model.transformer.input_mode == 'embed' or self.config.model.transformer.input_mode == 'ffn':
                    if self.config.model.transformer.input_scope == 'local':
                        input_tokens = input_tokens.view(b * v_input, n_patches + self.num_registers, d)
                        input_tokens = self.input_self_attn_blocks[idx](input_tokens)
                        if extract_features:
                            layer_features[f'input_self_attn_{idx}'] = input_tokens.clone().detach()
                        if self.config.training.enable_repa:
                            if idx + 1 in self.repa_x['input'].keys():
                                self.repa_x['input'][idx + 1] = input_tokens[:, self.num_registers:, :]
                        input_tokens = input_tokens.view(b, v_input * (n_patches + self.num_registers), d)
                
                target_tokens = self.cross_attn_blocks[2*idx](input_tokens, target_tokens, attn_bias=attn_mask)
                if extract_features:
                    layer_features[f'cross_attn_{2*idx}'] = target_tokens.clone().detach()
                
                target_tokens = self.cross_attn_blocks[2*idx+1](input_tokens, target_tokens, attn_bias=attn_mask)
                if extract_features:
                    layer_features[f'cross_attn_{2*idx+1}'] = target_tokens.clone().detach()
            
            target_tokens = target_tokens.view(b * v_target, n_patches + self.num_registers, d)
        
        # Self-Cross
        if self.self_cross_blocks is not None:
            for idx, block in enumerate(self.self_cross_blocks):
                if self.config.model.transformer.input_mode == 'embed' or self.config.model.transformer.input_mode == 'ffn':
                    if self.config.model.transformer.input_scope == 'local':
                        input_tokens = input_tokens.view(b * v_input, n_patches + self.num_registers, d)
                        input_tokens = self.input_self_attn_blocks[idx](input_tokens)
                        if extract_features:
                            layer_features[f'input_self_attn_{idx}'] = input_tokens.clone().detach()
                        if self.config.training.enable_repa:
                            if idx + 1 in self.repa_x['input'].keys():
                                self.repa_x['input'][idx + 1] = input_tokens[:, self.num_registers:, :]
                        input_tokens = input_tokens.view(b, v_input * (n_patches + self.num_registers), d)
                    elif self.config.model.transformer.input_scope == 'global':
                        input_tokens = self.input_self_attn_blocks[idx](input_tokens)
                        if extract_features:
                            layer_features[f'input_self_attn_{idx}'] = input_tokens.clone().detach()
                    else:
                        raise ValueError(f"Invalid input scope: {self.config.model.transformer.input_scope}")
                
                target_tokens = block(input_tokens, target_tokens, attn_bias=attn_mask)
                if extract_features:
                    layer_features[f'self_cross_{idx}'] = target_tokens.clone().detach()
                    
                if self.config.training.enable_repa:
                    if idx + 1 in self.repa_x['target'].keys():
                        self.repa_x['target'][idx + 1] = target_tokens[:, self.num_registers:, :]
        
        if extract_features:
            return target_tokens, layer_features
        return target_tokens[:, self.num_registers:, :]
            
    # @torch._dynamo.assume_constant_result
    def get_posed_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        '''
        Args:
            images: [b, v, c, h, w]
            ray_o: [b, v, 3, h, w]
            ray_d: [b, v, 3, h, w]
            method: Method for creating pose conditioning
        Returns:
            posed_images: [b, v, c+6, h, w] or [b, v, 6, h, w] if images is None
        '''

        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)
            
        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)
            
        else:  # default_plucker
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
    
    
    def forward(self, data_batch, input, target, has_target_image=True, detach=False, train=True, extract_features=False):

        # Process input images
        if self.config.model.concat_rgb:
            posed_input_images = self.get_posed_input(
                images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
            )
        else:
            posed_input_images = self.get_posed_input(
                images=None, ray_o=input.ray_o, ray_d=input.ray_d
            )
        b, v_input, c, h, w = posed_input_images.size()
        # [I; P]
        rgbp_token = self.rgbp_tokenizer(posed_input_images)  # [b*v, n_patches, d]
        # x = Linear([I; P]) (b, np, d)
        bv, n_patches, d = rgbp_token.size()  # [b*v, n_patches, d]
        # rgbp_token = rgbp_token.view(b, v_input * n_patches, d)  # [b, v*n_patches, d]
        layer_features = {}
        if self.num_registers == 0:
            registers = None
        else:
            registers = self.register_input.expand(bv, -1, -1)
            rgbp_token = torch.cat([registers, rgbp_token], dim=1)
        if self.extra_enc is not None:
            for idx, block in enumerate(self.extra_enc):
                rgbp_token = block(rgbp_token)
                if self.config.training.enable_repa:
                    if idx + 101 in self.repa_x['input'].keys():
                        self.repa_x['input'][idx + 101] = rgbp_token[:, self.num_registers:, :]
                if extract_features:
                    layer_features[f'extra_enc_{idx}'] = rgbp_token.clone().detach()
        if self.num_registers > 0:
            registers = rgbp_token[:, :self.num_registers, :]
        rgbp_token = rgbp_token[:, self.num_registers:, :]
        if self.image_encoder is not None and not self.config.training.enable_repa:
            input_img_features = self.get_image_feature(input.image) # Linear(encoder(I)) (b, np, d)
            input_img_features = input_img_features.reshape(b * v_input, n_patches, -1)  # [b*v, n_patches, d]
            input_img_tokens = torch.cat((input_img_features, rgbp_token), dim=2)  # [b*v, n_patches, d*2]
            input_img_tokens = self.align_projector(input_img_tokens) # [b*v, n_patches, d]
            # Linear([F;Linear([P;I])]) (b, np, d1+d2) # TODO 图片内self
        elif self.image_encoder is not None and self.config.training.enable_repa:
            input_img_tokens = rgbp_token
            self.get_repa_feature(input.image, 'input')
            self.get_repa_feature(target.image, 'target')
        else:
            input_img_tokens = rgbp_token
        
        # lvsm:256*256/(16*16)=256  dino:224*224/(14*14)=256 pe:448*448/(14*14)
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)

        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v, n_patches, d]
        # P = Linear([P]) (b*out, np, d)

        checkpoint_every = self.config.training.grad_checkpoint_every
        token_shape = (v_input, v_target, n_patches)
        
        # 生成attention mask（如果启用）
        attn_mask = None
        if self.config.training.use_view_masking and train:
            attn_mask = self.create_attention_mask(
                v_input, v_target, n_patches,
                input_img_tokens.device,
                batch_size=b,
                mask_strategy=self.config.training.view_mask_strategy,
                view_min=self.config.training.view_min,
                view_max=self.config.training.view_max
            )
        
        if extract_features:
            target_image_tokens, layer_features_attn = self.pass_layers(
                input_img_tokens, target_pose_tokens, registers, token_shape, 
                gradient_checkpoint=False, checkpoint_every=checkpoint_every, 
                attn_mask=attn_mask, extract_features=True
            )
            layer_features.update(layer_features_attn)
        else:
            target_image_tokens = self.pass_layers(
                input_img_tokens, target_pose_tokens, registers, token_shape, 
                gradient_checkpoint=False, checkpoint_every=checkpoint_every, 
                attn_mask=attn_mask, extract_features=False
            )
            layer_features = None

        # [b * v_target, n_patches, d]

        # [b*v_target, n_patches, p*p*3]
        rendered_images = self.image_token_decoder(target_image_tokens)
        
        height, width = target.image_h_w

        patch_size = self.config.model.target_pose_tokenizer.patch_size
        rendered_images = rearrange(
            rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v_target,
            h=height // patch_size, 
            w=width // patch_size, 
            p1=patch_size, 
            p2=patch_size, 
            c=3
        )
        if has_target_image:
            if self.config.training.enable_repa:
                loss_metrics = self.loss_computer(
                    rendered_images,
                    target.image,
                    self.repa_x,
                    self.repa_label,
                    self.repa_projector,
                    self.repa_config,
                    self.config.training.enable_repa,
                    train=(train and self.config.training.enable_repa)
                )
            else:
                loss_metrics = self.loss_computer(
                    rendered_images,
                    target.image,
                    train=train
                )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images,
            layer_features=layer_features
            )
        
        return result



    @torch.no_grad()
    def render_video(self, data_batch, traj_type="interpolate", num_frames=60, loop_video=False, order_poses=False):
        """
        Render a video from the model.
        
        Args:
            result: Edict from forward pass or just data
            traj_type: Type of trajectory
            num_frames: Number of frames to render
            loop_video: Whether to loop the video
            order_poses: Whether to order poses
            
        Returns:
            result: Updated with video rendering
        """
    
        if data_batch.input is None:
            input, target = self.process_data(data_batch, has_target_image=False, target_has_input=self.config.training.target_has_input, compute_rays=True)
            data_batch = edict(input=input, target=target)
        else:
            input, target = data_batch.input, data_batch.target
        
        # Prepare input tokens; [b, v, 3+6, h, w]
        posed_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        bs, v_input, c, h, w = posed_images.size()

        input_img_tokens = self.image_tokenizer(posed_images)  # [b*v_input, n_patches, d]

        _, n_patches, d = input_img_tokens.size()  # [b*v_input, n_patches, d]
        input_img_tokens = input_img_tokens.reshape(bs, v_input * n_patches, d)  # [b, v_input*n_patches, d]

        # target_pose_cond_list = []
        if traj_type == "interpolate":
            c2ws = input.c2w # [b, v, 4, 4]
            fxfycxcy = input.fxfycxcy #  [b, v, 4]
            device = input.c2w.device

            # Create intrinsics from fxfycxcy
            intrinsics = torch.zeros((c2ws.shape[0], c2ws.shape[1], 3, 3), device=device) # [b, v, 3, 3]
            intrinsics[:, :,  0, 0] = fxfycxcy[:, :, 0]
            intrinsics[:, :,  1, 1] = fxfycxcy[:, :, 1]
            intrinsics[:, :,  0, 2] = fxfycxcy[:, :, 2]
            intrinsics[:, :,  1, 2] = fxfycxcy[:, :, 3]

            # Loop video if requested
            if loop_video:
                c2ws = torch.cat([c2ws, c2ws[:, [0], :]], dim=1)
                intrinsics = torch.cat([intrinsics, intrinsics[:, [0], :]], dim=1)

            # Interpolate camera poses
            all_c2ws, all_intrinsics = [], []
            for b in range(input.image.size(0)):
                cur_c2ws, cur_intrinsics = camera_utils.get_interpolated_poses_many(
                    c2ws[b, :, :3, :4], intrinsics[b], num_frames, order_poses=order_poses
                )
                all_c2ws.append(cur_c2ws.to(device))
                all_intrinsics.append(cur_intrinsics.to(device))

            all_c2ws = torch.stack(all_c2ws, dim=0) # [b, num_frames, 3, 4]
            all_intrinsics = torch.stack(all_intrinsics, dim=0) # [b, num_frames, 3, 3]

            # Add homogeneous row to c2ws
            homogeneous_row = torch.tensor([[[0, 0, 0, 1]]], device=device).expand(all_c2ws.shape[0], all_c2ws.shape[1], -1, -1)
            all_c2ws = torch.cat([all_c2ws, homogeneous_row], dim=2)

            # Convert intrinsics to fxfycxcy format
            all_fxfycxcy = torch.zeros((all_intrinsics.shape[0], all_intrinsics.shape[1], 4), device=device)
            all_fxfycxcy[:, :, 0] = all_intrinsics[:, :, 0, 0]  # fx
            all_fxfycxcy[:, :, 1] = all_intrinsics[:, :, 1, 1]  # fy
            all_fxfycxcy[:, :, 2] = all_intrinsics[:, :, 0, 2]  # cx
            all_fxfycxcy[:, :, 3] = all_intrinsics[:, :, 1, 2]  # cy

        # Compute rays for rendering
        rendering_ray_o, rendering_ray_d = self.process_data.compute_rays(
            fxfycxcy=all_fxfycxcy, c2w=all_c2ws, h=h, w=w, device=device
        )

        # Get pose conditioning for target views
        target_pose_cond = self.get_posed_input(
            ray_o=rendering_ray_o.to(input.image.device), 
            ray_d=rendering_ray_d.to(input.image.device)
        )
                
        _, num_views, c, h, w = target_pose_cond.size()
    
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [bs*v_target, n_patches, d]
        _, n_patches, d = target_pose_tokens.size()  # [b*v_target, n_patches, d]
        target_pose_tokens = target_pose_tokens.reshape(bs, num_views * n_patches, d)  # [b, v_target*n_patches, d]

        view_chunk_size = 4

        video_rendering_list = []
        for cur_chunk in range(0, num_views, view_chunk_size):
            cur_view_chunk_size = min(view_chunk_size, num_views - cur_chunk)

            # [b, (v_input*n_patches), d] -> [(b * cur_v_target), (v_input*n_patches), d]
            repeated_input_img_tokens = repeat(input_img_tokens.detach(), 'b np d -> (b chunk) np d', chunk=cur_view_chunk_size, np=n_patches* v_input)

            start_idx, end_idx = cur_chunk * n_patches, (cur_chunk + cur_view_chunk_size) * n_patches            
            # [b, v_target * n_patches, d] -> [b, cur_v_target*n_patches, d] -> [b*cur_v_target, n_patches, d]
            cur_target_pose_tokens = rearrange(target_pose_tokens[:, start_idx:end_idx,: ], 
                                               "b (v_chunk p) d -> (b v_chunk) p d", 
                                               v_chunk=cur_view_chunk_size, p=n_patches)

            cur_concat_input_tokens = torch.cat((repeated_input_img_tokens, cur_target_pose_tokens,), dim=1) # [b*cur_v_target, v_input*n_patches+n_patches, d]
            cur_concat_input_tokens = self.transformer_input_layernorm(
                cur_concat_input_tokens
            )

            transformer_output_tokens = self.pass_layers(cur_concat_input_tokens, gradient_checkpoint=False)

            _, pred_target_image_tokens = transformer_output_tokens.split(
                [v_input * n_patches, n_patches], dim=1
            ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]

            height, width = target.image_h_w

            patch_size = self.config.model.target_pose_tokenizer.patch_size

            # [b, v_target*n_patches, p*p*3]
            video_rendering = self.image_token_decoder(pred_target_image_tokens)
            
            video_rendering = rearrange(
                video_rendering, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
                v=cur_view_chunk_size,
                h=height // patch_size, 
                w=width // patch_size, 
                p1=patch_size, 
                p2=patch_size, 
                c=3
            ).cpu()

            video_rendering_list.append(video_rendering)
        video_rendering = torch.cat(video_rendering_list, dim=1)
        data_batch.video_rendering = video_rendering


        return data_batch

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if (file_name.endswith(".pt") and not file_name.startswith('ckpt_t'))]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
            print(f'load checkpoint from {ckpt_paths[-1]}')
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        state_dict = checkpoint["model"]
        if not self.config.training.use_compile:
            self.logger.info("discard _orig_mod. in loading model")
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        return 0


