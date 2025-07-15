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
from model.transformer import QK_Norm_SelfAttentionBlock, QK_Norm_CrossAttentionBlock, QK_Norm_SelfCrossAttentionBlock, init_weights
from model.loss import LossComputer
from model.encoder import preprocess_raw_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import core.vision_encoder.pe as pe
from torchvision.transforms import Normalize

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

        # REPA
        if config.training.enable_repa:
            if 'dinov2' in self.config.model.encoder_type:
                z_dim = 768
            elif 'PE' in self.config.model.encoder_type:
                z_dim = 1536 if "Spatial" in self.config.model.encoder_type else 1024
            self.projectors = nn.Sequential(
                nn.Linear(self.config.model.transformer.d, config.model.projector_dim),
                nn.SiLU(),
                nn.Linear(config.model.projector_dim, config.model.projector_dim),
                nn.SiLU(),
                nn.Linear(config.model.projector_dim, z_dim)
            )
        else:
            self.projectors = None

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

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.logger.info(f'create tokenizer with {self.config.model.image_tokenizer.type}')
        self.rgbp_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.image_tokenizer.in_channels,
            patch_size = self.config.model.image_tokenizer.patch_size,
            d_model = self.config.model.transformer.d
        )
        if self.config.model.image_tokenizer.type == 'PE':
            model_type = 'PE-Core-L14-336'
            encoder = pe.VisionTransformer.from_config(model_type, pretrained=True)
            # encoder = encoder.to(self.device)
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
            self.image_encoder = encoder
            self.feature_projector = nn.Linear(1024, self.config.model.transformer.d)
        elif self.config.model.image_tokenizer.type == 'dino':
            import timm
            encoder_type = self.config.model.image_tokenizer.type
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            del encoder.head
            patch_resolution = 16
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            # encoder = encoder.to(self.device)
            self.image_encoder = encoder
            freeze_encoder = self.config.model.image_tokenizer.get("freeze_image_encoder", True)
            if freeze_encoder:
                for param in self.image_encoder.parameters():
                    param.requires_grad = False
                self.image_encoder.eval()
            self.feature_projector = nn.Linear(768, self.config.model.transformer.d)
        else:
            self.image_encoder = None

        if self.image_encoder is None:
            self.align_projector = None
        else:
            d_model = self.config.model.transformer.d
            tokenizer = nn.Sequential(
                nn.Linear(
                    d_model * 2,
                    d_model,
                        bias=False,
                    ),
                )
            tokenizer.apply(init_weights)
            self.align_projector = tokenizer
        
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

        # Create transformer blocks
        if config.mode == 'self':
            self.logger.info(f'init transformer with self-attention only')
            self.self_attn_blocks = nn.ModuleList([
                QK_Norm_SelfAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(config.n_layer)
            ])
            self.cross_attn_blocks = None
            self.self_cross_blocks = None
        elif config.mode == 'cross':
            self.logger.info(f'init transformer with cross-attention only')
            self.cross_attn_blocks = nn.ModuleList([
                QK_Norm_CrossAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(config.n_layer)
            ])
            self.self_attn_blocks = None
            self.self_cross_blocks = None
        elif config.mode == 'alternate':
            self.logger.info(f'init transformer with alternating self-cross attention')
            self.self_cross_blocks = nn.ModuleList([
                QK_Norm_SelfCrossAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(config.n_layer // 2)
            ])
            self.self_attn_blocks = None
            self.cross_attn_blocks = None
        else:
            self.logger.info(f'init transformer with both self- and cross-attention')
            n_layer = config.n_layer // 2
            self.self_attn_blocks = nn.ModuleList([
                QK_Norm_SelfAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(n_layer)
            ])
            self.cross_attn_blocks = nn.ModuleList([
                QK_Norm_CrossAttentionBlock(
                    config.d, config.d_head, use_qk_norm=use_qk_norm
                ) for _ in range(n_layer)
            ])
            self.self_cross_blocks = None
        
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


    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        self.loss_computer.eval()

    def get_image_feature(self, image):
        enc_type = self.config.model.image_tokenizer.type
        inter_mode = 'bicubic'
        x = rearrange(image, "b v c h w -> (b v) c h w")
        if 'dino' in enc_type:
            x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
            x = torch.nn.functional.interpolate(x, 448, mode=inter_mode)
        elif 'PE' in enc_type:
            x = torch.nn.functional.interpolate(x, 448, mode=inter_mode, align_corners=False)
            x = (x - 0.5) / 0.5
        x = self.image_encoder.forward_features(x)
        if 'dino' in enc_type: x = x['x_norm_patchtokens']
        if 'PE' in enc_type: x = x[:, 1:, :]
        x = self.feature_projector(x)
        return x

    
    # @torch._dynamo.disable
    def pass_layers(self, tokens, target_patch, gradient_checkpoint=False, checkpoint_every=1):
        """
        concat_tokens: [B, n_input + n_target, D]
        target_patch: int, input tokens数量
        """
        zs_tilde = None
        if gradient_checkpoint:
            raise NotImplementedError("gradient checkpoint not supported")
        if self.config.training.enable_repa:
            raise NotImplementedError("repa not supported")
        
        # 1. Self-Attention阶段
        use_reentrant = self.config.training.use_compile
        if self.self_attn_blocks is not None:
            for idx, block in enumerate(self.self_attn_blocks):
                tokens = torch.utils.checkpoint.checkpoint(block, tokens, use_reentrant=use_reentrant)

        input_tokens, target_tokens = tokens[:, :target_patch, :], tokens[:, target_patch:, :]

        # 2. Cross-Attention阶段
        if self.cross_attn_blocks is not None:
            for idx, block in enumerate(self.cross_attn_blocks):
                target_tokens = torch.utils.checkpoint.checkpoint(
                    block, target_tokens, input_tokens, use_reentrant=use_reentrant
                )
        
        # 3. 交替的 Self-Cross 阶段
        if self.self_cross_blocks is not None:
            for idx, block in enumerate(self.self_cross_blocks):
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, target_patch, use_reentrant=use_reentrant
                )
            input_tokens, target_tokens = tokens[:, :target_patch, :], tokens[:, target_patch:, :]

        tokens = torch.cat([input_tokens, target_tokens], dim=1)
        return tokens, zs_tilde
            


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
    
    
    def forward(self, data_batch, zs_label, input, target, has_target_image=True, detach=False, train=True):

        # Process input images
        posed_input_images = self.get_posed_input(
            images=input.image, ray_o=input.ray_o, ray_d=input.ray_d
        )
        b, v_input, c, h, w = posed_input_images.size()
        # [I; P]
        rgbp_token = self.rgbp_tokenizer(posed_input_images)  # [b*v, n_patches, d]
        # x = Linear([I; P]) (b, np, d)
        _, n_patches, d = rgbp_token.size()  # [b*v, n_patches, d]
        rgbp_token = rgbp_token.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]

        if self.image_encoder is not None:
            input_img_features = self.get_image_feature(input.image)
            input_img_features = input_img_features.reshape(b, v_input * n_patches, d)  # [b, v*n_patches, d]
            input_img_tokens = torch.cat((input_img_features, rgbp_token), dim=2)  # [b, v*n_patches, d*2]
            input_img_tokens = self.align_projector(input_img_tokens) # [b, v*n_patches, d]
        else:
            input_img_tokens = rgbp_token
        
        # lvsm:256*256/(16*16)=256  dino:224*224/(14*14)=256 pe:448*448/(14*14)
        target_pose_cond= self.get_posed_input(ray_o=target.ray_o, ray_d=target.ray_d)

        b, v_target, c, h, w = target_pose_cond.size()
        target_pose_tokens = self.target_pose_tokenizer(target_pose_cond) # [b*v, n_patches, d]
        # P = Linear([P]) (b*out, np, d)
        # Repeat input tokens for each target view
        repeated_input_img_tokens = repeat(
            input_img_tokens, 'b np d -> (b v_target) np d', 
            v_target=v_target, np=n_patches * v_input
        ) # x = (b*out, np*in, d)

        # Concatenate input and target tokens
        transformer_input = torch.cat((repeated_input_img_tokens, target_pose_tokens), dim=1)
        # x = [MLP(I;P);NLP(P_out)] (b*out, np*in+np, d)   
        concat_img_tokens = self.transformer_input_layernorm(transformer_input)
        checkpoint_every = self.config.training.grad_checkpoint_every
        transformer_output_tokens, zs_tilde = self.pass_layers(concat_img_tokens, v_input * n_patches, gradient_checkpoint=False, checkpoint_every=checkpoint_every)

        # Discard the input tokens
        _, target_image_tokens = transformer_output_tokens.split(
            [v_input * n_patches, n_patches], dim=1
        ) # [b * v_target, v*n_patches, d], [b * v_target, n_patches, d]

        # REPA
        if zs_tilde is None and self.config.training.enable_repa:
            print('REPA at last layer')
            if detach:
                target_image_tokens_ = target_image_tokens.clone().detach()
                zs_tilde = self.projectors(target_image_tokens_)
            else:
                zs_tilde = self.projectors(target_image_tokens)

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
            loss_metrics = self.loss_computer(
                rendered_images,
                target.image,
                zs_tilde,
                zs_label,
                train=(train and self.config.training.enable_repa)
            )
        else:
            loss_metrics = None

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=rendered_images        
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
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        state_dict = checkpoint["model"]
        if not self.config.training.use_compile:
            logger.info("discard _orig_mod. in loading model")
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)
        return 0


