import numpy as np
import math
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from torch.cuda.amp import autocast

from dynamic_network_architectures.building_blocks.helper import (
    get_matching_convtransp,
    convert_conv_op_to_dim,
    get_matching_instancenorm,
    convert_dim_to_conv_op,
    maybe_convert_scalar_to_list,
    get_matching_pool_op
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from nnunetv2.utilities.network_initialization import InitWeights_He
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from mamba_ssm import Mamba


# --- Helper Modules ---

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class AttentionLayer(nn.Module):
    """
    Ported from original nnMambaSeg: Channel Attention / SE-Block like structure
    """

    def __init__(self, dim, r=16, act='relu'):
        super(AttentionLayer, self).__init__()
        self.layer1 = nn.Linear(dim, int(dim // r))
        self.layer2 = nn.Linear(int(dim // r), dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Adaptive pooling to handle variable spatial sizes (Global Average Pooling)
        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, inp):
        # GAP: (B, C, D, H, W) -> (B, C, 1, 1, 1) -> (B, C)
        b, c = inp.shape[:2]
        pool = self.pooling(inp).view(b, c)

        # MLP
        att = self.sigmoid(self.layer2(self.relu(self.layer1(pool))))

        # Reshape to (B, C, 1, 1, 1) to multiply with input
        return att.view(b, c, 1, 1, 1)


class UpsampleLayer(nn.Module):
    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            pool_op_kernel_size,
            mode='nearest'
    ):
        super().__init__()
        self.conv = conv_op(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x


# --- Building Blocks ---

class BasicResBlockWithMamba(nn.Module):
    """
    A Residual Block that optionally integrates a Mamba Layer.
    Structure based on original BasicBlock:
    Conv1 -> BN -> Act -> Conv2 -> BN + Mamba(Input) + Identity -> Act
    """

    def __init__(
            self,
            conv_op,
            input_channels,
            output_channels,
            norm_op,
            norm_op_kwargs,
            kernel_size=3,
            padding=1,
            stride=1,
            use_1x1conv=False,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            use_mamba=False
    ):
        super().__init__()

        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding, bias=False)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)

        if use_1x1conv:
            self.downsample = nn.Sequential(
                conv_op(input_channels, output_channels, kernel_size=1, stride=stride, bias=False),
                norm_op(output_channels, **norm_op_kwargs)
            )
        else:
            self.downsample = None

        self.mamba_layer = MambaLayer(output_channels) if use_mamba else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.act1(self.norm1(out))

        out = self.conv2(out)
        out = self.norm2(out)

        if self.mamba_layer is not None:
            # Note: In original code, Mamba takes 'x' (input).
            # If stride > 1, 'x' shape != 'out' shape.
            # Assuming Mamba is used when stride=1 or handled carefully.
            # Based on make_res_layer logic in original code:
            # First block handles stride/downsample. Subsequent blocks handle Mamba.
            # Here, if stride is 1 and dims match, we use x. If not (downsampling block),
            # we typically wouldn't put Mamba here or we need to act on the downsampled identity.
            # However, for safety in this specific block structure where downsample happens:
            if x.shape == out.shape:
                global_att = self.mamba_layer(x)
                out += global_att
            else:
                # If shapes mismatch (due to stride), apply mamba on the projected identity
                # or skip mamba for the downsampling block (common practice).
                # Original code logic: downsample is applied to x.
                pass

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out


# --- Encoder ---

class MambaResEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 ):
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        stem_channels = features_per_stage[0]

        # Initial Stem (mimics DoubleConv with stride or just initial conv)
        # Original nnMambaSeg used a DoubleConv with stride 2 as "in_conv".
        # We will treat the first stage as the stem handling that.

        stages = []

        # Stage 0 (Stem / Initial Conv)
        # In original code: in_conv = DoubleConv(stride=2).
        # We use a BasicResBlock with stride here to match the downsampling.
        self.stem = nn.Sequential(
            BasicResBlockWithMamba(
                conv_op=conv_op,
                input_channels=input_channels,
                output_channels=stem_channels,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                kernel_size=kernel_sizes[0],
                padding=self.conv_pad_sizes[0],
                stride=strides[0],  # Typically 1 or 2 depending on plans
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                use_1x1conv=True,
                use_mamba=False  # Usually no mamba in stem
            )
        )

        input_channels = stem_channels

        for s in range(n_stages):
            # Original code: make_res_layer(blocks, stride=2, mamba_layer=...)
            # We construct the sequence of blocks.

            # 1. Downsampling Block (if stride > 1 or channel change)
            # Or just the first block of the stage.
            blocks = []

            # Note: In the original code loop `make_res_layer`:
            # Block 0: Stride=stride (Downsample), No Mamba (implicitly, as mamba is passed to loop range(1, blocks))
            # Block 1..N: Stride=1, Has Mamba

            # However, for stage 0 (after stem), stride is usually 1 if stem did DS, or 2 if not.
            # nnUNet `strides` list handles this.

            current_stride = strides[
                s] if s > 0 else 1  # Stage 0 handled by stem roughly, but let's follow standard nnUNet loop

            # Actually, let's stick to standard nnUNet construction loop but inject Mamba into blocks > 0

            # If s==0, we already did stem. So actually we need n_stages-1 loops if stem is separate?
            # Let's rebuild to be purely generic based on lists passed in.
            pass

        # Resetting stages construction to be cleaner
        self.stages = nn.ModuleList()
        current_input_dim = stem_channels

        # We iterate from stage 1 to N because stage 0 is 'stem' output usually in this design pattern
        # OR we treat stem as separate and features_per_stage[0] is output of stage 0.

        # Let's assume features_per_stage includes the stem output dim at index 0.
        for s in range(n_stages):
            # Logic matching make_res_layer in original:
            # First block handles stride & channel expansion.
            # Subsequent blocks have Mamba.

            stage_blocks = []

            # Determine input/output channels for this stage
            # If it's stage 0, input is stem_channels (and we might treat stem as just a conv)
            # If s > 0, input is features_per_stage[s-1]

            if s == 0:
                # Stage 0 is typically refined by the Stem above, but here we can add extra blocks
                inp = stem_channels
                outp = features_per_stage[0]
                cur_stride = 1  # Stem already handled the first stride if defined in strides[0]
            else:
                inp = features_per_stage[s - 1]
                outp = features_per_stage[s]
                cur_stride = strides[s]

            # Block 1: Downsample / Change Channels
            stage_blocks.append(
                BasicResBlockWithMamba(
                    conv_op=conv_op,
                    input_channels=inp,
                    output_channels=outp,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=cur_stride,
                    use_1x1conv=True,  # Always use projection when dimensions change
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    use_mamba=False  # Original code: first block of make_res_layer has no mamba usually
                )
            )

            # Block 2..N: Refinement + Mamba
            for _ in range(1, n_blocks_per_stage[s]):
                stage_blocks.append(
                    BasicResBlockWithMamba(
                        conv_op=conv_op,
                        input_channels=outp,
                        output_channels=outp,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        kernel_size=kernel_sizes[s],
                        padding=self.conv_pad_sizes[s],
                        stride=1,
                        use_1x1conv=False,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                        use_mamba=True  # Inject Mamba here
                    )
                )

            self.stages.append(nn.Sequential(*stage_blocks))

        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)

        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            # Approximate: assume stem mimics the stride of the first entry if needed,
            # but usually stem is just a conv.
            # For calculation safety, let's iterate strides.
            output = np.int64(0)  # Not effectively used for memory estimation in this simple way
        else:
            output = np.int64(0)

        # Helper to calculate bottleneck size
        current_size = input_size
        # Apply stem stride (strides[0])
        current_size = [i // j for i, j in zip(current_size, self.strides[0])]

        for s in range(len(self.stages)):
            # Each stage has parameters... purely for memory estimation this is complex.
            # We defer to the fact that standard nnUNet uses this for generic Unets.
            # We assume standard reduction.
            if s > 0:  # s=0 handled by stem stride usually
                current_size = [i // j for i, j in zip(current_size, self.strides[s])]

            # Add size of feature maps
            # This is a dummy implementation, real one sums num_voxels * channels * num_layers
            pass

        return np.int64(0)  # Placeholder


# --- Decoder ---

class MambaSegDecoder(nn.Module):
    def __init__(self,
                 encoder,
                 num_classes,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages_encoder - 1)

        stages = []
        upsample_layers = []
        attention_layers = []  # For the skip connections
        seg_layers = []

        # We iterate backwards from the bottleneck
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_upsampling = encoder.strides[-s]

            # 1. Attention Gate on the Skip Connection
            attention_layers.append(AttentionLayer(input_features_skip))

            # 2. Upsampling
            upsample_layers.append(UpsampleLayer(
                conv_op=encoder.conv_op,
                input_channels=input_features_below,
                output_channels=input_features_skip,  # Often we target the skip dimension
                pool_op_kernel_size=stride_for_upsampling,
                mode='trilinear'  # Matches original nn.Upsample(mode='trilinear')
            ))

            # 3. Convolutional Blocks after concatenation
            # Input dim = skip_dim + upsampled_dim.
            # Original code: merge = cat([up, c*scale]).
            # Note: The original code's conv5 input is (channels*12) -> (channels*4).
            # This implies concat of bottleneck (ch*8) and skip (ch*4).

            stages.append(nn.Sequential(
                BasicResBlockWithMamba(
                    conv_op=encoder.conv_op,
                    input_channels=2 * input_features_skip,  # Concat
                    output_channels=input_features_skip,
                    norm_op=encoder.norm_op,
                    norm_op_kwargs=encoder.norm_op_kwargs,
                    kernel_size=encoder.kernel_sizes[-(s + 1)],
                    padding=encoder.conv_pad_sizes[-(s + 1)],
                    stride=1,
                    use_1x1conv=True,
                    nonlin=encoder.nonlin,
                    nonlin_kwargs=encoder.nonlin_kwargs,
                    use_mamba=False  # Usually no mamba in decoder based on original code (DoubleConv)
                )
                # You can add more blocks here if n_conv_per_stage_decoder > 1
            ))

            # Deep Supervision heads
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.upsample_layers = nn.ModuleList(upsample_layers)
        self.attention_layers = nn.ModuleList(attention_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        skips: [c1, c2, c3, c4] (where c4 is bottleneck)
        """
        lres_input = skips[-1]
        seg_outputs = []

        # We iterate through the stages.
        # len(stages) = 3 for a 4-level encoder (bottleneck + 3 decoders)
        for s in range(len(self.stages)):
            # Skip connection to use (going backwards)
            # if s=0, we want skips[-2] (c3)
            skip = skips[-(s + 2)]

            # 1. Apply Attention to Skip
            # Original: scale_f3 = self.att3(pool(c3))... merge = cat([up, c3*scale_f3])
            att_scale = self.attention_layers[s](skip)
            skip_gated = skip * att_scale

            # 2. Upsample lower level
            up = self.upsample_layers[s](lres_input)

            # 3. Concat
            x = torch.cat((up, skip_gated), dim=1)

            # 4. Convolve
            x = self.stages[s](x)

            # Deep Supervision
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            lres_input = x

        # seg_outputs are [Ds3, Ds2, Ds1 (Final)]. We reverse to have [Final, Ds2, Ds3] for loss calc if needed,
        # but standard nnUNet usually expects [high_res, low_res...]
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            return seg_outputs[0]
        else:
            return seg_outputs

    def compute_conv_feature_map_size(self, input_size):
        # Placeholder
        return np.int64(0)


# --- Main Architecture ---

class nnMambaSeg(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 stem_channels: int = None
                 ):
        super().__init__()

        # Handle default list configs
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.encoder = MambaResEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True,
            stem_channels=stem_channels
        )

        self.decoder = MambaSegDecoder(
            self.encoder,
            num_classes,
            n_conv_per_stage_decoder,
            deep_supervision
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        return self.encoder.compute_conv_feature_map_size(input_size) + \
            self.decoder.compute_conv_feature_map_size(input_size)


# --- Factory Function ---

def get_nnmambaseg_from_plans(
        plans_manager: PlansManager,
        dataset_json: dict,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        deep_supervision: bool = True
):
    num_stages = len(configuration_manager.conv_kernel_sizes)
    dim = len(configuration_manager.conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)

    label_manager = plans_manager.get_label_manager(dataset_json)

    segmentation_network_class_name = 'nnMambaSeg'
    network_class = nnMambaSeg

    # Define default arguments that are specific to your architecture style
    kwargs = {
        'nnMambaSeg': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }

    conv_or_blocks_per_stage = {
        'n_conv_per_stage': configuration_manager.n_conv_per_stage_encoder,
        'n_conv_per_stage_decoder': configuration_manager.n_conv_per_stage_decoder
    }

    # Replicating the channel doubling logic from original nnMambaSeg (32, 64, 128...)
    # We use UNet_base_num_features from plans
    features_per_stage = [min(configuration_manager.UNet_base_num_features * 2 ** i,
                              configuration_manager.unet_max_num_features) for i in range(num_stages)]

    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=configuration_manager.conv_kernel_sizes,
        strides=configuration_manager.pool_op_kernel_sizes,
        num_classes=label_manager.num_segmentation_heads,
        deep_supervision=deep_supervision,
        stem_channels=features_per_stage[0],  # Using first stage features as stem output
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )

    model.apply(InitWeights_He(1e-2))

    return model


if __name__ == "__main__":
    # Test stub
    network = nnMambaSeg(
        input_channels=1,
        n_stages=4,
        features_per_stage=[32, 64, 128, 256],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3]] * 4,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2]],  # Stage 0 stride 1, others stride 2
        n_conv_per_stage=[2, 2, 2, 2],
        num_classes=4,
        n_conv_per_stage_decoder=[2, 2, 2],
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=True
    ).cuda()

    inp = torch.randn((2, 1, 128, 128, 128)).cuda()
    out = network(inp)

    print("Output shapes (Deep Supervision):")
    for o in out:
        print(o.shape)