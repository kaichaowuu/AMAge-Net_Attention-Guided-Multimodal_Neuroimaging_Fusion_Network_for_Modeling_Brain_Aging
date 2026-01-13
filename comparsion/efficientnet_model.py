import torch
from torch import nn
import torch.nn.functional as F

from utils_efficient import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv3d,
    Swish,
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet3d,
    efficientnet_params
)


class MBConvBlock3D(nn.Module):
    """MBConvBlock for EfficientNet3D (mobile inverted residual bottleneck)"""

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_momentum = global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        Conv3d = get_same_padding_conv3d(image_size=image_size)
        self.expand_ratio = block_args.expand_ratio
        inp = block_args.input_filters
        oup = block_args.input_filters * block_args.expand_ratio
        self._image_size = image_size

        # Expansion phase
        if self.expand_ratio != 1:
            self._expand_conv = Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_momentum, eps=self._bn_eps)
        else:
            self._expand_conv = None

        # Depthwise convolution phase
        self._depthwise_conv = Conv3d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=block_args.kernel_size,
            stride=block_args.stride[0],
            bias=False,
        )
        self._bn1 = nn.BatchNorm3d(num_features=oup, momentum=self._bn_momentum, eps=self._bn_eps)

        # Squeeze and Excitation
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * block_args.se_ratio))
            self._se_reduce = nn.Conv3d(oup, num_squeezed_channels, 1)
            self._se_expand = nn.Conv3d(num_squeezed_channels, oup, 1)

        # Output phase
        final_oup = block_args.output_filters
        self._project_conv = Conv3d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm3d(num_features=final_oup, momentum=self._bn_momentum, eps=self._bn_eps)

        self._swish = Swish()

    def forward(self, inputs):
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        x = self._project_conv(x)
        x = self._bn2(x)

        if self.id_skip and self._block_args.stride == [1] and self._block_args.input_filters == self._block_args.output_filters:
            # if self.training and 0 < self._block_args.drop_connect_rate:
            if self.training and 0 < getattr(self._block_args, 'drop_connect_rate', 0.0):
                x = drop_connect(x, p=self._block_args.drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet3D(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert blocks_args is not None and global_params is not None

        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        # Stem
        in_channels = 1  # 单通道输入
        out_channels = round_filters(32, global_params)
        self._conv_stem = Conv3d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm3d(num_features=out_channels, momentum=global_params.batch_norm_momentum,
                                  eps=global_params.batch_norm_epsilon)
        self._swish = Swish()

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params),
            )
            self._blocks.append(MBConvBlock3D(block_args, global_params, image_size=global_params.image_size))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=[1])
                for _ in range(block_args.num_repeat - 1):
                    self._blocks.append(MBConvBlock3D(block_args, global_params, image_size=global_params.image_size))

        # Head
        in_channels = self._blocks_args[-1].output_filters
        in_channels = round_filters(in_channels, global_params)
        self._conv_head = Conv3d(in_channels, round_filters(1280, global_params), kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(num_features=round_filters(1280, global_params),
                                  momentum=global_params.batch_norm_momentum,
                                  eps=global_params.batch_norm_epsilon)

        # Final linear layer for regression (脑龄预测)
        self._dropout = nn.Dropout(global_params.dropout_rate)
        self._fc = nn.Linear(round_filters(1280, global_params), 1)  # 输出1个脑龄值

        self._initialize_weights()

    def forward(self, inputs):
        x = inputs
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        for block in self._blocks:
            x = block(x)

        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        x = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self._dropout(x)
        x = self._fc(x)
        return x.squeeze(-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def get_efficientnet3d_model(model_name='efficientnet-b0', image_size=128, dropout_rate=0.2):
    blocks_args, global_params = efficientnet3d(
        *efficientnet_params(model_name)[:2],
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        image_size=image_size,
        num_classes=1
    )
    model = EfficientNet3D(blocks_args, global_params)
    return model


class SmriModel(nn.Module):
    def __init__(self, model_name='efficientnet-b0', image_size=128, dropout_rate=0.2):
        super().__init__()
        self.model = get_efficientnet3d_model(model_name=model_name,image_size=image_size,dropout_rate=dropout_rate)

    def forward(self, x):
        return self.model(x)
