# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    """
    对输入数据x进行调制。

    该函数通过应用一个缩放和平移操作来调制输入数据x。这种调制机制常用于神经网络中，
    以实现对数据的非线性变换，可以看作是批归一化的一个逆操作或风格转换的一部分。

    参数:
    - x: 输入数据，通常是一个特征图或神经网络的激活输出。
    - shift: 平移量，用于对输入数据进行平移操作的张量。
    - scale: 缩放因子，用于对输入数据进行缩放操作的张量。

    返回值:
    返回调制后的数据, 即经过缩放和平移变换后的输入数据x。
    """
    # 对输入x进行调制，结合缩放因子和平移量
    # unqueeze(1)用于增加一个维度，以匹配输入x的维度，确保广播配对正确
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

# note：将时间步 t 编码成一个固定维度的向量表示，便于模型在处理扩散过程时利用时间信息。
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    该类用于将标量时间步(timesteps)嵌入为向量表示。

    它首先使用正弦和余弦函数生成频率编码(sinusoidal frequency encoding)，然后通过一个 MLP(多层感知器)
    将其映射到更高维度的空间，以供模型使用。

    属性:
    - mlp: 多层感知器，用于将频率编码转换为最终的时间步嵌入。
    - frequency_embedding_size: 频率编码的维度大小。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        初始化 TimestepEmbedder 类。

        参数:
        - hidden_size (int): 输出嵌入的维度大小。
        - frequency_embedding_size (int): 频率编码的维度, 默认为256。

        返回:
        None
        """
        super().__init__()
        # 定义MLP网络结构：输入为频率编码，输出为高维嵌入
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # 使用SiLU激活函数（即Swish）
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        生成正弦和余弦形式的时间步嵌入。

        这种方法基于不同频率的正弦和余弦波来编码时间步信息，是一种常见的位置编码方式。

        参数:
        - t (Tensor): 一维张量，形状为(N,)，表示N个时间步索引（可以是小数）。
        - dim (int): 输出嵌入的维度。
        - max_period (int): 控制最低频率的周期，默认为10000。

        返回:
        - embedding (Tensor): 形状为(N, D)的时间步嵌入张量。
        """
        half = dim // 2
        # 计算频率因子，指数衰减确保低频到高频的变化
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0,
                                                 end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # 对每个时间步乘以频率因子，得到相位参数
        args = t.float() * freqs[None]
        # 拼接cos和sin部分作为嵌入向量
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # 如果dim为奇数，在最后补零以保证维度正确
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        前向传播，将输入的时间步 `t` 转换为嵌入向量。

        参数:
        - t (Tensor): 输入的时间步张量，形状为(N,)。

        返回:
        - t_emb (Tensor): 时间步的嵌入表示，形状为(N, hidden_size)。
        """
        # 先生成频率编码，输出：(N, 256)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 再通过MLP映射到目标维度，输出：(N, hidden_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# note：将动作 xy 编码成一个固定维度的向量表示，便于模型在处理动作信息时利用空间信息。
class ActionEmbedder(nn.Module):
    """
    Embeds action xy into vector representations.
    ActionEmbedder类用于将动作[平移u(x,y)和旋转φ]嵌入为向量表示。

    该类通过三个独立的 TimestepEmbedder 实例分别对 x 坐标、y 坐标和角度进行嵌入编码，
    然后将这三个嵌入向量在最后一个维度上拼接，形成一个完整的动作嵌入表示。

    属性:
    - x_emb: 用于嵌入 x 坐标的 TimestepEmbedder。
    - y_emb: 用于嵌入 y 坐标的 TimestepEmbedder。
    - angle_emb: 用于嵌入角度的 TimestepEmbedder。

    方法:
    - __init__: 初始化 ActionEmbedder 类。
    - forward: 前向传播，将输入的动作数据转换为嵌入向量。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        初始化 ActionEmbedder 类。

        参数:
        - hidden_size (int): 动作嵌入的总输出维度。该维度会被分成三部分，分别用于 x 坐标、y 坐标和角度。
        - frequency_embedding_size (int): 频率嵌入的维度，默认为256。

        返回:
        None
        """
        super().__init__()
        hsize = hidden_size // 3  # 将隐藏层大小分为三等分，其中两个用于x和y坐标，剩下的一个用于角度
        self.x_emb = TimestepEmbedder(
            hsize, frequency_embedding_size)  # 创建用于x坐标的嵌入器
        self.y_emb = TimestepEmbedder(
            hsize, frequency_embedding_size)  # 创建用于y坐标的嵌入器
        self.angle_emb = TimestepEmbedder(
            hidden_size - 2 * hsize, frequency_embedding_size)  # 创建用于角度的嵌入器

    def forward(self, xya):
        """
        前向传播，将输入的动作数据转换为嵌入向量。

        参数:
        - xya: 输入的动作数据，形状为(..., 3)，其中最后一维包含 x 坐标、y 坐标和角度。

        返回:
        - 返回动作的嵌入表示，形状为(..., hidden_size)，其中 hidden_size 是指定的输出维度。
        """
        # 分别对x坐标、y坐标和角度进行嵌入编码，并在最后一个维度上拼接
        return torch.cat([
            self.x_emb(xya[..., 0:1]),   # 提取并嵌入x坐标
            self.y_emb(xya[..., 1:2]),   # 提取并嵌入y坐标
            self.angle_emb(xya[..., 2:3])  # 提取并嵌入角度
        ], dim=-1)

#################################################################################
#                                 Core CDiT Model                                #
#################################################################################


class CDiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.cttn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 11 * hidden_size, bias=True)
        )

        self.norm3 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, x_cond):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(11, dim=1)
        x = x + \
            gate_msa.unsqueeze(
                1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x_cond_norm = modulate(self.norm_cond(
            x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(query=modulate(self.norm2(
            x), shift_ca_x, scale_ca_x), key=x_cond_norm, value=x_cond_norm, need_weights=False)[0]
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    DiT 模型的最后一层，用于将 Transformer 输出的特征映射回图像空间。

    该层包含一个自适应归一化（adaLN）调制模块、一个 LayerNorm 层和一个线性变换层，
    用于将模型输出的高维特征向量转换为图像 patch 的像素值。

    属性:
    - norm_final: 最终的 LayerNorm 层，不包含可学习的缩放和平移参数（elementwise_affine=False）。
    - linear: 将隐藏维度映射到每个图像 patch 所需像素值数量的线性层。
    - adaLN_modulation: 两层的 MLP，用于生成 adaLN 的 shift 和 scale 参数。
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        """
        初始化 FinalLayer 类。

        参数:
        - hidden_size (int): Transformer 中的隐藏层维度。
        - patch_size (int): 图像 patch 的大小（如 2x2），决定每个 patch 包含的像素数。
        - out_channels (int): 输出图像的通道数（例如 4 表示潜在空间表示，8 表示预测均值和方差）。

        返回:
        None
        """
        super().__init__()
        # 最终的 LayerNorm 层，无 learnable 参数（即不使用 gamma/beta）
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        # 线性层：将隐藏维度映射到每个 patch 的像素数（patch_size^2 * out_channels）
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        # adaLN 调制网络，输出两个部分：shift 和 scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),  # SiLU 激活函数
            # 输出是 2*hidden_size，用于 shift 和 scale
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        前向传播函数，将输入特征 x 和条件信息 c 映射到输出图像空间。

        参数:
        - x (Tensor): 输入特征张量，形状为 (N, T, D)，其中：
                      N = batch size
                      T = token 数量（等于 H * W / patch_size^2）
                      D = hidden_size
        - c (Tensor): 条件信息张量，形状为 (N, D)，来自时间步、动作或上下文等。

        返回:
        - x (Tensor): 输出图像张量，形状为 (N, T, patch_size**2 * out_channels)
        """
        # 通过 adaLN 调制网络生成 shift 和 scale
        shift, scale = self.adaLN_modulation(c).chunk(
            2, dim=1)  # chunk 分割成两个张量，分别用于 shift 和 scale

        # 应用 adaLN 调制 + LayerNorm
        x = modulate(self.norm_final(x), shift, scale)

        # 通过最终的线性层，将隐藏状态映射到图像像素空间
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    基于 Transformer 的扩散模型。

    该类实现了一个基于 Transformer 的扩散模型，用于处理图像生成任务。它通过将输入图像分割成 patch，
    并对这些 patch 进行嵌入编码，结合时间步、动作等条件信息，最终生成目标图像。

    属性:
    - context_size (int): 上下文的大小，表示模型在处理当前帧时，需要考虑的历史帧数，通常用于视频生成任务。
    - learn_sigma (bool): 控制是否学习生成图像的噪声水平，如果为 True，则输出通道数为 in_channels * 2，分别表示均值和方差。
    - in_channels (int): 输入图像的通道数。
    - out_channels (int): 输出图像的通道数，如果 [learn_sigma](file:///root/private/WorldModel/nwm/models.py#L0-L0) 为 True，则输出通道数为 `in_channels * 2`。
    - patch_size (int): 图像 patch 的大小，表示每个 patch 的宽度和高度。
    - num_heads (int): Transformer 中多头注意力机制的头数。

    - x_embedder (PatchEmbed): 将【输入图像】分割成 patch 并进行嵌入编码，以便后续的 Transformer 处理。
    - t_embedder (TimestepEmbedder): 用于将【时间步】嵌入为向量表示的模块，作为条件信息输入模型。
    - y_embedder (ActionEmbedder): 用于将【动作信息】嵌入为向量表示的模块，作为条件信息输入模型。

    - pos_embed (nn.Parameter): 位置编码，用于为每个 patch 添加位置信息，帮助模型理解 patch 在图像中的位置。
    - blocks (nn.ModuleList): 包含多个[CDiTBlock]的模块列表，每个 CDiTBlock 是一个 Transformer 层，用于处理嵌入后的 patch。
    - final_layer (FinalLayer): 最后一层，用于将 Transformer 输出的高维特征映射回图像空间，生成最终的图像 patch。

    - time_embedder (TimestepEmbedder): 用于将【相对时间】嵌入为向量表示的模块，作为条件信息输入模型。
    """

    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        """
        初始化 CDiT 类。

        参数:
        - input_size (int): 输入图像的尺寸，默认为 32。
        - context_size (int): 上下文的大小，默认为 2。
        - patch_size (int): 图像 patch 的大小，默认为 2。
        - in_channels (int): 输入图像的通道数，默认为 4。? 应该是 RGBA 吧！
        - hidden_size (int): Transformer 的隐藏层维度，默认为 1152。
        - depth (int): Transformer 的层数，默认为 28。
        - num_heads (int): 多头注意力机制的头数，默认为 16。
        - mlp_ratio (float): MLP 层的隐藏层维度与输入维度的比例，默认为 4.0。
        - learn_sigma (bool): 是否学习 sigma，默认为 True。

        返回:
        None
        """
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches  # num_patches：表示图像被分割成多少个 patch。
        # for context and for predicted frame --- 为上下文和预测帧生成位置编码
        self.pos_embed = nn.Parameter(torch.zeros(
            self.context_size + 1, num_patches, hidden_size), requires_grad=True)
        self.blocks = nn.ModuleList(
            [CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels)
        self.time_embedder = TimestepEmbedder(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):

        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize action embedding:
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

# 将模型输出的图像块（patches）重新组合成完整的图像，输出形状为 (N, H, W, C)
    def unpatchify(self, x):
        """
        将patch转换回原始图像尺寸。

        x: (N, T, patch_size**2 * C), N 是 batch_size, T 是图像块的数量, patch_size**2 * C 表示每个图像块展开后的像素数(C 是通道数)
        imgs: (N, H, W, C), H 和 W 是图像的高度和宽度
        """

        # 获取输出通道数
        c = self.out_channels
        # 获取patch的尺寸
        p = self.x_embedder.patch_size[0]
        # 计算图像的高度和宽度(x.shape[1] 是图像块的数量 T )
        h = w = int(x.shape[1] ** 0.5)
        # 确保高度和宽度的乘积等于patch的长度
        assert h * w == x.shape[1]

        # 将patch重新整形为六维张量，便于后续操作
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # 使用einsum()函数改变维度顺序，将通道维度提到前面
        x = torch.einsum('nhwpqc->nchpwq', x)
        # 将六维张量重新整形为四维，恢复图像格式
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        # 返回重建的图像张量
        return imgs

    def forward(self, x, t, y, x_cond, rel_t):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # 对输入图像 x 进行嵌入，并加上位置编码（从 context_size 开始）
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]

        # (N, T, D), where T = H * W / patch_size ** 2.flatten(1, 2)
        # 将条件图像 x_cond 展平后进行嵌入，再恢复形状并加上位置编码（前 context_size 个位置）
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(
            0, (x_cond.shape[0], x_cond.shape[1])) + self.pos_embed[:self.context_size]

        # 再次展平 x_cond，以便后续处理
        x_cond = x_cond.flatten(1, 2)
        # 对时间步 t 进行嵌入
        t = self.t_embedder(t[..., None])
        # 对动作标签 y 进行嵌入
        y = self.y_embedder(y)
        # 对相对时间 rel_t 进行嵌入
        time_emb = self.time_embedder(rel_t[..., None])
        # 合并时间嵌入、相对时间嵌入和动作嵌入作为最终的条件向量 c
        c = t + time_emb + y  # if training on unlabeled data, dont add y.

        # 通过每个 CDiTBlock 处理 x，传入条件 c 和条件输入 x_cond
        for block in self.blocks:
            x = block(x, c, x_cond)

        # 最终层处理 x，传入条件 c
        x = self.final_layer(x, c)
        # 将 patch 转换回原始图像尺寸
        x = self.unpatchify(x)

        # 返回输出图像张量
        return x

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                  #
#################################################################################

# 由配置文件 config 中的 nwm_cdit_xl.yaml 决定
def CDiT_XL_2(**kwargs):
    # 这里使用了 **kwargs 语法，将所有传入 CDiT_XL_2 函数的额外关键字参数转发给 CDiT 函数
    return CDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2,
    'CDiT-L/2':  CDiT_L_2,
    'CDiT-B/2':  CDiT_B_2,
    'CDiT-S/2':  CDiT_S_2
}
