import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "naive" ##### debug "absorb"

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    # dim: int = 2048
    dim: int = 128 ##### debug 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    # n_layers: int = 27
    n_layers: int = 2 ##### debug 27
    n_dense_layers: int = 1
    # n_heads: int = 16
    n_heads: int = 4 ##### debug 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats. 应用线性变换 y = xA^T + b，支持多种量化计算模式
    这个函数是模型计算的核心组件之一，通过量化技术实现了计算效率和精度的平衡。适配不同精度的全连接层计算。

    Args:
        x (torch.Tensor): The input tensor. 输入张量
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases. 权重矩阵，可能被量化
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None. 可选的偏置项

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters. 线性变换结果张量

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation. 如果权重未量化(元素大小>1字节)，直接使用标准线性变换
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied. 如果使用bf16模式，先反量化权重再计算
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation. 其他情况(如fp8模式)，先量化输入再使用fp8矩阵乘法
    """
    if weight.element_size() > 1: # 情况1：权重未量化(32位浮点等)
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16": # 情况2：使用bf16模式
        weight = weight_dequant(weight, weight.scale) # 先对量化权重进行反量化
        return F.linear(x, weight, bias) # 使用反量化后的权重计算线性变换
    else: # 情况3：其他量化模式(如fp8)
        x, scale = act_quant(x, block_size) # 对输入进行量化
        y = fp8_gemm(x, scale, weight, weight.scale) # 使用fp8矩阵乘法
        if bias is not None: # 添加偏置(如果有)
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.
    并行线性层，将输入特征拆分到多个进程并行计算，最后合并结果。
    适用于分布式训练场景，每个进程处理输入特征的一部分。

    
    Args:
        in_features (int): Total number of input features. 输入特征数，必须能被world_size整除
        out_features (int): Number of output features. 输出特征数
        bias (bool): Whether to include a bias term. Defaults to False. 是否包含偏置项
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`. 数据类型
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    RoPE（旋转位置编码）是一种将位置信息通过旋转变换引入到模型输入中的方法，其核心思想是将输入向量中的一部分看作复数（由实部和虚部构成），并与预先计算好的正弦余弦复数因子相乘，从而实现“旋转”操作。这样，相邻位置之间的关系就能自然地体现为旋转角度的差异，从而捕捉到相对位置信息。

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features. 输入特征的维度。
        n_heads (int): Number of attention heads. 注意力头的数量。
        n_local_heads (int): Number of local attention heads for distributed systems. 分布式训练中本地处理的注意力头数
        q_lora_rank (int): Rank for low-rank query projection. 查询投影的低秩(LoRA)秩。
        kv_lora_rank (int): Rank for low-rank key/value projection. 键/值投影的低秩(LoRA)秩。
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections. 无位置编码的查询/键的头维度。
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections. 带旋转位置编码的查询/键的头维度。
        qk_head_dim (int): Total dimensionality of query/key projections. 查询/键的总头维度。
        v_head_dim (int): Dimensionality of value projections. 值投影的头维度。
        softmax_scale (float): Scaling factor for softmax in attention computation. 注意力计算中softmax的缩放因子。
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 模型维度参数
        self.dim = args.dim
        # 注意力头总数
        self.n_heads = args.n_heads  
        # 分布式环境下本地处理的注意力头数
        self.n_local_heads = args.n_heads // world_size
        # LoRA相关参数
        self.q_lora_rank = args.q_lora_rank  # 查询投影的LoRA秩
        self.kv_lora_rank = args.kv_lora_rank  # 键值投影的LoRA秩
        # 注意力头维度配置
        self.qk_nope_head_dim = args.qk_nope_head_dim  # 无位置编码的QK头维度
        self.qk_rope_head_dim = args.qk_rope_head_dim  # 带旋转位置编码的QK头维度
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim  # QK总头维度
        self.v_head_dim = args.v_head_dim  # 值投影头维度
        
        # 是否需要做低秩压缩
        if self.q_lora_rank == 0:
            # 不压缩直接投影
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim) # 假定参数量为lora*nqk，2048*16*192=630w
        else:
            # 压缩则两阶段投影，低秩压缩参数量为d * lora + lora * nqk = lora * (d + nqk)，512 * (2048+16*192) = 260w
            self.wq_a = Linear(self.dim, self.q_lora_rank)  # LoRA第一阶段
            self.q_norm = RMSNorm(self.q_lora_rank)  # 归一化层，参数量不是一个量级忽略不计
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)  # LoRA第二阶段
        
        # 键值投影分支
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)  # 合并KV和位置编码
        self.kv_norm = RMSNorm(self.kv_lora_rank)  # KV归一化层
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))  # KV投影
        
        # 输出层
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        
        # 注意力分数缩放因子
        self.softmax_scale = self.qk_head_dim ** -0.5
        # 长序列场景进一步调整缩放因子
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 注意力实现方式选择
        if attn_impl == "naive":
            # 基础KV缓存模式
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # 优化的KV缓存(合并KV和位置编码)
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer. 多头注意力层(MLA)的前向传播。

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim). 输入张量，形状为(batch_size, seq_len, dim)
            start_pos (int): Starting position in the sequence for caching. 序列中开始位置，用于缓存
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings. 提前计算的旋转位置编码复数值
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention. 注意力掩码张量，主要用于排除某些位置

        Returns:
            torch.Tensor: Output tensor with the same shape as the input. 输出张量，形状与输入相同
        """
        # 获取输入张量的batch大小和序列长度
        bsz, seqlen, _ = x.size()
        # 计算结束位置
        end_pos = start_pos + seqlen

        # 是否需要做低秩压缩
        if self.q_lora_rank == 0:
            # 不压缩直接投影
            q = self.wq(x)
        else:
            # 压缩则两阶段投影
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # 调整query张量维度并分割为无位置编码和有位置编码部分
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # 对有位置编码部分应用旋转位置编码
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        # 键值投影分支
        kv = self.wkv_a(x)
        # 分割键值张量为KV和位置编码部分
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # 对键的位置编码部分应用旋转位置编码
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # “传统”注意力实现方式，基础流程
        if attn_impl == "naive":
            # 合并无位置编码和有位置编码的查询
            q = torch.cat([q_nope, q_pe], dim=-1)
            # 键值投影，这里就是所谓的latent低秩联合压缩
            kv = self.wkv_b(self.kv_norm(kv))
            # 调整键值张量维度
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            # 分割键值张量为键和值
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # 合并无位置编码和有位置编码的键
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            # 更新键值缓存
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            # 计算注意力分数 (batch_size, seq_len, n_heads, seq_len)
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else: # 改进版运算
            # 获取权重并处理量化情况
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            # 重塑权重形状 (n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            # 计算无位置编码部分的查询投影
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            # 更新键值缓存和位置编码缓存
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
            # 计算注意力分数 (包含无位置编码和有位置编码两部分)
            scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        
        # 应用注意力掩码(如有)
        if mask is not None:
            scores += mask.unsqueeze(1)
        
        # 计算softmax注意力权重
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # 根据实现方式计算注意力输出
        if attn_impl == "naive":
            # 传统方式: 使用值缓存计算输出
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            # 优化方式: 使用键值缓存计算输出
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            # 应用值投影
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        # 输出投影
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer. 改进版的MLP作为FFN

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation. 输入到隐藏层的线性变换
        w2 (nn.Module): Linear layer for hidden-to-output transformation. 隐藏层到输出的线性变换
        w3 (nn.Module): Additional linear layer for feature transformation. 额外的特征变换线性层
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality. 输入和输出维度
            inter_dim (int): Hidden layer dimensionality. 隐藏层维度
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim) # 维度变换
        self.w2 = RowParallelLinear(inter_dim, dim) # 维度恢复
        self.w3 = ColumnParallelLinear(dim, inter_dim) # 维度变换

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        # w1和w3变换维度，w2维度恢复，其实也有点门控的意思，只是比较简单，用w1和silu激活函数生成权重来调整w3生成的特征
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model. MoE里的门控模型

    Attributes:
        dim (int): Dimensionality of input features. 输入特征维度
        topk (int): Number of top experts activated for each input. 每个输入激活的top-k个专家数
        n_groups (int): Number of groups for routing. 路由分组数量，默认1
        topk_groups (int): Number of groups to route inputs to. 输入路由到的top-k组数，默认1
        score_func (str): Scoring function ('softmax' or 'sigmoid'). 评分函数类型
        route_scale (float): Scaling factor for routing weights. 路由权重的缩放因子
        weight (torch.nn.Parameter): Learnable weights for the gate. 可学习的门控权重
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate. 可选的门控偏置项
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        # 输入特征的维度
        self.dim = args.dim
        # 每个输入激活的专家数量，默认是6
        self.topk = args.n_activated_experts
        # 专家分组数量，默认是1不用管
        self.n_groups = args.n_expert_groups
        # 输入路由到的组数，默认是1不用管
        self.topk_groups = args.n_limited_groups
        # 评分函数类型或者说归一化方法('softmax'或'sigmoid')
        self.score_func = args.score_func
        # 路由权重的缩放因子，默认是1不用管
        self.route_scale = args.route_scale
        # 可学习的门控权重(shape: 专家数量×特征维度)
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        # 门控偏置项(shape: 专家数量)，671B的模型用
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        # 计算专家得分 (n, n_experts)
        scores = linear(x, self.weight)

        if self.score_func == "softmax": # softmax归一化
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else: # sigmoid归一化
            scores = scores.sigmoid()

        # 保存原始得分用于后续计算，后面scores仅用于计算topk
        original_scores = scores

        # 如果有偏置项直接相加
        if self.bias is not None:
            scores = scores + self.bias
        
        # 分组逻辑，这里暂时忽略即可
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        
        # topk逻辑，这里是选择得分最高的topk个专家的序号
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # 取出对应序号的得分权重
        weights = original_scores.gather(1, indices)

        # 如果是sigmoid需要进一步处理使其加和为1，softmax略过即可
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale # 缩放，默认参数为1忽略

        # 返回归一化后的专家得分权重和对应的专家索引
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models. 改进版的MLP

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = Linear(dim, inter_dim) # 维度变换
        self.w2 = Linear(inter_dim, dim) # 维度恢复
        self.w3 = Linear(dim, inter_dim) # 维度变换

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        # w1和w3变换维度，w2维度恢复，其实也有点门控的意思，只是比较简单，用w1生成权重来调整w3生成的特征
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module. 混合专家

    Attributes:
        dim (int): Dimensionality of input features. 输入特征维度
        n_routed_experts (int): Total number of experts in the model. 总专家数量
        n_local_experts (int): Number of experts handled locally in distributed systems. 当前进程的本地专家数量
        n_activated_experts (int): Number of experts activated for each input. 每个输入激活的专家数量
        gate (nn.Module): Gating mechanism to route inputs to experts. 门控
        experts (nn.ModuleList): List of expert modules. 专家列表
        shared_experts (nn.Module): Shared experts applied to all inputs. 共享专家
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        # 输入特征维度
        self.dim = args.dim
        # 确保专家数量能被进程数整除(分布式训练要求)
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        # 总专家数量
        self.n_routed_experts = args.n_routed_experts
        # 当前进程负责的本地专家数量
        self.n_local_experts = args.n_routed_experts // world_size
        # 每个输入激活的专家数量
        self.n_activated_experts = args.n_activated_experts
        # 当前进程负责的专家索引范围
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        # 门控机制(路由网络)
        self.gate = Gate(args)
        # 专家列表(仅初始化当前进程负责的专家)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        # 共享专家(所有输入都会处理)
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        # 保存原始输入形状
        shape = x.size()
        # 将输入展平为二维张量 (batch_size * seq_len, dim)
        x = x.view(-1, self.dim)

        # 通过门控网络获取路由权重和专家索引
        weights, indices = self.gate(x)

        # 初始化输出张量
        y = torch.zeros_like(x)

        # 统计每个专家被选中的次数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # 遍历当前进程负责的专家范围
        for i in range(self.experts_start_idx, self.experts_end_idx):
            # 跳过未被选中的专家
            if counts[i] == 0:
                continue

            # 获取当前专家模型
            expert = self.experts[i]
            # 找出选择了当前专家的索引
            idx, top = torch.where(indices == i)
            # 计算专家输出并加权累加
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        
        # 计算共享专家的输出
        z = self.shared_experts(x)
        # 分布式训练处理输出
        if world_size > 1:
            dist.all_reduce(y)
        
        # 将共享专家和选中专家的输出合并，并恢复原始shape
        return (y + z).view(shape)


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.
    包括MLA、MLP或MOE，还有两个RMSNorm

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.
    embeddings、layers、output projection，重点就是中间的multiple layers。

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        seqlen = tokens.size(1)
        h = self.embed(tokens) # 并行改造的embedding层
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers: # layer
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h) # output projection，并行改造的linear层
        if world_size > 1: # 分布式训练的进程总数，忽略即可
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


# 模型验证
if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    # out = model(x)
    # print(x.shape, out.shape)

    from torchinfo import summary
    summary(model, input_data=x)
    