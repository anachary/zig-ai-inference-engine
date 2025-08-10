const std = @import("std");

// Format and Architecture identifiers
pub const FormatTag = enum { gguf, unknown };
pub const ArchitectureTag = enum { llama, qwen, unknown };

// Data types and quantization
pub const DType = enum {
    f32,
    f16,
    bf16,
    i32,
    q8_0,
    q5_k,
    q4_k_m,
};

pub const QuantType = enum { none, ggml };

// Normalization and activations (RMSNorm: Zhang & Sennrich, 2019; SwiGLU: Shazeer, 2020)
pub const NormType = enum { rmsnorm, layernorm };
pub const ActivationType = enum { swiglu, gelu, silu, relu };

// Attention variants (MQA/GQA per Shazeer 2019 et al.)
pub const AttentionKind = enum { mha, mqa, gqa };

// Positional encodings (RoPE per Su et al., 2021)
pub const PositionEncoding = enum { rope, alibi, none };
pub const RopeVariant = enum { llama, ntk_aware, x_pos };

pub const RopeConfig = struct {
    variant: RopeVariant = .llama,
    theta: f32 = 10000.0, // LLaMA default
    alpha: f32 = 1.0, // NTK-aware scaling
};

// Backends / device hints (future extensibility)
pub const Backend = enum { pure_zig, simd };
pub const Device = enum { cpu_x86_64, cpu_arm64 };

// Sampling params (align with common LLM controls)
pub const SamplingParams = struct {
    temperature: f32 = 0.7,
    top_k: u32 = 40,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    min_p: f32 = 0.0,
    seed: u64 = 0,
};

// Generic result + errors
pub const Result = enum { ok, unimplemented };

pub const Error = error{
    Unimplemented,
    Unsupported,
    InvalidFormat,
    InvalidArchitecture,
    ParseFailed,
};
