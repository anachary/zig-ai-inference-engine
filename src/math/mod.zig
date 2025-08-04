const std = @import("std");

// Core mathematical modules
pub const matrix = @import("matrix.zig");
pub const activations = @import("activations.zig");
pub const normalization = @import("normalization.zig");
pub const attention = @import("attention.zig");
pub const embeddings = @import("embeddings.zig");
pub const simd = @import("simd.zig");

// Re-export commonly used types
pub const Matrix = matrix.Matrix;
pub const MatrixView = matrix.MatrixView;

// Re-export common functions
pub const matmul = matrix.matmul;
pub const softmax = activations.softmax;
pub const layerNorm = normalization.layerNorm;
pub const scaledDotProductAttention = attention.scaledDotProductAttention;
