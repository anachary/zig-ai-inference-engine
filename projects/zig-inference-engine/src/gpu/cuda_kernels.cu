// CUDA kernels for Zig AI inference engine
// These kernels provide GPU acceleration for common ML operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>

extern "C" {

// ============================================================================
// Element-wise Operations
// ============================================================================

__global__ void vectorAdd_f32(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorMul_f32(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void vectorSub_f32(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

__global__ void vectorDiv_f32(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

__global__ void relu_f32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void gelu_f32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void sigmoid_f32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void tanh_f32(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

// ============================================================================
// Softmax (optimized with shared memory)
// ============================================================================

__global__ void softmax_f32(const float* input, float* output, int batch_size, int dim) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * dim;
    float* batch_output = output + batch_idx * dim;
    
    extern __shared__ float shared_data[];
    
    // Find maximum value for numerical stability
    float max_val = -INFINITY;
    for (int i = tid; i < dim; i += blockDim.x) {
        max_val = fmaxf(max_val, batch_input[i]);
    }
    
    // Reduce to find global maximum
    shared_data[tid] = max_val;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    
    max_val = shared_data[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float exp_val = expf(batch_input[i] - max_val);
        batch_output[i] = exp_val;
        sum += exp_val;
    }
    
    // Reduce sum
    shared_data[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    sum = shared_data[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < dim; i += blockDim.x) {
        batch_output[i] /= sum;
    }
}

// ============================================================================
// Layer Normalization
// ============================================================================

__global__ void layernorm_f32(const float* input, const float* gamma, const float* beta,
                              float* output, int batch_size, int dim, float epsilon) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = input + batch_idx * dim;
    float* batch_output = output + batch_idx * dim;
    
    extern __shared__ float shared_data[];
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        sum += batch_input[i];
    }
    
    shared_data[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float mean = shared_data[0] / dim;
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float diff = batch_input[i] - mean;
        var_sum += diff * diff;
    }
    
    shared_data[tid] = var_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    float variance = shared_data[0] / dim;
    float inv_std = rsqrtf(variance + epsilon);
    __syncthreads();
    
    // Apply normalization
    for (int i = tid; i < dim; i += blockDim.x) {
        float normalized = (batch_input[i] - mean) * inv_std;
        batch_output[i] = normalized * gamma[i] + beta[i];
    }
}

// ============================================================================
// Embedding Lookup
// ============================================================================

__global__ void embedding_lookup_f32(const int* indices, const float* weights,
                                     float* output, int batch_size, int seq_len,
                                     int vocab_size, int embed_dim) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int embed_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || embed_idx >= embed_dim) return;
    
    int token_idx = indices[batch_idx * seq_len + seq_idx];
    if (token_idx >= 0 && token_idx < vocab_size) {
        int output_offset = (batch_idx * seq_len + seq_idx) * embed_dim + embed_idx;
        int weight_offset = token_idx * embed_dim + embed_idx;
        output[output_offset] = weights[weight_offset];
    }
}

// ============================================================================
// Matrix Operations (using cuBLAS for optimal performance)
// ============================================================================

// Wrapper for cuBLAS GEMM
extern "C" void cuda_gemm_f32(cublasHandle_t handle,
                              cublasOperation_t transa, cublasOperation_t transb,
                              int m, int n, int k,
                              const float* alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              const float* beta,
                              float* C, int ldc) {
    cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// ============================================================================
// Convolution (using cuDNN for optimal performance)
// ============================================================================

// Wrapper for cuDNN convolution
extern "C" cudnnStatus_t cuda_conv2d_f32(cudnnHandle_t handle,
                                         const cudnnTensorDescriptor_t input_desc,
                                         const float* input,
                                         const cudnnFilterDescriptor_t filter_desc,
                                         const float* filter,
                                         const cudnnConvolutionDescriptor_t conv_desc,
                                         const cudnnTensorDescriptor_t output_desc,
                                         float* output,
                                         cudnnConvolutionFwdAlgo_t algo,
                                         void* workspace,
                                         size_t workspace_size) {
    const float alpha = 1.0f, beta = 0.0f;
    return cudnnConvolutionForward(handle, &alpha, input_desc, input,
                                  filter_desc, filter, conv_desc, algo,
                                  workspace, workspace_size, &beta,
                                  output_desc, output);
}

// ============================================================================
// Attention Mechanism (simplified)
// ============================================================================

__global__ void scaled_dot_product_attention_f32(const float* query, const float* key, const float* value,
                                                 float* output, float* attention_weights,
                                                 int batch_size, int seq_len, int head_dim,
                                                 float scale) {
    int batch_idx = blockIdx.x;
    int seq_i = blockIdx.y;
    int seq_j = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_i >= seq_len || seq_j >= seq_len) return;
    
    // Compute attention score: Q * K^T
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        int q_idx = batch_idx * seq_len * head_dim + seq_i * head_dim + d;
        int k_idx = batch_idx * seq_len * head_dim + seq_j * head_dim + d;
        score += query[q_idx] * key[k_idx];
    }
    
    score *= scale;
    
    // Store attention weight (will be softmaxed later)
    int attn_idx = batch_idx * seq_len * seq_len + seq_i * seq_len + seq_j;
    attention_weights[attn_idx] = score;
}

} // extern "C"
