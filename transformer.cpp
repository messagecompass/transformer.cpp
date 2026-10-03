// Single-file educational transformer: trains a tiny GPT model from scratch in C++ with zero external dependencies.
// Uses a dynamic corpus text, automatic tokenization, GeLU activation, and learning rate warmup/decay.

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <fstream>

// ============================================================================
// HYPERPARAMETERS & CONFIGURATION (Modify these easily to experiment!)
// ============================================================================
// Transformer Architecture Specifications:
// - Context Length (T): Maximum sequence length the model can process at once.
// - Embedding Dimension (D_MODEL): Dimension of token & positional embeddings.
// - Multi-Head Attention: Splits D_MODEL into N_HEADS of size D_HEAD (D_MODEL = N_HEADS * D_HEAD).
// - Feed-Forward Expansion (D_MLP): Standard GPT convention expands D_MODEL by 4x in the MLP block.

// Model Architecture
const int   MAX_VOCAB      = 4900;
const int   Context_LEN    = 128;      
const int   N_LAYERS       = 4;
const int   D_MODEL        = 128;      
const int   N_HEADS        = 4;
const int   D_HEAD         = D_MODEL / N_HEADS;   
const int   D_MLP          = D_MODEL * 4;          

// Training Settings
const int   N_STEPS        = 12000;
const int   WARMUP_STEPS   = 1000;
const float MAX_LR         = 0.001f;
const float MIN_LR         = 0.0001f;
const float EARLY_STOP_LOSS = 2.0f;

// Generation & File Settings
const int   MAX_NEW_TOKENS = 100;
const char* TRAINING_FILE  = "TinyStories-valid.txt";//"aesop_fables.txt";//
const char* MODEL_FILE     = "tinystories_model.bin";//"aesop_fables.bin"; //

// ============================================================================
// MODEL PARAMETERS & GRADIENT BUFFERS
// ============================================================================

int VOCAB = 0;
char VOCAB_WORDS[MAX_VOCAB][32];

// ----------------------------------------------------------------------------
// Model Weights (Parameters)
// ----------------------------------------------------------------------------
// 1. Token Embeddings: Maps discrete token IDs to continuous vectors of size D_MODEL.
float token_embedding[MAX_VOCAB][D_MODEL];

// 2. Multi-Head Attention Weights (per layer):
//    - W_Q, W_K, W_V: Linear projections to construct Query, Key, and Value vectors.
//    - W_O: Output projection weight to mix information across all attention heads.
float W_Q[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float W_K[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float W_V[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float W_O[N_LAYERS][D_MODEL][D_MODEL];

// 3. Feed-Forward / MLP Weights (per layer):
//    - W_mlp1: Projects D_MODEL -> D_MLP (expansion layer).
//    - W_mlp2: Projects D_MLP -> D_MODEL (contraction layer).
float W_mlp1[N_LAYERS][D_MLP][D_MODEL];
float W_mlp2[N_LAYERS][D_MODEL][D_MLP];

// 4. Layer Normalization Parameters (Gain/Scale & Bias):
//    - Pre-Attention LayerNorm parameters (ln_attn)
//    - Pre-MLP LayerNorm parameters (ln_mlp)
//    - Final LayerNorm parameters before logits projection (ln_final)
float ln_attn_scale[N_LAYERS][D_MODEL];
float ln_attn_bias [N_LAYERS][D_MODEL];
float ln_mlp_scale [N_LAYERS][D_MODEL];
float ln_mlp_bias  [N_LAYERS][D_MODEL];
float ln_final_scale[D_MODEL];
float ln_final_bias [D_MODEL];

// 5. Unembedding Head:
//    - Maps final hidden state D_MODEL back to vocabulary size to produce logits.
float W_unembed[MAX_VOCAB][D_MODEL];

// ----------------------------------------------------------------------------
// Parameter Gradients (Accumulate gradients computed during backward_pass)
// ----------------------------------------------------------------------------
float g_token_embedding[MAX_VOCAB][D_MODEL];
float g_W_Q[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float g_W_K[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float g_W_V[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float g_W_O[N_LAYERS][D_MODEL][D_MODEL];
float g_W_mlp1[N_LAYERS][D_MLP][D_MODEL];
float g_W_mlp2[N_LAYERS][D_MODEL][D_MLP];
float g_ln_attn_scale[N_LAYERS][D_MODEL];
float g_ln_attn_bias [N_LAYERS][D_MODEL];
float g_ln_mlp_scale[N_LAYERS][D_MODEL];
float g_ln_mlp_bias  [N_LAYERS][D_MODEL];
float g_ln_final_scale[D_MODEL];
float g_ln_final_bias  [D_MODEL];
float g_W_unembed[MAX_VOCAB][D_MODEL];

// 6. Learned Positional Embeddings:
//    - Self-attention is permutation-invariant. Positional embeddings encode order.
float pos_embedding[Context_LEN][D_MODEL];
float g_pos_embedding[Context_LEN][D_MODEL];

// ----------------------------------------------------------------------------
// Intermediate Activation Saved Memory (Saved during Forward for Backward Pass)
// ----------------------------------------------------------------------------
float s_xhat_attn[N_LAYERS][Context_LEN][D_MODEL];
float s_inv_std_attn[N_LAYERS][Context_LEN];
float s_normed_attn[N_LAYERS][Context_LEN][D_MODEL];
float s_q[N_LAYERS][N_HEADS][Context_LEN][D_HEAD];
float s_k[N_LAYERS][N_HEADS][Context_LEN][D_HEAD];
float s_v[N_LAYERS][N_HEADS][Context_LEN][D_HEAD];
float s_attn_w[N_LAYERS][N_HEADS][Context_LEN][Context_LEN];
float s_attn_out[N_LAYERS][Context_LEN][D_MODEL];
float s_xhat_mlp[N_LAYERS][Context_LEN][D_MODEL];
float s_inv_std_mlp[N_LAYERS][Context_LEN];
float s_normed_mlp[N_LAYERS][Context_LEN][D_MODEL];
float s_mlp_pre_relu[N_LAYERS][Context_LEN][D_MLP];
float s_xhat_final[Context_LEN][D_MODEL];
float s_inv_std_final[Context_LEN];
float s_normed_final[Context_LEN][D_MODEL];

// Residual Stream & Logit Buffers
float residual[Context_LEN][D_MODEL];
float logits  [Context_LEN][MAX_VOCAB];
float d_residual[Context_LEN][D_MODEL];
float d_logits  [Context_LEN][MAX_VOCAB];
float d_attn_out_buf[Context_LEN][D_MODEL];
float d_normed_buf  [Context_LEN][D_MODEL];
float d_q[N_HEADS][Context_LEN][D_HEAD];
float d_k[N_HEADS][Context_LEN][D_HEAD];
float d_v[N_HEADS][Context_LEN][D_HEAD];

// ============================================================================
// ADAMW OPTIMIZER BUFFERS (1st and 2nd moments)
// ============================================================================
// AdamW keeps track of running averages of past gradients (m) and past squared
// gradients (v) for adaptive per-parameter learning rates with decoupled weight decay.
float m_token_embedding[MAX_VOCAB][D_MODEL], v_token_embedding[MAX_VOCAB][D_MODEL];
float m_W_Q[N_LAYERS][N_HEADS][D_HEAD][D_MODEL], v_W_Q[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float m_W_K[N_LAYERS][N_HEADS][D_HEAD][D_MODEL], v_W_K[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float m_W_V[N_LAYERS][N_HEADS][D_HEAD][D_MODEL], v_W_V[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float m_W_O[N_LAYERS][D_MODEL][D_MODEL], v_W_O[N_LAYERS][D_MODEL][D_MODEL];
float m_W_mlp1[N_LAYERS][D_MLP][D_MODEL], v_W_mlp1[N_LAYERS][D_MLP][D_MODEL];
float m_W_mlp2[N_LAYERS][D_MODEL][D_MLP], v_W_mlp2[N_LAYERS][D_MODEL][D_MLP];
float m_ln_attn_scale[N_LAYERS][D_MODEL], v_ln_attn_scale[N_LAYERS][D_MODEL];
float m_ln_attn_bias [N_LAYERS][D_MODEL], v_ln_attn_bias [N_LAYERS][D_MODEL];
float m_ln_mlp_scale [N_LAYERS][D_MODEL], v_ln_mlp_scale [N_LAYERS][D_MODEL];
float m_ln_mlp_bias  [N_LAYERS][D_MODEL], v_ln_mlp_bias  [N_LAYERS][D_MODEL];
float m_ln_final_scale[D_MODEL], v_ln_final_scale[D_MODEL];
float m_ln_final_bias [D_MODEL], v_ln_final_bias [D_MODEL];
float m_W_unembed[MAX_VOCAB][D_MODEL], v_W_unembed[MAX_VOCAB][D_MODEL];
float m_pos_embedding[Context_LEN][D_MODEL], v_pos_embedding[Context_LEN][D_MODEL];

// ----------------------------------------------------------------------------
// AdamW Optimization Step
// ----------------------------------------------------------------------------
// Computes decoupled weight decay, updates 1st (m) & 2nd (v) moments with bias
// correction, and updates weight array w.
void adamw_step(float lr, int t_step, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weight_decay = 0.01f)
{
    // Bias correction factors to compensate for initial zero values in m and v
    float bias_correction1 = 1.0f - powf(beta1, t_step + 1);
    float bias_correction2 = 1.0f - powf(beta2, t_step + 1);

    #define ADAMW_UPDATE(w, gw, m, v, size) \
        for (int i = 0; i < (size); i++) { \
            float* w_ptr  = ((float*)(w)) + i; \
            float* g_ptr  = ((float*)(gw)) + i; \
            float* m_ptr  = ((float*)(m)) + i; \
            float* v_ptr  = ((float*)(v)) + i; \
            /* Decoupled Weight Decay */ \
            *w_ptr -= lr * weight_decay * (*w_ptr); \
            /* Update 1st and 2nd moment estimates */ \
            *m_ptr = beta1 * (*m_ptr) + (1.0f - beta1) * (*g_ptr); \
            *v_ptr = beta2 * (*v_ptr) + (1.0f - beta2) * (*g_ptr) * (*g_ptr); \
            /* Bias-corrected moment estimates */ \
            float m_hat = *m_ptr / bias_correction1; \
            float v_hat = *v_ptr / bias_correction2; \
            /* Parameter Update */ \
            *w_ptr -= lr * m_hat / (sqrtf(v_hat) + eps); \
        }

    ADAMW_UPDATE(token_embedding, g_token_embedding, m_token_embedding, v_token_embedding, VOCAB * D_MODEL);
    ADAMW_UPDATE(pos_embedding, g_pos_embedding, m_pos_embedding, v_pos_embedding, Context_LEN * D_MODEL);
    ADAMW_UPDATE(W_Q, g_W_Q, m_W_Q, v_W_Q, N_LAYERS * N_HEADS * D_HEAD * D_MODEL);
    ADAMW_UPDATE(W_K, g_W_K, m_W_K, v_W_K, N_LAYERS * N_HEADS * D_HEAD * D_MODEL);
    ADAMW_UPDATE(W_V, g_W_V, m_W_V, v_W_V, N_LAYERS * N_HEADS * D_HEAD * D_MODEL);
    ADAMW_UPDATE(W_O, g_W_O, m_W_O, v_W_O, N_LAYERS * D_MODEL * D_MODEL);
    ADAMW_UPDATE(W_mlp1, g_W_mlp1, m_W_mlp1, v_W_mlp1, N_LAYERS * D_MLP * D_MODEL);
    ADAMW_UPDATE(W_mlp2, g_W_mlp2, m_W_mlp2, v_W_mlp2, N_LAYERS * D_MODEL * D_MLP);
    ADAMW_UPDATE(ln_attn_scale, g_ln_attn_scale, m_ln_attn_scale, v_ln_attn_scale, N_LAYERS * D_MODEL);
    ADAMW_UPDATE(ln_attn_bias,  g_ln_attn_bias,  m_ln_attn_bias,  v_ln_attn_bias,  N_LAYERS * D_MODEL);
    ADAMW_UPDATE(ln_mlp_scale,  g_ln_mlp_scale,  m_ln_mlp_scale,  v_ln_mlp_scale,  N_LAYERS * D_MODEL);
    ADAMW_UPDATE(ln_mlp_bias,   g_ln_mlp_bias,   m_ln_mlp_bias,   v_ln_mlp_bias,   N_LAYERS * D_MODEL);
    ADAMW_UPDATE(ln_final_scale, g_ln_final_scale, m_ln_final_scale, v_ln_final_scale, D_MODEL);
    ADAMW_UPDATE(ln_final_bias,  g_ln_final_bias,  m_ln_final_bias,  v_ln_final_bias,  D_MODEL);
    ADAMW_UPDATE(W_unembed, g_W_unembed, m_W_unembed, v_W_unembed, VOCAB * D_MODEL);

    #undef ADAMW_UPDATE
}

std::mt19937 rng(42);

// ============================================================================
// ACTIVATION FUNCTIONS (GeLU - Gaussian Error Linear Unit)
// ============================================================================
// GeLU provides non-linearity: gelu(x) = x * Phi(x)
// Used in GPT models instead of ReLU as it provides smoother gradients around 0.
inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x)));
}

// Exact derivative of the GeLU approximation for backpropagation.
inline float gelu_grad(float x) {
    const float kAlpha = 0.7978845608f;
    const float kBeta  = 0.044715f;

    float x2 = x * x;
    float tanh_arg = kAlpha * (x + kBeta * x2);
    float tanh_val = tanhf(tanh_arg);

    float left = 0.5f * (1.0f + tanh_val);
    float right = 0.5f * x * (1.0f - tanh_val * tanh_val) *
                  kAlpha * (1.0f + 2.0f * kBeta * x);

    return left + right;
}

// ----------------------------------------------------------------------------
// Learning Rate Schedule
// ----------------------------------------------------------------------------
// Combines linear warmup for stability at start with cosine decay for convergence.
float get_lr(int step, int total_steps) {
    if (step < WARMUP_STEPS) {
        return MAX_LR * ((float)step / WARMUP_STEPS);
    }
    float progress = (float)(step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS);
    return MIN_LR + 0.5f * (MAX_LR - MIN_LR) * (1.0f + cosf(progress * 3.14159265f));
}

// ----------------------------------------------------------------------------
// Parameter Initialization
// ----------------------------------------------------------------------------
// Initializes embeddings and weight projections with uniform random values.
// Scale/Gain parameters for LayerNorm are initialized to 1.0, biases to 0.0.
void initialize_weights()
{
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    auto fill_random = [&](float* ptr, int size) { for (int i = 0; i < size; i++) ptr[i] = dis(rng); };
    auto fill_constant = [&](float* ptr, int size, float val) { for (int i = 0; i < size; i++) ptr[i] = val; };

    fill_random(&token_embedding[0][0], VOCAB * D_MODEL);
    fill_random(&pos_embedding[0][0], Context_LEN * D_MODEL);

    for (int layer = 0; layer < N_LAYERS; layer++) {
        for (int h = 0; h < N_HEADS; h++) {
            fill_random(&W_Q[layer][h][0][0], D_HEAD * D_MODEL);
            fill_random(&W_K[layer][h][0][0], D_HEAD * D_MODEL);
            fill_random(&W_V[layer][h][0][0], D_HEAD * D_MODEL);
        }
        fill_random(&W_O[layer][0][0], D_MODEL * D_MODEL);
        fill_random(&W_mlp1[layer][0][0], D_MLP * D_MODEL);
        fill_random(&W_mlp2[layer][0][0], D_MODEL * D_MLP);


        fill_constant(&ln_attn_scale[layer][0], D_MODEL, 1.0f);
        fill_constant(&ln_attn_bias [layer][0], D_MODEL, 0.0f);
        fill_constant(&ln_mlp_scale [layer][0], D_MODEL, 1.0f);
        fill_constant(&ln_mlp_bias  [layer][0], D_MODEL, 0.0f);
    }
    fill_constant(ln_final_scale, D_MODEL, 1.0f);
    fill_constant(ln_final_bias,  D_MODEL, 0.0f);
    fill_random(&W_unembed[0][0], VOCAB * D_MODEL);

    printf("Dynamic model weights initialized (Vocab Size = %d, D_MODEL = %d).\n", VOCAB, D_MODEL);
}

// ============================================================================
// LAYER NORMALIZATION (LayerNorm)
// ============================================================================
// Standardizes feature activations across D_MODEL per token position:
// xhat = (x - mean) / sqrt(var + eps)
// out  = xhat * scale + bias
void layer_norm_forward(const float* x, const float* scale, const float* bias,
                        float* out, float* xhat, float* inv_std_out)
{
    float mean = 0.0f;
    for (int i = 0; i < D_MODEL; i++) mean += x[i];
    mean /= D_MODEL;

    float var = 0.0f;
    for (int i = 0; i < D_MODEL; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= D_MODEL;

    float is = 1.0f / sqrtf(var + 1e-5f);
    *inv_std_out = is;

    for (int i = 0; i < D_MODEL; i++) {
        xhat[i] = (x[i] - mean) * is;
        out[i] = xhat[i] * scale[i] + bias[i];
    }
}

// Backward pass for LayerNorm: calculates gradients w.r.t input (dx) and scale/bias parameters.
void layer_norm_backward(const float* dy, const float* xhat, float inv_std,
                         const float* scale, float* dx,
                         float* d_scale, float* d_bias)
{
    for (int i = 0; i < D_MODEL; i++) {
        d_scale[i] += dy[i] * xhat[i];
        d_bias[i]  += dy[i];
    }
    float d_xhat[D_MODEL];
    for (int i = 0; i < D_MODEL; i++) d_xhat[i] = dy[i] * scale[i];

    float mean_dxhat = 0.0f;
    for (int i = 0; i < D_MODEL; i++) mean_dxhat += d_xhat[i];
    mean_dxhat /= D_MODEL;

    float mean_dxhat_xhat = 0.0f;
    for (int i = 0; i < D_MODEL; i++) mean_dxhat_xhat += d_xhat[i] * xhat[i];
    mean_dxhat_xhat /= D_MODEL;

    for (int i = 0; i < D_MODEL; i++)
        dx[i] = inv_std * (d_xhat[i] - mean_dxhat - xhat[i] * mean_dxhat_xhat);
}

// Reset gradient arrays before calculating backpropagation for a batch.
void zero_gradients()
{
    memset(g_token_embedding, 0, sizeof(g_token_embedding));
    memset(g_pos_embedding, 0, sizeof(g_pos_embedding));
    memset(g_W_Q,  0, sizeof(g_W_Q));
    memset(g_W_K,  0, sizeof(g_W_K));
    memset(g_W_V,  0, sizeof(g_W_V));
    memset(g_W_O,  0, sizeof(g_W_O));
    memset(g_W_mlp1, 0, sizeof(g_W_mlp1));
    memset(g_W_mlp2, 0, sizeof(g_W_mlp2));
    memset(g_ln_attn_scale, 0, sizeof(g_ln_attn_scale));
    memset(g_ln_attn_bias,  0, sizeof(g_ln_attn_bias));
    memset(g_ln_mlp_scale, 0, sizeof(g_ln_mlp_scale));
    memset(g_ln_mlp_bias,   0, sizeof(g_ln_mlp_bias));
    memset(g_ln_final_scale, 0, sizeof(g_ln_final_scale));
    memset(g_ln_final_bias,  0, sizeof(g_ln_final_bias));
    memset(g_W_unembed, 0, sizeof(g_W_unembed));
}

// ============================================================================
// FORWARD PASS
// ============================================================================
// Executes full forward pass:
// 1. Embedding lookup (Token + Positional embeddings).
// 2. Transformer layers (Pre-LN -> Multi-Head Causal Self-Attention -> Residual -> Pre-LN -> MLP -> Residual).
// 3. Final LayerNorm & Unembedding projection to compute vocabulary logits.
// 4. Categorical Cross-Entropy Loss over sequence.
float forward_pass(int* tokens, int* targets, int seq_len)
{
    // STEP 1: EMBEDDINGS (Token Embedding + Learned Positional Embedding)
    for (int pos = 0; pos < seq_len; pos++)
        for (int d = 0; d < D_MODEL; d++)
            residual[pos][d] = token_embedding[tokens[pos]][d] + pos_embedding[pos][d];

    // Scaling factor 1 / sqrt(d_k) for Scaled Dot-Product Attention
    float scale = 1.0f / sqrtf((float)D_HEAD);

    // STEP 2: TRANSFORMER BLOCK STACK
    for (int layer = 0; layer < N_LAYERS; layer++)
    {
        // 2a. Pre-Layer Normalization for Attention Block
        for (int pos = 0; pos < seq_len; pos++)
            layer_norm_forward(residual[pos], ln_attn_scale[layer], ln_attn_bias[layer],
                             s_normed_attn[layer][pos], s_xhat_attn[layer][pos], &s_inv_std_attn[layer][pos]);

        // 2b. Compute Query, Key, and Value Projections across all Attention Heads
        //     Q = Normed * W_Q, K = Normed * W_K, V = Normed * W_V
        for (int h = 0; h < N_HEADS; h++) {
            int offset = h * D_HEAD;
            for (int pos = 0; pos < seq_len; pos++) {
                for (int dh = 0; dh < D_HEAD; dh++) {
                    float qv = 0, kv = 0, vv = 0;
                    const float* norm_ptr = s_normed_attn[layer][pos];
                    for (int d = 0; d < D_MODEL; d++) {
                        qv += W_Q[layer][h][dh][d] * norm_ptr[d];
                        kv += W_K[layer][h][dh][d] * norm_ptr[d];
                        vv += W_V[layer][h][dh][d] * norm_ptr[d];
                    }
                    s_q[layer][h][pos][dh] = qv;
                    s_k[layer][h][pos][dh] = kv;
                    s_v[layer][h][pos][dh] = vv;
                }
            }
        }

        for (int pos = 0; pos < seq_len; pos++)
            for (int d = 0; d < D_MODEL; d++)
                s_attn_out[layer][pos][d] = 0.0f;

        // 2c. Scaled Dot-Product Causal Self-Attention
        //     Attention(Q,K,V) = softmax( (Q * K^T) / sqrt(d_k) + CausalMask ) * V
        for (int h = 0; h < N_HEADS; h++) {
            int offset = h * D_HEAD;
            for (int i = 0; i < seq_len; i++) {
                float max_val = -1e30f;
                // Causal Mask: Query position 'i' can only look at Key positions 'j' <= 'i'
                for (int j = 0; j <= i; j++) {
                    float dot = 0.0f;
                    const float* q_ptr = s_q[layer][h][i];
                    const float* k_ptr = s_k[layer][h][j];
                    for (int dh = 0; dh < D_HEAD; dh++)
                        dot += q_ptr[dh] * k_ptr[dh];
                    float val = dot * scale;
                    s_attn_w[layer][h][i][j] = val;
                    if (val > max_val) max_val = val;
                }
                // Mask out future positions (j > i) with negative infinity
                for (int j = i + 1; j < seq_len; j++)
                    s_attn_w[layer][h][i][j] = -1e30f;

                // Softmax normalization over row i
                float sum = 0.0f;
                for (int j = 0; j <= i; j++) {
                    float ev = expf(s_attn_w[layer][h][i][j] - max_val);
                    s_attn_w[layer][h][i][j] = ev;
                    sum += ev;
                }
                float inv_sum = 1.0f / sum;
                for (int j = 0; j <= i; j++) {
                    s_attn_w[layer][h][i][j] *= inv_sum;
                }

                // Weighted sum over Value vectors
                for (int j = 0; j <= i; j++) {
                    float weight = s_attn_w[layer][h][i][j];
                    const float* v_ptr = s_v[layer][h][j];
                    float* out_ptr = &s_attn_out[layer][i][offset];
                    for (int dh = 0; dh < D_HEAD; dh++)
                        out_ptr[dh] += weight * v_ptr[dh];
                }
            }
        }

        // 2d. Output Projection & Residual Connection (Attn Sub-layer)
        //     residual = residual + W_O * attn_out
        for (int pos = 0; pos < seq_len; pos++) {
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                const float* attn_out_ptr = s_attn_out[layer][pos];
                const float* w_o_ptr = W_O[layer][d];
                for (int d2 = 0; d2 < D_MODEL; d2++)
                    val += w_o_ptr[d2] * attn_out_ptr[d2];
                residual[pos][d] += val;
            }
        }

        // 2e. Pre-Layer Normalization for MLP Block
        for (int pos = 0; pos < seq_len; pos++)
            layer_norm_forward(residual[pos], ln_mlp_scale[layer], ln_mlp_bias[layer],
                             s_normed_mlp[layer][pos], s_xhat_mlp[layer][pos], &s_inv_std_mlp[layer][pos]);

        // 2f. Feed-Forward Network (MLP) & Residual Connection
        //     MLP(x) = W_mlp2 * GeLU(W_mlp1 * x)
        for (int pos = 0; pos < seq_len; pos++) {
            // Expansion layer: D_MODEL -> D_MLP
            for (int m = 0; m < D_MLP; m++) {
                float val = 0.0f;
                const float* norm_ptr = s_normed_mlp[layer][pos];
                const float* w_mlp1_ptr = W_mlp1[layer][m];
                for (int d = 0; d < D_MODEL; d++)
                    val += w_mlp1_ptr[d] * norm_ptr[d];
                s_mlp_pre_relu[layer][pos][m] = val; 
            }
            // Non-linear activation (GeLU) + Contraction layer (D_MLP -> D_MODEL) + Residual Add
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                const float* w_mlp2_ptr = W_mlp2[layer][d];
                for (int m = 0; m < D_MLP; m++) {
                    float gelu_out = gelu(s_mlp_pre_relu[layer][pos][m]);
                    val += w_mlp2_ptr[m] * gelu_out;
                }
                residual[pos][d] += val;

            }

        }
    }

    // STEP 3: FINAL LAYERNORM & UNEMBEDDING HEAD (LOGITS)
    for (int pos = 0; pos < seq_len; pos++) {
        layer_norm_forward(residual[pos], ln_final_scale, ln_final_bias,
                         s_normed_final[pos], s_xhat_final[pos], &s_inv_std_final[pos]);

        for (int voc = 0; voc < VOCAB; voc++) {
            float val = 0.0f;
            const float* unembed_ptr = W_unembed[voc];
            const float* final_ptr = s_normed_final[pos];
            for (int d = 0; d < D_MODEL; d++)
                val += unembed_ptr[d] * final_ptr[d];
            logits[pos][voc] = val;
        }
    }

    // STEP 4: CATEGORICAL CROSS-ENTROPY LOSS COMPUTATION
    // Loss = -log( softmax(logits)[target_token] )
    float loss = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
        float max_val = logits[pos][0];
        for (int v = 1; v < VOCAB; v++)
            if (logits[pos][v] > max_val) max_val = logits[pos][v];
        float sum_exp = 0.0f;
        for (int v = 0; v < VOCAB; v++) sum_exp += expf(logits[pos][v] - max_val);
        loss -= (logits[pos][targets[pos]] - (max_val + logf(sum_exp)));
    }
    return loss / seq_len;
}

// ============================================================================
// BACKWARD PASS (Backpropagation Through Time / Reverse Computational Graph)
// ============================================================================
// Calculates exact parameter gradients using chain rule from Loss -> Logits ->
// Unembed -> Final LayerNorm -> Layers (MLP -> Attn) -> Embeddings.
void backward_pass(int* tokens, int* targets, int seq_len)
{
    float scale = 1.0f / sqrtf((float)D_HEAD);

    // STEP 1: SOFTMAX & CROSS-ENTROPY LOSS GRADIENTS w.r.t LOGITS
    // dL/d(logit_v) = p_v - y_v
    for (int pos = 0; pos < seq_len; pos++) {
        float max_val = logits[pos][0];
        for (int v = 1; v < VOCAB; v++)
            if (logits[pos][v] > max_val) max_val = logits[pos][v];
        float sum_exp = 0.0f;
        for (int v = 0; v < VOCAB; v++) {
            d_logits[pos][v] = expf(logits[pos][v] - max_val);
            sum_exp += d_logits[pos][v];
        }
        for (int v = 0; v < VOCAB; v++) {
            d_logits[pos][v] /= sum_exp;
            if (v == targets[pos]) d_logits[pos][v] -= 1.0f;
            d_logits[pos][v] /= seq_len;
        }
    }

    // STEP 2: UNEMBEDDING HEAD BACKWARD PASS
    float d_normed_final[Context_LEN][D_MODEL];
    memset(d_normed_final, 0, sizeof(d_normed_final));
    for (int pos = 0; pos < seq_len; pos++)
        for (int v = 0; v < VOCAB; v++) {
            for (int d = 0; d < D_MODEL; d++) {
                d_normed_final[pos][d] += d_logits[pos][v] * W_unembed[v][d];
                g_W_unembed[v][d] += d_logits[pos][v] * s_normed_final[pos][d];
            }
        }

    // STEP 3: FINAL LAYERNORM BACKWARD PASS
    memset(d_residual, 0, sizeof(d_residual));
    for (int pos = 0; pos < seq_len; pos++)
        layer_norm_backward(d_normed_final[pos], s_xhat_final[pos], s_inv_std_final[pos],
                          ln_final_scale, d_residual[pos], g_ln_final_scale, g_ln_final_bias);

    // STEP 4: TRANSFORMER LAYERS BACKWARD (Reverse Order: N_LAYERS-1 down to 0)
    for (int layer = N_LAYERS - 1; layer >= 0; layer--)
    {
        // 4a. MLP Sub-layer Backward
        for (int pos = 0; pos < seq_len; pos++) {
            // Gradient through W_mlp2
            float d_gelu_out[D_MLP];
            for (int m = 0; m < D_MLP; m++) {
                float val = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    val += d_residual[pos][d] * W_mlp2[layer][d][m];
                d_gelu_out[m] = val;
            }

            for (int d = 0; d < D_MODEL; d++)
                for (int m = 0; m < D_MLP; m++) {
                    float gelu_val = gelu(s_mlp_pre_relu[layer][pos][m]);
                    g_W_mlp2[layer][d][m] += d_residual[pos][d] * gelu_val;
                }

            // Gradient through GeLU non-linearity
            float d_pre_gelu[D_MLP];
            for (int m = 0; m < D_MLP; m++)
                d_pre_gelu[m] = d_gelu_out[m] * gelu_grad(s_mlp_pre_relu[layer][pos][m]);

            // Gradient through W_mlp1
            float d_normed_mlp_pos[D_MODEL];
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                for (int m = 0; m < D_MLP; m++)
                    val += d_pre_gelu[m] * W_mlp1[layer][m][d];
                d_normed_mlp_pos[d] = val;
            }

            for (int m = 0; m < D_MLP; m++)
                for (int d = 0; d < D_MODEL; d++)
                    g_W_mlp1[layer][m][d] += d_pre_gelu[m] * s_normed_mlp[layer][pos][d];

            // Gradient through Pre-MLP LayerNorm
            float d_res_from_mlp_ln[D_MODEL];
            layer_norm_backward(d_normed_mlp_pos, s_xhat_mlp[layer][pos], s_inv_std_mlp[layer][pos],
                              ln_mlp_scale[layer], d_res_from_mlp_ln, g_ln_mlp_scale[layer], g_ln_mlp_bias[layer]);

            for (int d = 0; d < D_MODEL; d++) d_residual[pos][d] += d_res_from_mlp_ln[d];
        }

        // 4b. Multi-Head Attention Output Projection (W_O) Backward
        memset(d_attn_out_buf, 0, sizeof(d_attn_out_buf));
        for (int pos = 0; pos < seq_len; pos++) {
            for (int d2 = 0; d2 < D_MODEL; d2++) {
                float val = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    val += d_residual[pos][d] * W_O[layer][d][d2];
                d_attn_out_buf[pos][d2] = val;
            }
            for (int d = 0; d < D_MODEL; d++)
                for (int d2 = 0; d2 < D_MODEL; d2++)
                    g_W_O[layer][d][d2] += d_residual[pos][d] * s_attn_out[layer][pos][d2];
        }

        memset(d_q, 0, sizeof(d_q));
        memset(d_k, 0, sizeof(d_k));
        memset(d_v, 0, sizeof(d_v));

        // 4c. Attention Weights & Softmax Backward
        for (int h = 0; h < N_HEADS; h++) {
            int offset = h * D_HEAD;
            for (int i = 0; i < seq_len; i++) {
                float d_attn_w[Context_LEN];
                memset(d_attn_w, 0, sizeof(d_attn_w));
                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        d_attn_w[j] += d_attn_out_buf[i][offset + dh] * s_v[layer][h][j][dh];
                        d_v[h][j][dh] += s_attn_w[layer][h][i][j] * d_attn_out_buf[i][offset + dh];
                    }

                float dot = 0.0f;
                for (int j = 0; j <= i; j++)
                    dot += d_attn_w[j] * s_attn_w[layer][h][i][j];

                // Backward pass through Softmax
                float d_score[Context_LEN];
                for (int j = 0; j <= i; j++)
                    d_score[j] = s_attn_w[layer][h][i][j] * (d_attn_w[j] - dot);

                // Gradients for Queries (Q) and Keys (K)
                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        d_q[h][i][dh] += d_score[j] * scale * s_k[layer][h][j][dh];
                        d_k[h][j][dh] += d_score[j] * scale * s_q[layer][h][i][dh];
                    }
            }
        }

        // 4d. Linear Projections (W_Q, W_K, W_V) Backward
        memset(d_normed_buf, 0, sizeof(d_normed_buf));
        for (int h = 0; h < N_HEADS; h++)
            for (int pos = 0; pos < seq_len; pos++)
                for (int dh = 0; dh < D_HEAD; dh++)
                    for (int d = 0; d < D_MODEL; d++) {
                        d_normed_buf[pos][d] += d_q[h][pos][dh] * W_Q[layer][h][dh][d];
                        d_normed_buf[pos][d] += d_k[h][pos][dh] * W_K[layer][h][dh][d];
                        d_normed_buf[pos][d] += d_v[h][pos][dh] * W_V[layer][h][dh][d];
                        g_W_Q[layer][h][dh][d] += d_q[h][pos][dh] * s_normed_attn[layer][pos][d];
                        g_W_K[layer][h][dh][d] += d_k[h][pos][dh] * s_normed_attn[layer][pos][d];
                        g_W_V[layer][h][dh][d] += d_v[h][pos][dh] * s_normed_attn[layer][pos][d];
                    }

        // 4e. Pre-Attention LayerNorm Backward
        for (int pos = 0; pos < seq_len; pos++) {
            float d_res_from_attn_ln[D_MODEL];
            layer_norm_backward(d_normed_buf[pos], s_xhat_attn[layer][pos], s_inv_std_attn[layer][pos],
                              ln_attn_scale[layer], d_res_from_attn_ln, g_ln_attn_scale[layer], g_ln_attn_bias[layer]);
            for (int d = 0; d < D_MODEL; d++) d_residual[pos][d] += d_res_from_attn_ln[d];
        }
    }

    // STEP 5: EMBEDDINGS BACKWARD (Token & Positional Embeddings)
    for (int pos = 0; pos < seq_len; pos++)
        for (int d = 0; d < D_MODEL; d++)
        {
            g_token_embedding[tokens[pos]][d] += d_residual[pos][d];
            g_pos_embedding[pos][d] += d_residual[pos][d];
        }
}

// Simple Stochastic Gradient Descent (SGD) step alternative.
void sgd_step(float lr)
{
    #define UPDATE(w, gw, size) \
        for (int _i = 0; _i < (size); _i++) \
            ((float*)(w))[_i] -= lr * ((float*)(gw))[_i];

    UPDATE(token_embedding, g_token_embedding, VOCAB * D_MODEL);
    UPDATE(pos_embedding, g_pos_embedding, Context_LEN * D_MODEL);
    UPDATE(W_Q, g_W_Q, N_LAYERS * N_HEADS * D_HEAD * D_MODEL);
    UPDATE(W_K, g_W_K, N_LAYERS * N_HEADS * D_HEAD * D_MODEL);
    UPDATE(W_V, g_W_V, N_LAYERS * N_HEADS * D_HEAD * D_MODEL);
    UPDATE(W_O, g_W_O, N_LAYERS * D_MODEL * D_MODEL);
    UPDATE(W_mlp1, g_W_mlp1, N_LAYERS * D_MLP * D_MODEL);
    UPDATE(W_mlp2, g_W_mlp2, N_LAYERS * D_MODEL * D_MLP);
    UPDATE(ln_attn_scale, g_ln_attn_scale, N_LAYERS * D_MODEL);
    UPDATE(ln_attn_bias,  g_ln_attn_bias,  N_LAYERS * D_MODEL);
    UPDATE(ln_mlp_scale,  g_ln_mlp_scale,  N_LAYERS * D_MODEL);
    UPDATE(ln_mlp_bias,   g_ln_mlp_bias,   N_LAYERS * D_MODEL);
    UPDATE(ln_final_scale, g_ln_final_scale, D_MODEL);
    UPDATE(ln_final_bias,  g_ln_final_bias,  D_MODEL);
    UPDATE(W_unembed, g_W_unembed, VOCAB * D_MODEL);
    #undef UPDATE
}

// ----------------------------------------------------------------------------
// Tokenization & Parsing Helper
// ----------------------------------------------------------------------------
// Converts a raw text prompt into a vector of token IDs based on vocabulary map.
std::vector<int> prompt_to_vector(const std::string& prompt_str, const std::map<std::string, int>& word_to_id)
{
    std::vector<int> prompt_ids;
    std::stringstream ss(prompt_str);
    std::string raw_word;

    while (ss >> raw_word) {
        std::string current_word = "";
        for (char c : raw_word) {
            if (ispunct(c) && c != '\'' && c != '-') {
                if (!current_word.empty()) {
                    if (word_to_id.find(current_word) != word_to_id.end()) {
                        prompt_ids.push_back(word_to_id.at(current_word));
                    }
                    current_word = "";
                }
                std::string p(1, c);
                if (word_to_id.find(p) != word_to_id.end()) {
                    prompt_ids.push_back(word_to_id.at(p));
                }
            } else {
                current_word += tolower(c);
            }
        }
        if (!current_word.empty()) {
            if (word_to_id.find(current_word) != word_to_id.end()) {
                prompt_ids.push_back(word_to_id.at(current_word));
            }
        }
    }
    return prompt_ids;
}

// Save trained model binary weights to disk.
void save_model(const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Could not open file '%s' for saving weights.\n", filename.c_str());
        return;
    }

    file.write((char*)token_embedding, sizeof(token_embedding));
    file.write((char*)pos_embedding, sizeof(pos_embedding));
    file.write((char*)W_Q, sizeof(W_Q));
    file.write((char*)W_K, sizeof(W_K));
    file.write((char*)W_V, sizeof(W_V));
    file.write((char*)W_O, sizeof(W_O));
    file.write((char*)W_mlp1, sizeof(W_mlp1));
    file.write((char*)W_mlp2, sizeof(W_mlp2));
    file.write((char*)ln_attn_scale, sizeof(ln_attn_scale));
    file.write((char*)ln_attn_bias,  sizeof(ln_attn_bias));
    file.write((char*)ln_mlp_scale,  sizeof(ln_mlp_scale));
    file.write((char*)ln_mlp_bias,   sizeof(ln_mlp_bias));
    file.write((char*)ln_final_scale, sizeof(ln_final_scale));
    file.write((char*)ln_final_bias,  sizeof(ln_final_bias));
    file.write((char*)W_unembed, sizeof(W_unembed));

    file.close();
    printf("Model successfully saved to binary file '%s'.\n", filename.c_str());
}

// Load trained model binary weights from disk.
void load_model(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Could not open file '%s' for loading weights.\n", filename.c_str());
        return;
    }

    file.read((char*)token_embedding, sizeof(token_embedding));
    file.read((char*)pos_embedding, sizeof(pos_embedding));
    file.read((char*)W_Q, sizeof(W_Q));
    file.read((char*)W_K, sizeof(W_K));
    file.read((char*)W_V, sizeof(W_V));
    file.read((char*)W_O, sizeof(W_O));
    file.read((char*)W_mlp1, sizeof(W_mlp1));
    file.read((char*)W_mlp2, sizeof(W_mlp2));
    file.read((char*)ln_attn_scale, sizeof(ln_attn_scale));
    file.read((char*)ln_attn_bias,  sizeof(ln_attn_bias));
    file.read((char*)ln_mlp_scale,  sizeof(ln_mlp_scale));
    file.read((char*)ln_mlp_bias,   sizeof(ln_mlp_bias));
    file.read((char*)ln_final_scale, sizeof(ln_final_scale));
    file.read((char*)ln_final_bias,  sizeof(ln_final_bias));
    file.read((char*)W_unembed, sizeof(W_unembed));

    file.close();
    printf("Model successfully loaded from binary file '%s'.\n", filename.c_str());
}

// ============================================================================
// INFERENCE & AUTOREGRESSIVE SAMPLING
// ============================================================================
// Generates text autoregressively (token by token) given a prompt.
// Incorporates 3 sampling techniques:
// 1. Repetition Penalty: Reduces probability of recently generated tokens.
// 2. Temperature Scaling: Controls randomness (lower = deterministic, higher = creative).
// 3. Top-K Sampling: Restricts sampling to the top-K most likely candidates.
void generate_words(const int* prompt_ids, int prompt_len, int max_new_tokens)
{
    std::vector<int> current_tokens;
    for (int i = 0; i < prompt_len && i < Context_LEN; i++) {
        current_tokens.push_back(prompt_ids[i]);
    }

    printf("\n--- Generating Words ---\nPrompt: ");
    for (int id : current_tokens) {
        printf("%s ", VOCAB_WORDS[id]);
    }
    printf("\nOutput: ");
    for (int id : current_tokens) {
        printf("%s ", VOCAB_WORDS[id]);
    }

    // Sampling parameters
    const float temperature = 0.7f; // Lower = more focused, Higher = more creative
    const int top_k = 20;           // Only sample from the top 20 most likely words
    const float penalty = 1.2f;     // Mild additive penalty

    for (int t = 0; t < max_new_tokens; t++) {
        int seq_len = (int)current_tokens.size();
        if (seq_len > Context_LEN) break;

        forward_pass(current_tokens.data(), current_tokens.data(), seq_len);
        int last_pos = seq_len - 1;

        // 1. Gentle Additive Repetition Penalty (Last 15 tokens only)
        int lookback = std::max(0, seq_len - 15);
        for (int i = lookback; i < seq_len; i++) {
            int recent_word = current_tokens[i];
            if (recent_word > 0 && recent_word < VOCAB) {
                logits[last_pos][recent_word] -= penalty;
            }
        }

        // 2. Temperature Scaling
        for (int v = 0; v < VOCAB; v++) {
            logits[last_pos][v] /= temperature;
        }

        // 3. Find Max Logit for Softmax Numerical Stability
        float max_logit = logits[last_pos][0];
        for (int v = 1; v < VOCAB; v++) {
            if (logits[last_pos][v] > max_logit) max_logit = logits[last_pos][v];
        }

        // 4. Compute Softmax Probabilities & Gather Top-K
        std::vector<std::pair<float, int>> probs(VOCAB);
        float sum_exp = 0.0f;
        for (int v = 0; v < VOCAB; v++) {
            float exp_val = expf(logits[last_pos][v] - max_logit);
            probs[v] = {exp_val, v};
            sum_exp += exp_val;
        }

        // Sort logits descending
        std::sort(probs.rbegin(), probs.rend());

        // 5. Sample from Top-K Candidates
        int actual_k = std::min(top_k, VOCAB);
        float top_k_sum = 0.0f;
        for (int i = 0; i < actual_k; i++) {
            top_k_sum += probs[i].first;
        }

        std::uniform_real_distribution<float> dist(0.0f, top_k_sum);
        float r = dist(rng);
        float accum = 0.0f;
        int chosen_token = probs[0].second;

        for (int i = 0; i < actual_k; i++) {
            accum += probs[i].first;
            if (r <= accum) {
                chosen_token = probs[i].second;
                break;
            }
        }

        std::string token_str = VOCAB_WORDS[chosen_token];
        if (token_str == "." || token_str == "," || token_str == "?" || token_str == "!" || token_str == ":" || token_str == ";")
        {
            printf("%s", token_str.c_str());
        }
        else
        {
            printf(" %s", token_str.c_str());
        }

        current_tokens.push_back(chosen_token);

        if (std::string(VOCAB_WORDS[chosen_token]) == "<|endoftext|>") {
            printf("\n[Reached end of text]\n");
            break;
        }
    }
    printf("\n------------------------\n");
}

// Generate vocabulary diagnostic report file.
void save_token_report(
    const std::string& model_file,
    const std::vector<std::pair<std::string, int>>& sorted_vocab,
    const std::map<std::string, int>& word_to_id)
{
    std::string report_file = model_file + std::string("_tokens.txt");

    std::ofstream out(report_file);
    if (!out.is_open()) {
        printf("Failed to create token report '%s'\n", report_file.c_str());
        return;
    }

    out << "TOKEN_ID\tFREQUENCY\tWORD\n";

    out << "0\tN/A\t<unk>\n";

    for (const auto& p : sorted_vocab) {
        auto it = word_to_id.find(p.first);
        if (it == word_to_id.end())
            continue;

        out << it->second
            << '\t'
            << p.second
            << '\t'
            << p.first
            << '\n';
    }

    out.close();
    printf("Token report saved to '%s'\n", report_file.c_str());
}

// ============================================================================
// TOKENIZATION & CORPUS PARSING
// ============================================================================
// Simple word-level + punctuation tokenizer:
// Reads raw text file, separates words and punctuation marks into vocabulary IDs,
// and maps training corpus into sequence of integer token IDs.
bool load_and_tokenize_corpus(const std::string& filename, std::vector<int>& training_word_ids, std::map<std::string, int>& word_to_id)
{
    std::string corpus = "";
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            corpus += line + " ";
        }
        file.close();
        printf("Successfully loaded training corpus from '%s'.\n", filename.c_str());
    } else {
        printf("'%s' not found. Falling back to default story corpus.\n", filename.c_str());
        corpus = "I walk down a wide road... <|endoftext|> Once upon a time...";
    }

    std::stringstream ss(corpus);
    std::string raw_token;
    std::vector<std::string> all_parsed_tokens;
    std::map<std::string, int> token_counts;

    while (ss >> raw_token) {
        if (raw_token == "<|endoftext|>") {
            all_parsed_tokens.push_back("<|endoftext|>");
            token_counts["<|endoftext|>"]++;
            continue;
        }

        std::string current_word = "";
        for (size_t i = 0; i < raw_token.size(); i++) {
            char c = raw_token[i];

            // Separate punctuation marks into standalone tokens
            if (ispunct(c) && c != '\'' && c != '-') { 
                if (!current_word.empty()) {
                    all_parsed_tokens.push_back(current_word);
                    token_counts[current_word]++;
                    current_word = "";
                }
                // Add punctuation as its own individual token
                std::string punct_token(1, c);
                all_parsed_tokens.push_back(punct_token);
                token_counts[punct_token]++;
            } else {
                current_word += tolower(c);
            }
        }
        if (!current_word.empty()) {
            all_parsed_tokens.push_back(current_word);
            token_counts[current_word]++;
        }
    }

    // Sort vocabulary by frequency
    std::vector<std::pair<std::string, int>> sorted_vocab(token_counts.begin(), token_counts.end());
    std::sort(sorted_vocab.begin(), sorted_vocab.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Token 0: <unk>
    word_to_id["<unk>"] = 0;
    snprintf(VOCAB_WORDS[0], sizeof(VOCAB_WORDS[0]), "%s", "<unk>");
    VOCAB = 1;

    // Token 1: <|endoftext|> (if present)
    bool has_endoftext = false;
    for (const auto& pair : sorted_vocab) {
        if (pair.first == "<|endoftext|>") {
            has_endoftext = true;
            break;
        }
    }

    if (has_endoftext && VOCAB < MAX_VOCAB) {
        word_to_id["<|endoftext|>"] = VOCAB;
        snprintf(VOCAB_WORDS[VOCAB], sizeof(VOCAB_WORDS[VOCAB]), "%s", "<|endoftext|>");
        VOCAB++;
    }

    // Assign Token IDs to words and punctuation marks
    for (const auto& pair : sorted_vocab) {
        if (pair.first == "<|endoftext|>") continue;

        if (VOCAB < MAX_VOCAB) {
            word_to_id[pair.first] = VOCAB;
            snprintf(VOCAB_WORDS[VOCAB], sizeof(VOCAB_WORDS[VOCAB]), "%s", pair.first.c_str());
            VOCAB++;
        } else {
            break;
        }
    }

    // Map parsed tokens to vocabulary IDs
    for (const std::string& token : all_parsed_tokens) {
        if (word_to_id.find(token) != word_to_id.end()) {
            training_word_ids.push_back(word_to_id[token]);
        } else {
            training_word_ids.push_back(0); // <unk>
        }
    }

    printf("Frequency-based Tokenization Complete. Unique Vocabulary Size: %d words/symbols (Max allowed: %d).\n", VOCAB, MAX_VOCAB);
    save_token_report(MODEL_FILE, sorted_vocab, word_to_id);
    return true;
}

// Wrapper for text generation inference.
void inference(const std::string& prompt, 
               const std::map<std::string,int>& word_to_id)
{
    auto prompt_ids = prompt_to_vector(prompt, word_to_id);
    generate_words(prompt_ids.data(), prompt_ids.size(), MAX_NEW_TOKENS);
}

// ============================================================================
// MODEL TRAINING LOOP
// ============================================================================
// Standard language model training loop:
// 1. Randomly sample sequence windows from text token buffer.
// 2. Perform forward pass (predict next token at each position).
// 3. Compute loss & gradients via backward pass.
// 4. Update model parameters with AdamW optimizer.
void train(const std::vector<int>& training_word_ids)
{
    int tokens[Context_LEN];
    int targets[Context_LEN];

    for (int step = 0; step < N_STEPS; step++) {

        zero_gradients();

        // Sample random sequence segment of length Context_LEN
        int start = rng() % (training_word_ids.size() - Context_LEN);
        for (int i = 0; i < Context_LEN; i++) {
            tokens[i]  = training_word_ids[start + i];
            targets[i] = training_word_ids[start + i + 1]; // Target is next token (shifted by 1)
        }

        float lr = get_lr(step, N_STEPS);
        float loss = forward_pass(tokens, targets, Context_LEN);
        backward_pass(tokens, targets, Context_LEN);
        adamw_step(lr, step);     //sgd_step(lr);

        if (step % 4000 == 0)
            printf("step %d loss %.4f\n", step, loss);
    }

    save_model(MODEL_FILE);
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
int main(int argc, char** argv)
{
    // Fallback to default prompt if no argument is supplied
    std::string prompt = (argc > 1) ? argv[1] : "A Frog and a";

    std::map<std::string,int> word_to_id;
    std::vector<int> training_word_ids;

    load_and_tokenize_corpus(TRAINING_FILE, training_word_ids, word_to_id);
    initialize_weights();

    std::ifstream check_file(MODEL_FILE, std::ios::binary);
    bool model_exists = check_file.is_open();
    if (model_exists) {
        check_file.close();
        load_model(MODEL_FILE);
    } else {
        train(training_word_ids);
    }

    inference(prompt, word_to_id);
}