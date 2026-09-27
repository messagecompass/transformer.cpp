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

// Dynamic vocabulary and corpus configuration
const int MAX_VOCAB = 20000;
int VOCAB = 0;
char VOCAB_WORDS[MAX_VOCAB][32];

const int SEQ_LEN  = 24;      
const int N_LAYERS = 6;
const int D_MODEL  = 256;      
const int N_HEADS  = 4;
const int D_HEAD   = D_MODEL / N_HEADS;   
const int D_MLP    = D_MODEL * 4;          

float token_embedding[MAX_VOCAB][D_MODEL];
float W_Q[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float W_K[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float W_V[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float W_O[N_LAYERS][D_MODEL][D_MODEL];
float W_mlp1[N_LAYERS][D_MLP][D_MODEL];
float W_mlp2[N_LAYERS][D_MODEL][D_MLP];
float ln_attn_scale[N_LAYERS][D_MODEL];
float ln_attn_bias [N_LAYERS][D_MODEL];
float ln_mlp_scale [N_LAYERS][D_MODEL];
float ln_mlp_bias  [N_LAYERS][D_MODEL];
float ln_final_scale[D_MODEL];
float ln_final_bias [D_MODEL];
float W_unembed[MAX_VOCAB][D_MODEL];

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

float s_xhat_attn[N_LAYERS][SEQ_LEN][D_MODEL];
float s_inv_std_attn[N_LAYERS][SEQ_LEN];
float s_normed_attn[N_LAYERS][SEQ_LEN][D_MODEL];
float s_q[N_LAYERS][N_HEADS][SEQ_LEN][D_HEAD];
float s_k[N_LAYERS][N_HEADS][SEQ_LEN][D_HEAD];
float s_v[N_LAYERS][N_HEADS][SEQ_LEN][D_HEAD];
float s_attn_w[N_LAYERS][N_HEADS][SEQ_LEN][SEQ_LEN];
float s_attn_out[N_LAYERS][SEQ_LEN][D_MODEL];
float s_xhat_mlp[N_LAYERS][SEQ_LEN][D_MODEL];
float s_inv_std_mlp[N_LAYERS][SEQ_LEN];
float s_normed_mlp[N_LAYERS][SEQ_LEN][D_MODEL];
float s_mlp_pre_relu[N_LAYERS][SEQ_LEN][D_MLP];
float s_xhat_final[SEQ_LEN][D_MODEL];
float s_inv_std_final[SEQ_LEN];
float s_normed_final[SEQ_LEN][D_MODEL];

float residual[SEQ_LEN][D_MODEL];
float logits  [SEQ_LEN][MAX_VOCAB];
float d_residual[SEQ_LEN][D_MODEL];
float d_logits  [SEQ_LEN][MAX_VOCAB];
float d_attn_out_buf[SEQ_LEN][D_MODEL];
float d_normed_buf  [SEQ_LEN][D_MODEL];
float d_q[N_HEADS][SEQ_LEN][D_HEAD];
float d_k[N_HEADS][SEQ_LEN][D_HEAD];
float d_v[N_HEADS][SEQ_LEN][D_HEAD];

std::mt19937 rng(42);

// GeLU activation function and its derivative
inline float relu(float x) { return x > 0.0f ? x : 0.0f; }
inline float relu_grad(float x) { return x > 0.0f ? 1.0f : 0.0f; }

// Learning rate schedule with warmup and cosine decay
float get_lr(int step, int total_steps) {
    int warmup_steps = 1000;
    float max_lr = 0.003f;
    float min_lr = 0.0001f;
    if (step < warmup_steps) {
        return max_lr * ((float)step / warmup_steps);
    }
    float progress = (float)(step - warmup_steps) / (total_steps - warmup_steps);
    return min_lr + 0.5f * (max_lr - min_lr) * (1.0f + cosf(progress * 3.14159265f));
}

void initialize_weights()
{
    std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
    auto fill_random = [&](float* ptr, int size) { for (int i = 0; i < size; i++) ptr[i] = dis(rng); };
    auto fill_constant = [&](float* ptr, int size, float val) { for (int i = 0; i < size; i++) ptr[i] = val; };

    fill_random(&token_embedding[0][0], VOCAB * D_MODEL);
    for (int layer = 0; layer < N_LAYERS; layer++) {
        for (int h = 0; h < N_HEADS; h++) {
            fill_random(&W_Q[layer][h][0][0], D_HEAD * D_MODEL);
            fill_random(&W_K[layer][h][0][0], D_HEAD * D_MODEL);
            fill_random(&W_V[layer][h][0][0], D_HEAD * D_MODEL);
        }
        fill_random(&W_O[layer][0][0], D_MODEL * D_MODEL);
        fill_random(&W_mlp1[layer][0][0], D_MLP * D_MODEL);
        fill_random(&W_mlp2[layer][0][D_MLP], D_MODEL * D_MLP); 
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

void zero_grads()
{
    memset(g_token_embedding, 0, sizeof(g_token_embedding));
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

float forward_pass(int* tokens, int* targets, int seq_len)
{
    for (int pos = 0; pos < seq_len; pos++)
        for (int d = 0; d < D_MODEL; d++)
            residual[pos][d] = token_embedding[tokens[pos]][d];

    float scale = 1.0f / sqrtf((float)D_HEAD);

    for (int layer = 0; layer < N_LAYERS; layer++)
    {
        for (int pos = 0; pos < seq_len; pos++)
            layer_norm_forward(residual[pos], ln_attn_scale[layer], ln_attn_bias[layer],
                             s_normed_attn[layer][pos], s_xhat_attn[layer][pos], &s_inv_std_attn[layer][pos]);

        // Optimized QKV projections
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

        // Optimized Attention Scoring with pre-allocated reduction
        for (int h = 0; h < N_HEADS; h++) {
            int offset = h * D_HEAD;
            for (int i = 0; i < seq_len; i++) {
                float max_val = -1e30f;
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
                for (int j = i + 1; j < seq_len; j++)
                    s_attn_w[layer][h][i][j] = -1e30f;

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

                for (int j = 0; j <= i; j++) {
                    float weight = s_attn_w[layer][h][i][j];
                    const float* v_ptr = s_v[layer][h][j];
                    float* out_ptr = &s_attn_out[layer][i][offset];
                    for (int dh = 0; dh < D_HEAD; dh++)
                        out_ptr[dh] += weight * v_ptr[dh];
                }
            }
        }

        // Output projection and residual add
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

        for (int pos = 0; pos < seq_len; pos++)
            layer_norm_forward(residual[pos], ln_mlp_scale[layer], ln_mlp_bias[layer],
                             s_normed_mlp[layer][pos], s_xhat_mlp[layer][pos], &s_inv_std_mlp[layer][pos]);

        for (int pos = 0; pos < seq_len; pos++) {
            for (int m = 0; m < D_MLP; m++) {
                float val = 0.0f;
                const float* norm_ptr = s_normed_mlp[layer][pos];
                const float* w_mlp1_ptr = W_mlp1[layer][m];
                for (int d = 0; d < D_MODEL; d++)
                    val += w_mlp1_ptr[d] * norm_ptr[d];
                s_mlp_pre_relu[layer][pos][m] = val; 
            }
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                const float* w_mlp2_ptr = W_mlp2[layer][d];
                for (int m = 0; m < D_MLP; m++) {
                    float gelu_out = relu(s_mlp_pre_relu[layer][pos][m]);
                    val += w_mlp2_ptr[m] * gelu_out;
                }
                residual[pos][d] += val;
            }
        }
    }

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


void backward_pass(int* tokens, int* targets, int seq_len)
{
    float scale = 1.0f / sqrtf((float)D_HEAD);

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

    float d_normed_final[SEQ_LEN][D_MODEL];
    memset(d_normed_final, 0, sizeof(d_normed_final));
    for (int pos = 0; pos < seq_len; pos++)
        for (int v = 0; v < VOCAB; v++) {
            for (int d = 0; d < D_MODEL; d++) {
                d_normed_final[pos][d] += d_logits[pos][v] * W_unembed[v][d];
                g_W_unembed[v][d] += d_logits[pos][v] * s_normed_final[pos][d];
            }
        }

    memset(d_residual, 0, sizeof(d_residual));
    for (int pos = 0; pos < seq_len; pos++)
        layer_norm_backward(d_normed_final[pos], s_xhat_final[pos], s_inv_std_final[pos],
                          ln_final_scale, d_residual[pos], g_ln_final_scale, g_ln_final_bias);

    for (int layer = N_LAYERS - 1; layer >= 0; layer--)
    {
        for (int pos = 0; pos < seq_len; pos++) {
            float d_gelu_out[D_MLP];
            for (int m = 0; m < D_MLP; m++) {
                float val = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    val += d_residual[pos][d] * W_mlp2[layer][d][m];
                d_gelu_out[m] = val;
            }

            for (int d = 0; d < D_MODEL; d++)
                for (int m = 0; m < D_MLP; m++) {
                    float gelu_val = relu(s_mlp_pre_relu[layer][pos][m]);
                    g_W_mlp2[layer][d][m] += d_residual[pos][d] * gelu_val;
                }

            float d_pre_gelu[D_MLP];
            for (int m = 0; m < D_MLP; m++)
                d_pre_gelu[m] = d_gelu_out[m] * relu_grad(s_mlp_pre_relu[layer][pos][m]);

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

            float d_res_from_mlp_ln[D_MODEL];
            layer_norm_backward(d_normed_mlp_pos, s_xhat_mlp[layer][pos], s_inv_std_mlp[layer][pos],
                              ln_mlp_scale[layer], d_res_from_mlp_ln, g_ln_mlp_scale[layer], g_ln_mlp_bias[layer]);

            for (int d = 0; d < D_MODEL; d++) d_residual[pos][d] += d_res_from_mlp_ln[d];
        }

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

        for (int h = 0; h < N_HEADS; h++) {
            int offset = h * D_HEAD;
            for (int i = 0; i < seq_len; i++) {
                float d_attn_w[SEQ_LEN];
                memset(d_attn_w, 0, sizeof(d_attn_w));
                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        d_attn_w[j] += d_attn_out_buf[i][offset + dh] * s_v[layer][h][j][dh];
                        d_v[h][j][dh] += s_attn_w[layer][h][i][j] * d_attn_out_buf[i][offset + dh];
                    }

                float dot = 0.0f;
                for (int j = 0; j < seq_len; j++) dot += d_attn_w[j] * s_attn_w[layer][h][i][j];
                float d_score[SEQ_LEN];
                for (int j = 0; j < seq_len; j++) d_score[j] = s_attn_w[layer][h][i][j] * (d_attn_w[j] - dot);

                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        d_q[h][i][dh] += d_score[j] * scale * s_k[layer][h][j][dh];
                        d_k[h][j][dh] += d_score[j] * scale * s_q[layer][h][i][dh];
                    }
            }
        }

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

        for (int pos = 0; pos < seq_len; pos++) {
            float d_res_from_attn_ln[D_MODEL];
            layer_norm_backward(d_normed_buf[pos], s_xhat_attn[layer][pos], s_inv_std_attn[layer][pos],
                              ln_attn_scale[layer], d_res_from_attn_ln, g_ln_attn_scale[layer], g_ln_attn_bias[layer]);
            for (int d = 0; d < D_MODEL; d++) d_residual[pos][d] += d_res_from_attn_ln[d];
        }
    }

    for (int pos = 0; pos < seq_len; pos++)
        for (int d = 0; d < D_MODEL; d++)
            g_token_embedding[tokens[pos]][d] += d_residual[pos][d];
}

void sgd_step(float lr)
{
    #define UPDATE(w, gw, size) \
        for (int _i = 0; _i < (size); _i++) \
            ((float*)(w))[_i] -= lr * ((float*)(gw))[_i];

    UPDATE(token_embedding, g_token_embedding, VOCAB * D_MODEL);
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

std::vector<int> prompt_to_vector(const std::string& prompt_str, const std::map<std::string, int>& word_to_id)
{
    std::vector<int> prompt_ids;
    std::stringstream ss(prompt_str);
    std::string raw_word;

    while (ss >> raw_word) {
        std::string cleaned = "";
        for (char c : raw_word) {
            if (c != '.' && c != ',' && c != ':' && c != '"' && c != ';') {
                cleaned += tolower(c);
            }
        }
        if (cleaned.empty()) continue;

        if (word_to_id.find(cleaned) != word_to_id.end()) {
            prompt_ids.push_back(word_to_id.at(cleaned));
        } else {
            printf("Warning: Prompt word '%s' not found in vocabulary. Ignored.\n", raw_word.c_str());
        }
    }
    return prompt_ids;
}

void save_model(const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Could not open file '%s' for saving weights.\n", filename.c_str());
        return;
    }

    // Write all core weight matrices sequentially
    file.write((char*)token_embedding, sizeof(token_embedding));
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

void load_model(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        printf("Error: Could not open file '%s' for loading weights.\n", filename.c_str());
        return;
    }

    file.read((char*)token_embedding, sizeof(token_embedding));
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

void generate_words(const int* prompt_ids, int prompt_len, int max_new_tokens)
{
    int current_tokens[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) current_tokens[i] = 0;

    for (int i = 0; i < prompt_len && i < SEQ_LEN; i++) {
        current_tokens[i] = prompt_ids[i];
    }

    printf("\n--- Generating Words ---\nPrompt: ");
    for (int i = 0; i < prompt_len; i++) {
        printf("%s ", VOCAB_WORDS[prompt_ids[i]]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < prompt_len; i++) {
        printf("%s ", VOCAB_WORDS[prompt_ids[i]]);
    }

    for (int t = 0; t < max_new_tokens; t++) {
        forward_pass(current_tokens, current_tokens, SEQ_LEN);

        int last_pos = 0;
        for (int i = 0; i < SEQ_LEN; i++) {
            if (current_tokens[i] != 0) last_pos = i;
        }

        for (int i = 0; i < SEQ_LEN; i++) {
            int recent_word = current_tokens[i];
            if (recent_word > 0 && recent_word < VOCAB) {
                logits[last_pos][recent_word] *= 0.05f;
            }
        }

        int best_token = 0;
        float max_logit = logits[last_pos][0];
        for (int v = 1; v < VOCAB; v++) {
            if (logits[last_pos][v] > max_logit) {
                max_logit = logits[last_pos][v];
                best_token = v;
            }
        }

        printf("%s ", VOCAB_WORDS[best_token]);

        for (int i = 0; i < SEQ_LEN - 1; i++) {
            current_tokens[i] = current_tokens[i + 1];
        }
        current_tokens[SEQ_LEN - 1] = best_token;
        if (std::string(VOCAB_WORDS[best_token]) == "<|endoftext|>") {
            printf("\n[Reached end of text]\n");
            break;
        }
    }
    printf("\n------------------------\n");
}

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
        corpus = "I walk down a wide road... <|endoftext|> Once upon a time..."; // fallback with special token
    }

    // Step 1: Parse and count frequencies of raw words
    std::stringstream ss(corpus);
    std::string raw_word;
    std::vector<std::string> all_parsed_words;
    std::map<std::string, int> word_counts;

    while (ss >> raw_word) {
        std::string cleaned = "";
        
        // Preserve special tokens like <|endoftext|> intact
        if (raw_word == "<|endoftext|>") {
            cleaned = "<|endoftext|>";
        } else {
            for (char c : raw_word) {
                if (c != '.' && c != ',' && c != ':' && c != '"' && c != ';') {
                    cleaned += tolower(c);
                }
            }
        }

        if (cleaned.empty()) continue;
        
        all_parsed_words.push_back(cleaned);
        word_counts[cleaned]++;
    }

    // Step 2: Sort unique words by frequency (descending)
    std::vector<std::pair<std::string, int>> sorted_vocab(word_counts.begin(), word_counts.end());
    std::sort(sorted_vocab.begin(), sorted_vocab.end(), [](const auto& a, const auto& b) {
        return a.second > b.second; // Higher frequency first
    });

    // Step 3: Initialize fixed special tokens
    word_to_id["<unk>"] = 0;
    snprintf(VOCAB_WORDS[0], sizeof(VOCAB_WORDS[0]), "%s", "<unk>");
    VOCAB = 1;

    // Ensure <|endoftext|> gets a high-priority slot if present
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

    // Fill the rest of the vocabulary up to MAX_VOCAB based on frequency
    for (const auto& pair : sorted_vocab) {
        if (pair.first == "<|endoftext|>") continue; // already added

        if (VOCAB < MAX_VOCAB) {
            word_to_id[pair.first] = VOCAB;
            snprintf(VOCAB_WORDS[VOCAB], sizeof(VOCAB_WORDS[VOCAB]), "%s", pair.first.c_str());
            VOCAB++;
        } else {
            // Stop once vocabulary limit is reached
            break;
        }
    }

    // Step 4: Map all parsed words into final training IDs
    for (const std::string& cleaned : all_parsed_words) {
        if (word_to_id.find(cleaned) != word_to_id.end()) {
            training_word_ids.push_back(word_to_id[cleaned]);
        } else {
            training_word_ids.push_back(0); // Map to <unk> if cut off by MAX_VOCAB
        }
    }

    printf("Frequency-based Tokenization Complete. Unique Vocabulary Size: %d words (Max allowed: %d).\n", VOCAB, MAX_VOCAB);
    return true;
}

int main(int argc, char* argv[])
{
    std::string trainingFile = "TinyStories-valid.txt";
    std::string modelFile = "tinystories_model.bin";
    std::map<std::string, int> word_to_id;
    std::vector<int> training_word_ids;

    if (!load_and_tokenize_corpus(trainingFile, training_word_ids, word_to_id)) {
        return 1;
    }

    initialize_weights();

    // Check if the saved model file already exists
    std::ifstream check_file(modelFile, std::ios::binary);
    bool model_exists = check_file.is_open();
    if (model_exists) {
        check_file.close();
    }

    if (model_exists) {
        printf("Found existing model '%s'. Skipping training and loading weights...\n", modelFile.c_str());
        load_model(modelFile);
    } else {
        int total_train_words = training_word_ids.size();
        if (total_train_words <= SEQ_LEN) {
            fprintf(stderr, "Error: Corpus is too short for SEQ_LEN (%d).\n", SEQ_LEN);
            return 1;
        }

        int tokens[SEQ_LEN], targets[SEQ_LEN];
        const int N_STEPS = 20000;

        printf("Training for %d steps ...\n", N_STEPS);

        for (int step = 0; step < N_STEPS; step++) {
            int start_idx = rng() % (total_train_words - SEQ_LEN);
            for (int i = 0; i < SEQ_LEN; i++) {
                tokens[i]  = training_word_ids[start_idx + i];
                targets[i] = training_word_ids[start_idx + i + 1];
            }

            float current_lr = get_lr(step, N_STEPS);

            zero_grads();
            float loss = forward_pass(tokens, targets, SEQ_LEN);
            backward_pass(tokens, targets, SEQ_LEN);
            sgd_step(current_lr);

            if (step % 4000 == 0 || step == N_STEPS - 1)
                printf("  step %5d: loss %.6f (lr: %.6f) [Time: %.1fs]\n", step, loss, current_lr, (float)clock() / CLOCKS_PER_SEC);
        }

        // Save weights after training completes
        save_model(modelFile);
    }

    // Process command-line prompt arguments properly as a combined string
    std::vector<int> prompt_ids;
    if (argc > 1) {
        std::string full_prompt = "";
        for (int c = 1; c < argc; c++) {
            full_prompt += std::string(argv[c]) + " ";
        }
        prompt_ids = prompt_to_vector(full_prompt, word_to_id);
    }

    // Default prompt fallback if none provided
    if (prompt_ids.empty()) {
        std::string default_prompt = "once upon a time";
        printf("No valid command-line prompt provided. Using default: \"%s\"\n", default_prompt.c_str());
        prompt_ids = prompt_to_vector(default_prompt, word_to_id);
    }

    int prompt_len = std::min((int)prompt_ids.size(), SEQ_LEN);
    generate_words(prompt_ids.data(), prompt_len, 100); // Set to 30 new tokens for a complete thought

    return 0;
}