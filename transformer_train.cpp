// Scalar C++ transformer training with backpropagation
// Loads initial weights from Python, trains with SGD, compares losses.
// Every operation is an explicit scalar loop — no libraries, no SIMD.

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// ─── Model dimensions (must match train_verify.py) ──────────────────────────
const int SEQ_LEN  = 16;
const int N_LAYERS = 4;
const int D_MODEL  = 32;
const int N_HEADS  = 4;
const int D_HEAD   = D_MODEL / N_HEADS;   // 8
const int D_MLP    = D_MODEL * 4;          // 128
const int VOCAB    = 256;

// ─── Weight arrays ──────────────────────────────────────────────────────────
float token_embedding[VOCAB][D_MODEL];
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
float W_unembed[VOCAB][D_MODEL];

// ─── Gradient arrays (same shapes, prefix g_) ──────────────────────────────
float g_token_embedding[VOCAB][D_MODEL];
float g_W_Q[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float g_W_K[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float g_W_V[N_LAYERS][N_HEADS][D_HEAD][D_MODEL];
float g_W_O[N_LAYERS][D_MODEL][D_MODEL];
float g_W_mlp1[N_LAYERS][D_MLP][D_MODEL];
float g_W_mlp2[N_LAYERS][D_MODEL][D_MLP];
float g_ln_attn_scale[N_LAYERS][D_MODEL];
float g_ln_attn_bias [N_LAYERS][D_MODEL];
float g_ln_mlp_scale [N_LAYERS][D_MODEL];
float g_ln_mlp_bias  [N_LAYERS][D_MODEL];
float g_ln_final_scale[D_MODEL];
float g_ln_final_bias [D_MODEL];
float g_W_unembed[VOCAB][D_MODEL];

// ─── Saved intermediates from forward pass (prefix s_) ──────────────────────
float s_xhat_attn[N_LAYERS][SEQ_LEN][D_MODEL];
float s_inv_std_attn[N_LAYERS][SEQ_LEN];
float s_normed_attn[N_LAYERS][SEQ_LEN][D_MODEL];
float s_q[N_LAYERS][N_HEADS][SEQ_LEN][D_HEAD];
float s_k[N_LAYERS][N_HEADS][SEQ_LEN][D_HEAD];
float s_v[N_LAYERS][N_HEADS][SEQ_LEN][D_HEAD];
float s_attn_w[N_LAYERS][N_HEADS][SEQ_LEN][SEQ_LEN];  // post-softmax
float s_attn_out[N_LAYERS][SEQ_LEN][D_MODEL];          // before W_O
float s_xhat_mlp[N_LAYERS][SEQ_LEN][D_MODEL];
float s_inv_std_mlp[N_LAYERS][SEQ_LEN];
float s_normed_mlp[N_LAYERS][SEQ_LEN][D_MODEL];
float s_mlp_pre_relu[N_LAYERS][SEQ_LEN][D_MLP];
float s_xhat_final[SEQ_LEN][D_MODEL];
float s_inv_std_final[SEQ_LEN];
float s_normed_final[SEQ_LEN][D_MODEL];

// ─── Working buffers ────────────────────────────────────────────────────────
float residual[SEQ_LEN][D_MODEL];
float logits  [SEQ_LEN][VOCAB];
float d_residual[SEQ_LEN][D_MODEL];
float d_logits  [SEQ_LEN][VOCAB];
float d_attn_out_buf[SEQ_LEN][D_MODEL];
float d_normed_buf  [SEQ_LEN][D_MODEL];
float d_q[N_HEADS][SEQ_LEN][D_HEAD];
float d_k[N_HEADS][SEQ_LEN][D_HEAD];
float d_v[N_HEADS][SEQ_LEN][D_HEAD];

// ─── Weight loader ──────────────────────────────────────────────────────────
static void read_floats(FILE* f, float* dst, int count)
{
    int n = fread(dst, sizeof(float), count, f);
    if (n != count) {
        fprintf(stderr, "Weight file too short (wanted %d, got %d)\n", count, n);
        exit(1);
    }
}

void load_weights(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    read_floats(f, &token_embedding[0][0], VOCAB * D_MODEL);
    for (int layer = 0; layer < N_LAYERS; layer++) {
        for (int h = 0; h < N_HEADS; h++) {
            read_floats(f, &W_Q[layer][h][0][0], D_HEAD * D_MODEL);
            read_floats(f, &W_K[layer][h][0][0], D_HEAD * D_MODEL);
            read_floats(f, &W_V[layer][h][0][0], D_HEAD * D_MODEL);
        }
        read_floats(f, &W_O[layer][0][0], D_MODEL * D_MODEL);
        read_floats(f, &W_mlp1[layer][0][0], D_MLP * D_MODEL);
        read_floats(f, &W_mlp2[layer][0][0], D_MODEL * D_MLP);
        read_floats(f, &ln_attn_scale[layer][0], D_MODEL);
        read_floats(f, &ln_attn_bias [layer][0], D_MODEL);
        read_floats(f, &ln_mlp_scale [layer][0], D_MODEL);
        read_floats(f, &ln_mlp_bias  [layer][0], D_MODEL);
    }
    read_floats(f, ln_final_scale, D_MODEL);
    read_floats(f, ln_final_bias,  D_MODEL);
    read_floats(f, &W_unembed[0][0], VOCAB * D_MODEL);
    fclose(f);
    printf("Weights loaded from %s\n", path);
}

// ─── Layer norm forward (saves x_hat and inv_std for backward) ──────────────
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

// ─── Layer norm backward ────────────────────────────────────────────────────
// Given:  y[i] = scale[i] * x_hat[i] + bias[i]
//         x_hat[i] = (x[i] - mean) * inv_std
// Computes dx[i] from dy[i], accumulates into d_scale and d_bias.
//
// Formula: dx[i] = inv_std * (d_xhat[i] - mean(d_xhat) - x_hat[i]*mean(d_xhat*x_hat))
//   where d_xhat[i] = dy[i] * scale[i]
void layer_norm_backward(const float* dy, const float* xhat, float inv_std,
                         const float* scale, float* dx,
                         float* d_scale, float* d_bias)
{
    for (int i = 0; i < D_MODEL; i++) {
        d_scale[i] += dy[i] * xhat[i];
        d_bias[i]  += dy[i];
    }

    float d_xhat[D_MODEL];
    for (int i = 0; i < D_MODEL; i++)
        d_xhat[i] = dy[i] * scale[i];

    float mean_dxhat = 0.0f;
    for (int i = 0; i < D_MODEL; i++) mean_dxhat += d_xhat[i];
    mean_dxhat /= D_MODEL;

    float mean_dxhat_xhat = 0.0f;
    for (int i = 0; i < D_MODEL; i++) mean_dxhat_xhat += d_xhat[i] * xhat[i];
    mean_dxhat_xhat /= D_MODEL;

    for (int i = 0; i < D_MODEL; i++)
        dx[i] = inv_std * (d_xhat[i] - mean_dxhat - xhat[i] * mean_dxhat_xhat);
}

// ─── Zero all gradients ─────────────────────────────────────────────────────
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
    memset(g_ln_mlp_scale,  0, sizeof(g_ln_mlp_scale));
    memset(g_ln_mlp_bias,   0, sizeof(g_ln_mlp_bias));
    memset(g_ln_final_scale, 0, sizeof(g_ln_final_scale));
    memset(g_ln_final_bias,  0, sizeof(g_ln_final_bias));
    memset(g_W_unembed, 0, sizeof(g_W_unembed));
}

// ─── Forward pass (saves all intermediates for backward) ────────────────────
float forward_train(int* tokens, int* targets, int seq_len)
{
    // Embed tokens into residual stream
    for (int pos = 0; pos < seq_len; pos++)
        for (int d = 0; d < D_MODEL; d++)
            residual[pos][d] = token_embedding[tokens[pos]][d];

    float scale = 1.0f / sqrtf((float)D_HEAD);

    for (int layer = 0; layer < N_LAYERS; layer++)
    {
        // ── Attention sublayer ──────────────────────────────────────────

        // Layer norm before attention
        for (int pos = 0; pos < seq_len; pos++)
            layer_norm_forward(residual[pos],
                             ln_attn_scale[layer], ln_attn_bias[layer],
                             s_normed_attn[layer][pos],
                             s_xhat_attn[layer][pos],
                             &s_inv_std_attn[layer][pos]);

        // Q, K, V projections
        for (int h = 0; h < N_HEADS; h++)
            for (int pos = 0; pos < seq_len; pos++)
                for (int dh = 0; dh < D_HEAD; dh++) {
                    float qv = 0, kv = 0, vv = 0;
                    for (int d = 0; d < D_MODEL; d++) {
                        qv += W_Q[layer][h][dh][d] * s_normed_attn[layer][pos][d];
                        kv += W_K[layer][h][dh][d] * s_normed_attn[layer][pos][d];
                        vv += W_V[layer][h][dh][d] * s_normed_attn[layer][pos][d];
                    }
                    s_q[layer][h][pos][dh] = qv;
                    s_k[layer][h][pos][dh] = kv;
                    s_v[layer][h][pos][dh] = vv;
                }

        // Attention: scores, softmax, weighted sum
        for (int pos = 0; pos < seq_len; pos++)
            for (int d = 0; d < D_MODEL; d++)
                s_attn_out[layer][pos][d] = 0.0f;

        for (int h = 0; h < N_HEADS; h++) {
            for (int i = 0; i < seq_len; i++) {
                // Compute raw scores
                for (int j = 0; j <= i; j++) {
                    float dot = 0.0f;
                    for (int dh = 0; dh < D_HEAD; dh++)
                        dot += s_q[layer][h][i][dh] * s_k[layer][h][j][dh];
                    s_attn_w[layer][h][i][j] = dot * scale;
                }
                for (int j = i + 1; j < seq_len; j++)
                    s_attn_w[layer][h][i][j] = -1e30f;

                // Softmax in place
                float max_val = s_attn_w[layer][h][i][0];
                for (int j = 1; j < seq_len; j++)
                    if (s_attn_w[layer][h][i][j] > max_val)
                        max_val = s_attn_w[layer][h][i][j];
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    s_attn_w[layer][h][i][j] = expf(s_attn_w[layer][h][i][j] - max_val);
                    sum += s_attn_w[layer][h][i][j];
                }
                for (int j = 0; j < seq_len; j++)
                    s_attn_w[layer][h][i][j] /= sum;

                // Weighted sum of values
                int offset = h * D_HEAD;
                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++)
                        s_attn_out[layer][i][offset + dh] +=
                            s_attn_w[layer][h][i][j] * s_v[layer][h][j][dh];
            }
        }

        // Output projection + residual addition
        for (int pos = 0; pos < seq_len; pos++)
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                for (int d2 = 0; d2 < D_MODEL; d2++)
                    val += W_O[layer][d][d2] * s_attn_out[layer][pos][d2];
                residual[pos][d] += val;
            }

        // ── MLP sublayer ────────────────────────────────────────────────

        // Layer norm before MLP
        for (int pos = 0; pos < seq_len; pos++)
            layer_norm_forward(residual[pos],
                             ln_mlp_scale[layer], ln_mlp_bias[layer],
                             s_normed_mlp[layer][pos],
                             s_xhat_mlp[layer][pos],
                             &s_inv_std_mlp[layer][pos]);

        // MLP: first linear → ReLU → second linear → residual addition
        for (int pos = 0; pos < seq_len; pos++) {
            for (int m = 0; m < D_MLP; m++) {
                float val = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    val += W_mlp1[layer][m][d] * s_normed_mlp[layer][pos][d];
                s_mlp_pre_relu[layer][pos][m] = val;
            }
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                for (int m = 0; m < D_MLP; m++) {
                    float relu_out = s_mlp_pre_relu[layer][pos][m] > 0
                                   ? s_mlp_pre_relu[layer][pos][m] : 0.0f;
                    val += W_mlp2[layer][d][m] * relu_out;
                }
                residual[pos][d] += val;
            }
        }
    }

    // ── Final layer norm + unembed ──────────────────────────────────────
    for (int pos = 0; pos < seq_len; pos++) {
        layer_norm_forward(residual[pos],
                         ln_final_scale, ln_final_bias,
                         s_normed_final[pos],
                         s_xhat_final[pos],
                         &s_inv_std_final[pos]);

        for (int voc = 0; voc < VOCAB; voc++) {
            float val = 0.0f;
            for (int d = 0; d < D_MODEL; d++)
                val += W_unembed[voc][d] * s_normed_final[pos][d];
            logits[pos][voc] = val;
        }
    }

    // ── Cross-entropy loss ──────────────────────────────────────────────
    float loss = 0.0f;
    for (int pos = 0; pos < seq_len; pos++) {
        float max_val = logits[pos][0];
        for (int v = 1; v < VOCAB; v++)
            if (logits[pos][v] > max_val) max_val = logits[pos][v];
        float sum_exp = 0.0f;
        for (int v = 0; v < VOCAB; v++)
            sum_exp += expf(logits[pos][v] - max_val);
        float log_sum_exp = max_val + logf(sum_exp);
        loss -= (logits[pos][targets[pos]] - log_sum_exp);
    }
    loss /= seq_len;
    return loss;
}

// ─── Backward pass ──────────────────────────────────────────────────────────
void backward(int* tokens, int* targets, int seq_len)
{
    float scale = 1.0f / sqrtf((float)D_HEAD);

    // ── 1. Cross-entropy gradient → d_logits ────────────────────────────
    // d_logits[pos][v] = (1/seq_len) * (softmax(logits[pos])[v] - one_hot)
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

    // ── 2. Unembed backward ─────────────────────────────────────────────
    // logits[pos][v] = sum_d W_unembed[v][d] * normed_final[pos][d]
    float d_normed_final[SEQ_LEN][D_MODEL];
    memset(d_normed_final, 0, sizeof(d_normed_final));
    for (int pos = 0; pos < seq_len; pos++)
        for (int v = 0; v < VOCAB; v++) {
            for (int d = 0; d < D_MODEL; d++) {
                d_normed_final[pos][d] += d_logits[pos][v] * W_unembed[v][d];
                g_W_unembed[v][d] += d_logits[pos][v] * s_normed_final[pos][d];
            }
        }

    // ── 3. Final layer norm backward ────────────────────────────────────
    memset(d_residual, 0, sizeof(d_residual));
    for (int pos = 0; pos < seq_len; pos++)
        layer_norm_backward(d_normed_final[pos],
                          s_xhat_final[pos], s_inv_std_final[pos],
                          ln_final_scale, d_residual[pos],
                          g_ln_final_scale, g_ln_final_bias);

    // ── 4. Layer backward (top to bottom) ───────────────────────────────
    for (int layer = N_LAYERS - 1; layer >= 0; layer--)
    {
        // ── MLP sublayer backward ───────────────────────────────────
        // residual_out = residual_in + mlp2(relu(mlp1(normed_mlp)))
        // d_residual carries both the identity-path and sublayer-path gradients.
        // We use d_residual as d_mlp_out (since d(residual_out)/d(mlp_out) = 1),
        // then ADD the layer-norm-path gradient back into d_residual.

        for (int pos = 0; pos < seq_len; pos++) {
            // mlp2 backward: d_relu_out[m] = sum_d d_residual[pos][d] * W_mlp2[d][m]
            float d_relu_out[D_MLP];
            for (int m = 0; m < D_MLP; m++) {
                float val = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    val += d_residual[pos][d] * W_mlp2[layer][d][m];
                d_relu_out[m] = val;
            }

            // mlp2 weight gradient
            for (int d = 0; d < D_MODEL; d++)
                for (int m = 0; m < D_MLP; m++) {
                    float relu_out = s_mlp_pre_relu[layer][pos][m] > 0
                                   ? s_mlp_pre_relu[layer][pos][m] : 0.0f;
                    g_W_mlp2[layer][d][m] += d_residual[pos][d] * relu_out;
                }

            // ReLU backward
            float d_pre_relu[D_MLP];
            for (int m = 0; m < D_MLP; m++)
                d_pre_relu[m] = s_mlp_pre_relu[layer][pos][m] > 0
                              ? d_relu_out[m] : 0.0f;

            // mlp1 backward: d_normed_mlp[d] = sum_m d_pre_relu[m] * W_mlp1[m][d]
            float d_normed_mlp_pos[D_MODEL];
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                for (int m = 0; m < D_MLP; m++)
                    val += d_pre_relu[m] * W_mlp1[layer][m][d];
                d_normed_mlp_pos[d] = val;
            }

            // mlp1 weight gradient
            for (int m = 0; m < D_MLP; m++)
                for (int d = 0; d < D_MODEL; d++)
                    g_W_mlp1[layer][m][d] += d_pre_relu[m] * s_normed_mlp[layer][pos][d];

            // MLP layer norm backward → accumulate into d_residual
            float d_res_from_mlp_ln[D_MODEL];
            layer_norm_backward(d_normed_mlp_pos,
                              s_xhat_mlp[layer][pos], s_inv_std_mlp[layer][pos],
                              ln_mlp_scale[layer], d_res_from_mlp_ln,
                              g_ln_mlp_scale[layer], g_ln_mlp_bias[layer]);

            for (int d = 0; d < D_MODEL; d++)
                d_residual[pos][d] += d_res_from_mlp_ln[d];
        }

        // ── Attention sublayer backward ─────────────────────────────

        // W_O backward: d_attn_out[pos][d2] = sum_d d_residual[pos][d] * W_O[d][d2]
        memset(d_attn_out_buf, 0, sizeof(d_attn_out_buf));
        for (int pos = 0; pos < seq_len; pos++) {
            for (int d2 = 0; d2 < D_MODEL; d2++) {
                float val = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    val += d_residual[pos][d] * W_O[layer][d][d2];
                d_attn_out_buf[pos][d2] = val;
            }
            // W_O weight gradient
            for (int d = 0; d < D_MODEL; d++)
                for (int d2 = 0; d2 < D_MODEL; d2++)
                    g_W_O[layer][d][d2] += d_residual[pos][d]
                                         * s_attn_out[layer][pos][d2];
        }

        // Attention backward (per head)
        memset(d_q, 0, sizeof(d_q));
        memset(d_k, 0, sizeof(d_k));
        memset(d_v, 0, sizeof(d_v));

        for (int h = 0; h < N_HEADS; h++) {
            int offset = h * D_HEAD;
            for (int i = 0; i < seq_len; i++) {
                // Backward through weighted sum of values
                float d_attn_w[SEQ_LEN];
                memset(d_attn_w, 0, sizeof(d_attn_w));
                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        d_attn_w[j] += d_attn_out_buf[i][offset + dh]
                                      * s_v[layer][h][j][dh];
                        d_v[h][j][dh] += s_attn_w[layer][h][i][j]
                                        * d_attn_out_buf[i][offset + dh];
                    }

                // Softmax backward
                // d_score[j] = w[j] * (d_w[j] - dot(d_w, w))
                float dot = 0.0f;
                for (int j = 0; j < seq_len; j++)
                    dot += d_attn_w[j] * s_attn_w[layer][h][i][j];
                float d_score[SEQ_LEN];
                for (int j = 0; j < seq_len; j++)
                    d_score[j] = s_attn_w[layer][h][i][j] * (d_attn_w[j] - dot);

                // Score backward: score[i][j] = (q·k) * scale
                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        d_q[h][i][dh] += d_score[j] * scale
                                        * s_k[layer][h][j][dh];
                        d_k[h][j][dh] += d_score[j] * scale
                                        * s_q[layer][h][i][dh];
                    }
            }
        }

        // Q, K, V projection backward
        memset(d_normed_buf, 0, sizeof(d_normed_buf));
        for (int h = 0; h < N_HEADS; h++)
            for (int pos = 0; pos < seq_len; pos++)
                for (int dh = 0; dh < D_HEAD; dh++)
                    for (int d = 0; d < D_MODEL; d++) {
                        d_normed_buf[pos][d] += d_q[h][pos][dh]
                                              * W_Q[layer][h][dh][d];
                        d_normed_buf[pos][d] += d_k[h][pos][dh]
                                              * W_K[layer][h][dh][d];
                        d_normed_buf[pos][d] += d_v[h][pos][dh]
                                              * W_V[layer][h][dh][d];
                        g_W_Q[layer][h][dh][d] += d_q[h][pos][dh]
                                                * s_normed_attn[layer][pos][d];
                        g_W_K[layer][h][dh][d] += d_k[h][pos][dh]
                                                * s_normed_attn[layer][pos][d];
                        g_W_V[layer][h][dh][d] += d_v[h][pos][dh]
                                                * s_normed_attn[layer][pos][d];
                    }

        // Attention layer norm backward → accumulate into d_residual
        for (int pos = 0; pos < seq_len; pos++) {
            float d_res_from_attn_ln[D_MODEL];
            layer_norm_backward(d_normed_buf[pos],
                              s_xhat_attn[layer][pos], s_inv_std_attn[layer][pos],
                              ln_attn_scale[layer], d_res_from_attn_ln,
                              g_ln_attn_scale[layer], g_ln_attn_bias[layer]);
            for (int d = 0; d < D_MODEL; d++)
                d_residual[pos][d] += d_res_from_attn_ln[d];
        }
    }

    // ── 5. Embedding backward ───────────────────────────────────────────
    for (int pos = 0; pos < seq_len; pos++)
        for (int d = 0; d < D_MODEL; d++)
            g_token_embedding[tokens[pos]][d] += d_residual[pos][d];
}

// ─── SGD update: param -= lr * grad ─────────────────────────────────────────
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

// ─── Main ───────────────────────────────────────────────────────────────────
int main()
{
    load_weights("weights.bin");

    // Fixed training data: tokens [1..16], targets [2..17]
    int tokens[SEQ_LEN], targets[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) {
        tokens[i]  = i + 1;
        targets[i] = i + 2;
    }

    const int N_STEPS = 50;
    const float LR = 0.01f;

    printf("\nTraining for %d steps with SGD (lr=%.4f)\n\n", N_STEPS, LR);

    float losses[N_STEPS];
    for (int step = 0; step < N_STEPS; step++) {
        zero_grads();
        float loss = forward_train(tokens, targets, SEQ_LEN);
        backward(tokens, targets, SEQ_LEN);
        sgd_step(LR);
        losses[step] = loss;
        if (step % 5 == 0 || step == N_STEPS - 1)
            printf("  step %3d: loss %.6f\n", step, loss);
    }

    // Write losses for comparison
    FILE* f = fopen("cpp_losses.txt", "w");
    for (int i = 0; i < N_STEPS; i++)
        fprintf(f, "%.6f\n", losses[i]);
    fclose(f);
    printf("\nC++ losses written to cpp_losses.txt\n");

    // Compare against Python losses if available
    FILE* pf = fopen("py_losses.txt", "r");
    if (pf) {
        printf("\nComparison with PyTorch:\n");
        float max_diff = 0.0f;
        for (int i = 0; i < N_STEPS; i++) {
            float py_loss;
            if (fscanf(pf, "%f", &py_loss) != 1) break;
            float diff = fabsf(losses[i] - py_loss);
            if (diff > max_diff) max_diff = diff;
            if (i % 10 == 0)
                printf("  step %3d: C++ %.6f  Py %.6f  diff %.6f\n",
                       i, losses[i], py_loss, diff);
        }
        fclose(pf);
        printf("\n  Max loss difference: %.6f\n", max_diff);
        if (max_diff < 0.01f)
            printf("  PASS: training losses match closely.\n");
        else
            printf("  WARNING: losses diverged.\n");
    } else {
        printf("(No py_losses.txt found — run train_verify.py first to compare.)\n");
    }

    return 0;
}
