// Sequential scalar transformer forward pass -- with weight loader
// Loads weights exported by train_and_export.py, runs inference,
// compares output against reference logits from PyTorch.

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

// ─── Model dimensions (must match train_and_export.py) ────────────────────────
const int SEQ_LEN = 16;
const int N_LAYERS = 4;
const int D_MODEL = 32;
const int N_HEADS = 4;
const int D_HEAD = D_MODEL / N_HEADS;
const int D_MLP = D_MODEL * 4;
const int VOCAB = 256;

// ─── Weight arrays ────────────────────────────────────────────────────────────
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

// ─── Working buffers ──────────────────────────────────────────────────────────
float residual  [SEQ_LEN][D_MODEL];
float normed    [SEQ_LEN][D_MODEL];
float q[N_HEADS][SEQ_LEN][D_HEAD];
float k[N_HEADS][SEQ_LEN][D_HEAD];
float v[N_HEADS][SEQ_LEN][D_HEAD];
float scores    [SEQ_LEN][SEQ_LEN];
float attn_out  [SEQ_LEN][D_MODEL];
float mlp_hidden[D_MLP];
float logits    [SEQ_LEN][VOCAB];

// ─── Weight loader ────────────────────────────────────────────────────────────
// Reads floats sequentially from the binary file in exactly the order
// that train_and_export.py wrote them.

static void read_floats(FILE* f, float* dst, int count)
{
    int n = fread(dst, sizeof(float), count, f);
    if (n != count) {
        fprintf(stderr, "Weight file too short (wanted %d floats, got %d)\n", count, n);
        exit(1);
    }
}

void load_weights(const char* path)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    // Token embedding [VOCAB, D_MODEL]
    read_floats(f, &token_embedding[0][0], VOCAB * D_MODEL);

    for (int layer = 0; layer < N_LAYERS; layer++)
    {
        // Q, K, V for each head [D_HEAD, D_MODEL] each
        // PyTorch nn.Linear stores weights as [out, in], which matches our layout.
        for (int h = 0; h < N_HEADS; h++) {
            read_floats(f, &W_Q[layer][h][0][0], D_HEAD * D_MODEL);
            read_floats(f, &W_K[layer][h][0][0], D_HEAD * D_MODEL);
            read_floats(f, &W_V[layer][h][0][0], D_HEAD * D_MODEL);
        }
        // W_O [D_MODEL, D_MODEL]
        read_floats(f, &W_O[layer][0][0], D_MODEL * D_MODEL);
        // MLP weights
        read_floats(f, &W_mlp1[layer][0][0], D_MLP * D_MODEL);
        read_floats(f, &W_mlp2[layer][0][0], D_MODEL * D_MLP);
        // Layer norms
        read_floats(f, &ln_attn_scale[layer][0], D_MODEL);
        read_floats(f, &ln_attn_bias [layer][0], D_MODEL);
        read_floats(f, &ln_mlp_scale [layer][0], D_MODEL);
        read_floats(f, &ln_mlp_bias  [layer][0], D_MODEL);
    }

    // Final layer norm
    read_floats(f, ln_final_scale, D_MODEL);
    read_floats(f, ln_final_bias,  D_MODEL);

    // Unembed [VOCAB, D_MODEL]
    read_floats(f, &W_unembed[0][0], VOCAB * D_MODEL);

    fclose(f);
    printf("Weights loaded from %s\n", path);
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

void layer_norm(float* x, const float* scale, const float* bias, int dim)
{
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;

    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++)
        x[i] = (x[i] - mean) * inv_std * scale[i] + bias[i];
}

void softmax(float* x, int n)
{
    float max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; i++)
        x[i] /= sum;
}

// ─── Forward pass ─────────────────────────────────────────────────────────────

void forward(int* tokens, int seq_len)
{
    // Load embeddings into residual stream
    for (int pos = 0; pos < seq_len; pos++)
        for (int d = 0; d < D_MODEL; d++)
            residual[pos][d] = token_embedding[tokens[pos]][d];

    float scale = 1.0f / sqrtf((float)D_HEAD);

    for (int layer = 0; layer < N_LAYERS; layer++)
    {
        // Layer norm before attention
        for (int pos = 0; pos < seq_len; pos++) {
            for (int d = 0; d < D_MODEL; d++)
                normed[pos][d] = residual[pos][d];
            layer_norm(normed[pos], ln_attn_scale[layer], ln_attn_bias[layer], D_MODEL);
        }

        // Project to Q, K, V
        for (int h = 0; h < N_HEADS; h++) {
            for (int pos = 0; pos < seq_len; pos++) {
                for (int dh = 0; dh < D_HEAD; dh++) {
                    float qv = 0.0f, kv = 0.0f, vv = 0.0f;
                    for (int d = 0; d < D_MODEL; d++) {
                        qv += W_Q[layer][h][dh][d] * normed[pos][d];
                        kv += W_K[layer][h][dh][d] * normed[pos][d];
                        vv += W_V[layer][h][dh][d] * normed[pos][d];
                    }
                    q[h][pos][dh] = qv;
                    k[h][pos][dh] = kv;
                    v[h][pos][dh] = vv;
                }
            }
        }

        // Attention: scores, softmax, weighted sum of values
        for (int pos = 0; pos < seq_len; pos++)
            for (int d = 0; d < D_MODEL; d++)
                attn_out[pos][d] = 0.0f;

        for (int h = 0; h < N_HEADS; h++) {
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j <= i; j++) {
                    float dot = 0.0f;
                    for (int dh = 0; dh < D_HEAD; dh++)
                        dot += q[h][i][dh] * k[h][j][dh];
                    scores[i][j] = dot * scale;
                }
                for (int j = i + 1; j < seq_len; j++)
                    scores[i][j] = -1e30f;

                softmax(scores[i], seq_len);

                int offset = h * D_HEAD;
                for (int j = 0; j <= i; j++)
                    for (int dh = 0; dh < D_HEAD; dh++)
                        attn_out[i][offset + dh] += scores[i][j] * v[h][j][dh];
            }
        }

        // Output projection + first residual addition
        for (int pos = 0; pos < seq_len; pos++) {
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                for (int d2 = 0; d2 < D_MODEL; d2++)
                    val += W_O[layer][d][d2] * attn_out[pos][d2];
                residual[pos][d] += val;
            }
        }

        // Layer norm before MLP
        for (int pos = 0; pos < seq_len; pos++) {
            for (int d = 0; d < D_MODEL; d++)
                normed[pos][d] = residual[pos][d];
            layer_norm(normed[pos], ln_mlp_scale[layer], ln_mlp_bias[layer], D_MODEL);
        }

        // MLP + second residual addition
        for (int pos = 0; pos < seq_len; pos++) {
            for (int m = 0; m < D_MLP; m++) {
                float val = 0.0f;
                for (int d = 0; d < D_MODEL; d++)
                    val += W_mlp1[layer][m][d] * normed[pos][d];
                mlp_hidden[m] = val > 0.0f ? val : 0.0f;  // ReLU
            }
            for (int d = 0; d < D_MODEL; d++) {
                float val = 0.0f;
                for (int m = 0; m < D_MLP; m++)
                    val += W_mlp2[layer][d][m] * mlp_hidden[m];
                residual[pos][d] += val;
            }
        }
    }

    // Final layer norm and unembed
    for (int pos = 0; pos < seq_len; pos++) {
        for (int d = 0; d < D_MODEL; d++)
            normed[pos][d] = residual[pos][d];
        layer_norm(normed[pos], ln_final_scale, ln_final_bias, D_MODEL);

        for (int voc = 0; voc < VOCAB; voc++) {
            float val = 0.0f;
            for (int d = 0; d < D_MODEL; d++)
                val += W_unembed[voc][d] * normed[pos][d];
            logits[pos][voc] = val;
        }
    }
}

// ─── Comparison against PyTorch reference logits ──────────────────────────────

void compare_logits(const char* ref_path)
{
    FILE* f = fopen(ref_path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", ref_path); exit(1); }

    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int count = 0;

    for (int pos = 0; pos < SEQ_LEN; pos++) {
        for (int voc = 0; voc < VOCAB; voc++) {
            float ref;
            fscanf(f, "%f", &ref);
            float diff = fabsf(logits[pos][voc] - ref);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
            count++;
        }
    }
    fclose(f);

    printf("Comparison against PyTorch reference:\n");
    printf("  Max absolute difference: %.6f\n", max_diff);
    printf("  Mean absolute difference: %.6f\n", sum_diff / count);
    printf("  Total values compared: %d\n", count);
    if (max_diff < 1e-4f)
        printf("  PASS: outputs match within floating-point tolerance.\n");
    else
        printf("  FAIL: outputs differ more than expected.\n");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    load_weights("weights.bin");

    int tokens[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) tokens[i] = i + 1;

    forward(tokens, SEQ_LEN);

    printf("\nC++ logits (first position, first 8 vocab entries):\n");
    for (int voc = 0; voc < 8; voc++)
        printf("  [%d] %.6f\n", voc, logits[0][voc]);
    printf("\n");

    compare_logits("ref_logits.txt");

    return 0;
}
