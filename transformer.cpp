// Single-file educational transformer: trains a tiny GPT model from scratch in C++ with zero external dependencies.
// Uses a dynamic corpus text, automatic tokenization, GeLU activation, and learning rate warmup/decay.

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <fstream>
#include <ctime>

// ============================================================================
// HYPERPARAMETERS & CONFIGURATION
// ============================================================================
const int   maximum_vocabulary_size             = 4900;
const int   context_length_limit                = 512;      
const int   total_transformer_layers            = 6;
const int   embedding_dimension_size            = 128;      
const int   attention_head_count                = 4;
const int   attention_head_dimension            = embedding_dimension_size / attention_head_count;   
const int   feed_forward_hidden_dimension_size  = embedding_dimension_size * 4;          

const int   total_training_steps                = 20000;
const int   learning_rate_warmup_steps          = 1000;
const float maximum_learning_rate               = 0.001f;
const float minimum_learning_rate               = 0.0001f;
const float early_stopping_loss_threshold       = 2.5f;

const int   maximum_new_tokens_to_generate      = 100;
std::string model_file_path;

// ============================================================================
// MODEL PARAMETERS & GRADIENT BUFFERS
// ============================================================================

int current_vocabulary_size = 0;
char vocabulary_word_strings[maximum_vocabulary_size][32];

// 1. Token & Positional Embeddings
float token_embedding_parameters[maximum_vocabulary_size][embedding_dimension_size];
float positional_embedding_parameters[context_length_limit][embedding_dimension_size];

// 2. Attention & Feed-Forward Weights
float query_projection_weight_matrix[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float key_projection_weight_matrix[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float value_projection_weight_matrix[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float output_projection_weight_matrix[total_transformer_layers][embedding_dimension_size][embedding_dimension_size];

float feed_forward_expansion_weight_matrix[total_transformer_layers][feed_forward_hidden_dimension_size][embedding_dimension_size];
float feed_forward_contraction_weight_matrix[total_transformer_layers][embedding_dimension_size][feed_forward_hidden_dimension_size];

// 3. Layer Normalization Parameters
float attention_layer_normalization_scale_parameters[total_transformer_layers][embedding_dimension_size];
float attention_layer_normalization_bias_parameters[total_transformer_layers][embedding_dimension_size];
float feed_forward_layer_normalization_scale_parameters[total_transformer_layers][embedding_dimension_size];
float feed_forward_layer_normalization_bias_parameters[total_transformer_layers][embedding_dimension_size];
float final_layer_normalization_scale_parameters[embedding_dimension_size];
float final_layer_normalization_bias_parameters[embedding_dimension_size];

// 4. Unembedding Head
float unembedding_projection_weight_matrix[maximum_vocabulary_size][embedding_dimension_size];

// 5. Parameter Gradients
float token_embedding_gradient_buffer[maximum_vocabulary_size][embedding_dimension_size];
float positional_embedding_gradient_buffer[context_length_limit][embedding_dimension_size];
float query_projection_weight_gradient_matrix[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float key_projection_weight_gradient_matrix[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float value_projection_weight_gradient_matrix[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float output_projection_weight_gradient_matrix[total_transformer_layers][embedding_dimension_size][embedding_dimension_size];
float feed_forward_expansion_weight_gradient_matrix[total_transformer_layers][feed_forward_hidden_dimension_size][embedding_dimension_size];
float feed_forward_contraction_weight_gradient_matrix[total_transformer_layers][embedding_dimension_size][feed_forward_hidden_dimension_size];
float attention_layer_normalization_scale_gradient_parameters[total_transformer_layers][embedding_dimension_size];
float attention_layer_normalization_bias_gradient_parameters[total_transformer_layers][embedding_dimension_size];
float feed_forward_layer_normalization_scale_gradient_parameters[total_transformer_layers][embedding_dimension_size];
float feed_forward_layer_normalization_bias_gradient_parameters[total_transformer_layers][embedding_dimension_size];
float final_layer_normalization_scale_gradient_parameters[embedding_dimension_size];
float final_layer_normalization_bias_gradient_parameters[embedding_dimension_size];
float unembedding_projection_weight_gradient_matrix[maximum_vocabulary_size][embedding_dimension_size];

// 6. Intermediate Saved Activations for Backward Pass
float saved_attention_normalized_hat[total_transformer_layers][context_length_limit][embedding_dimension_size];
float saved_attention_inverse_standard_deviation[total_transformer_layers][context_length_limit];
float saved_attention_normalized_values[total_transformer_layers][context_length_limit][embedding_dimension_size];
float saved_query_vectors[total_transformer_layers][attention_head_count][context_length_limit][attention_head_dimension];
float saved_key_vectors[total_transformer_layers][attention_head_count][context_length_limit][attention_head_dimension];
float saved_value_vectors[total_transformer_layers][attention_head_count][context_length_limit][attention_head_dimension];
float saved_attention_weights[total_transformer_layers][attention_head_count][context_length_limit][context_length_limit];
float saved_attention_output[total_transformer_layers][context_length_limit][embedding_dimension_size];
float saved_feed_forward_normalized_hat[total_transformer_layers][context_length_limit][embedding_dimension_size];
float saved_feed_forward_inverse_standard_deviation[total_transformer_layers][context_length_limit];
float saved_feed_forward_normalized_values[total_transformer_layers][context_length_limit][embedding_dimension_size];
float saved_feed_forward_pre_activation[total_transformer_layers][context_length_limit][feed_forward_hidden_dimension_size];
float saved_final_normalized_hat[context_length_limit][embedding_dimension_size];
float saved_final_inverse_standard_deviation[context_length_limit];
float saved_final_normalized_values[context_length_limit][embedding_dimension_size];

// Residual Stream & Logit Buffers
float residual_stream_buffer[context_length_limit][embedding_dimension_size];
float vocabulary_logits_buffer[context_length_limit][maximum_vocabulary_size];
float residual_gradient_buffer[context_length_limit][embedding_dimension_size];
float logits_gradient_buffer[context_length_limit][maximum_vocabulary_size];
float attention_output_gradient_buffer[context_length_limit][embedding_dimension_size];
float normalized_buffer[context_length_limit][embedding_dimension_size];
float query_gradient_buffer[attention_head_count][context_length_limit][attention_head_dimension];
float key_gradient_buffer[attention_head_count][context_length_limit][attention_head_dimension];
float value_gradient_buffer[attention_head_count][context_length_limit][attention_head_dimension];

// ============================================================================
// ADAMW OPTIMIZER MOMENTUM BUFFERS
// ============================================================================
float momentum_token_embedding[maximum_vocabulary_size][embedding_dimension_size], velocity_token_embedding[maximum_vocabulary_size][embedding_dimension_size];
float momentum_query_projection[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size], velocity_query_projection[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float momentum_key_projection[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size], velocity_key_projection[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float momentum_value_projection[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size], velocity_value_projection[total_transformer_layers][attention_head_count][attention_head_dimension][embedding_dimension_size];
float momentum_output_projection[total_transformer_layers][embedding_dimension_size][embedding_dimension_size], velocity_output_projection[total_transformer_layers][embedding_dimension_size][embedding_dimension_size];
float momentum_feed_forward_expansion[total_transformer_layers][feed_forward_hidden_dimension_size][embedding_dimension_size], velocity_feed_forward_expansion[total_transformer_layers][feed_forward_hidden_dimension_size][embedding_dimension_size];
float momentum_feed_forward_contraction[total_transformer_layers][embedding_dimension_size][feed_forward_hidden_dimension_size], velocity_feed_forward_contraction[total_transformer_layers][embedding_dimension_size][feed_forward_hidden_dimension_size];
float momentum_attention_scale_parameters[total_transformer_layers][embedding_dimension_size], velocity_attention_scale_parameters[total_transformer_layers][embedding_dimension_size];
float momentum_attention_bias_parameters[total_transformer_layers][embedding_dimension_size], velocity_attention_bias_parameters[total_transformer_layers][embedding_dimension_size];
float momentum_feed_forward_scale_parameters[total_transformer_layers][embedding_dimension_size], velocity_feed_forward_scale_parameters[total_transformer_layers][embedding_dimension_size];
float momentum_feed_forward_bias_parameters[total_transformer_layers][embedding_dimension_size], velocity_feed_forward_bias_parameters[total_transformer_layers][embedding_dimension_size];
float momentum_final_scale_parameters[embedding_dimension_size], velocity_final_scale_parameters[embedding_dimension_size];
float momentum_final_bias_parameters[embedding_dimension_size], velocity_final_bias_parameters[embedding_dimension_size];
float momentum_unembedding_projection[maximum_vocabulary_size][embedding_dimension_size], velocity_unembedding_projection[maximum_vocabulary_size][embedding_dimension_size];
float momentum_positional_embedding[context_length_limit][embedding_dimension_size], velocity_positional_embedding[context_length_limit][embedding_dimension_size];

// ============================================================================
// SELF-CONTAINED LOCAL RANDOM NUMBER GENERATOR
// ============================================================================
static inline int next_random(void) {
    static unsigned int state = 42;
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (int)(state & 0x7FFFFFFF);
}

static inline float sample_uniform_range(float minimum_value, float maximum_value) {
    int raw_value = next_random();
    float normalized_value = (float)raw_value / (float)0x7FFFFFFF;
    return minimum_value + (maximum_value - minimum_value) * normalized_value;
}

// ============================================================================
// OPTIMIZER & ACTIVATION FUNCTIONS
// ============================================================================
void execute_adamw_optimizer_step(float learning_rate, int current_step, float beta_one = 0.9f, float beta_two = 0.999f, float epsilon = 1e-8f, float weight_decay = 0.01f)
{
    float bias_correction_one = 1.0f - powf(beta_one, current_step + 1);
    float bias_correction_two = 1.0f - powf(beta_two, current_step + 1);

    #define APPLY_ADAMW_UPDATE(weight_array, gradient_array, momentum_array, velocity_array, total_size) \
        for (int index = 0; index < (total_size); index++) { \
            float* weight_pointer    = ((float*)(weight_array)) + index; \
            float* gradient_pointer  = ((float*)(gradient_array)) + index; \
            float* momentum_pointer  = ((float*)(momentum_array)) + index; \
            float* velocity_pointer  = ((float*)(velocity_array)) + index; \
            *weight_pointer -= learning_rate * weight_decay * (*weight_pointer); \
            *momentum_pointer = beta_one * (*momentum_pointer) + (1.0f - beta_one) * (*gradient_pointer); \
            *velocity_pointer = beta_two * (*velocity_pointer) + (1.0f - beta_two) * (*gradient_pointer) * (*gradient_pointer); \
            float momentum_hat = *momentum_pointer / bias_correction_one; \
            float velocity_hat = *velocity_pointer / bias_correction_two; \
            *weight_pointer -= learning_rate * momentum_hat / (sqrtf(velocity_hat) + epsilon); \
        }

    APPLY_ADAMW_UPDATE(token_embedding_parameters, token_embedding_gradient_buffer, momentum_token_embedding, velocity_token_embedding, current_vocabulary_size * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(positional_embedding_parameters, positional_embedding_gradient_buffer, momentum_positional_embedding, velocity_positional_embedding, context_length_limit * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(query_projection_weight_matrix, query_projection_weight_gradient_matrix, momentum_query_projection, velocity_query_projection, total_transformer_layers * attention_head_count * attention_head_dimension * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(key_projection_weight_matrix, key_projection_weight_gradient_matrix, momentum_key_projection, velocity_key_projection, total_transformer_layers * attention_head_count * attention_head_dimension * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(value_projection_weight_matrix, value_projection_weight_gradient_matrix, momentum_value_projection, velocity_value_projection, total_transformer_layers * attention_head_count * attention_head_dimension * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(output_projection_weight_matrix, output_projection_weight_gradient_matrix, momentum_output_projection, velocity_output_projection, total_transformer_layers * embedding_dimension_size * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(feed_forward_expansion_weight_matrix, feed_forward_expansion_weight_gradient_matrix, momentum_feed_forward_expansion, velocity_feed_forward_expansion, total_transformer_layers * feed_forward_hidden_dimension_size * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(feed_forward_contraction_weight_matrix, feed_forward_contraction_weight_gradient_matrix, momentum_feed_forward_contraction, velocity_feed_forward_contraction, total_transformer_layers * embedding_dimension_size * feed_forward_hidden_dimension_size);
    APPLY_ADAMW_UPDATE(attention_layer_normalization_scale_parameters, attention_layer_normalization_scale_gradient_parameters, momentum_attention_scale_parameters, velocity_attention_scale_parameters, total_transformer_layers * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(attention_layer_normalization_bias_parameters, attention_layer_normalization_bias_gradient_parameters, momentum_attention_bias_parameters, velocity_attention_bias_parameters, total_transformer_layers * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(feed_forward_layer_normalization_scale_parameters, feed_forward_layer_normalization_scale_gradient_parameters, momentum_feed_forward_scale_parameters, velocity_feed_forward_scale_parameters, total_transformer_layers * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(feed_forward_layer_normalization_bias_parameters, feed_forward_layer_normalization_bias_gradient_parameters, momentum_feed_forward_bias_parameters, velocity_feed_forward_bias_parameters, total_transformer_layers * embedding_dimension_size);
    APPLY_ADAMW_UPDATE(final_layer_normalization_scale_parameters, final_layer_normalization_scale_gradient_parameters, momentum_final_scale_parameters, velocity_final_scale_parameters, embedding_dimension_size);
    APPLY_ADAMW_UPDATE(final_layer_normalization_bias_parameters, final_layer_normalization_bias_gradient_parameters, momentum_final_bias_parameters, velocity_final_bias_parameters, embedding_dimension_size);
    APPLY_ADAMW_UPDATE(unembedding_projection_weight_matrix, unembedding_projection_weight_gradient_matrix, momentum_unembedding_projection, velocity_unembedding_projection, current_vocabulary_size * embedding_dimension_size);

    #undef APPLY_ADAMW_UPDATE
}

inline float compute_gelu_activation(float input_value) {
    return 0.5f * input_value * (1.0f + tanhf(0.7978845608f * (input_value + 0.044715f * input_value * input_value)));
}

inline float compute_gelu_gradient(float input_value) {
    const float alpha_constant = 0.7978845608f;
    const float beta_constant  = 0.044715f;

    float input_squared = input_value * input_value;
    float tanh_argument = alpha_constant * (input_value + beta_constant * input_squared);
    float tanh_value = tanhf(tanh_argument);

    float left_term = 0.5f * (1.0f + tanh_value);
    float right_term = 0.5f * input_value * (1.0f - tanh_value * tanh_value) *
                       alpha_constant * (1.0f + 2.0f * beta_constant * input_value);

    return left_term + right_term;
}

float calculate_learning_rate_schedule(int current_step, int total_steps) {
    if (current_step < learning_rate_warmup_steps) {
        return maximum_learning_rate * ((float)current_step / learning_rate_warmup_steps);
    }
    float training_progress = (float)(current_step - learning_rate_warmup_steps) / (total_steps - learning_rate_warmup_steps);
    return minimum_learning_rate + 0.5f * (maximum_learning_rate - minimum_learning_rate) * (1.0f + cosf(training_progress * 3.14159265f));
}

// ============================================================================
// WEIGHT INITIALIZATION & LAYER NORMALIZATION
// ============================================================================
void initialize_model_weights()
{
    auto fill_random_values = [&](float* pointer, int size) { 
        for (int index = 0; index < size; index++) {
            pointer[index] = sample_uniform_range(-0.1f, 0.1f);
        } 
    };
    auto fill_constant_values = [&](float* pointer, int size, float value) { for (int index = 0; index < size; index++) pointer[index] = value; };

    fill_random_values(&token_embedding_parameters[0][0], current_vocabulary_size * embedding_dimension_size);
    fill_random_values(&positional_embedding_parameters[0][0], context_length_limit * embedding_dimension_size);

    for (int layer_index = 0; layer_index < total_transformer_layers; layer_index++) {
        for (int head_index = 0; head_index < attention_head_count; head_index++) {
            fill_random_values(&query_projection_weight_matrix[layer_index][head_index][0][0], attention_head_dimension * embedding_dimension_size);
            fill_random_values(&key_projection_weight_matrix[layer_index][head_index][0][0], attention_head_dimension * embedding_dimension_size);
            fill_random_values(&value_projection_weight_matrix[layer_index][head_index][0][0], attention_head_dimension * embedding_dimension_size);
        }
        fill_random_values(&output_projection_weight_matrix[layer_index][0][0], embedding_dimension_size * embedding_dimension_size);
        fill_random_values(&feed_forward_expansion_weight_matrix[layer_index][0][0], feed_forward_hidden_dimension_size * embedding_dimension_size);
        fill_random_values(&feed_forward_contraction_weight_matrix[layer_index][0][0], embedding_dimension_size * feed_forward_hidden_dimension_size);

        fill_constant_values(&attention_layer_normalization_scale_parameters[layer_index][0], embedding_dimension_size, 1.0f);
        fill_constant_values(&attention_layer_normalization_bias_parameters[layer_index][0], embedding_dimension_size, 0.0f);
        fill_constant_values(&feed_forward_layer_normalization_scale_parameters[layer_index][0], embedding_dimension_size, 1.0f);
        fill_constant_values(&feed_forward_layer_normalization_bias_parameters[layer_index][0], embedding_dimension_size, 0.0f);
    }
    fill_constant_values(final_layer_normalization_scale_parameters, embedding_dimension_size, 1.0f);
    fill_constant_values(final_layer_normalization_bias_parameters, embedding_dimension_size, 0.0f);
    fill_random_values(&unembedding_projection_weight_matrix[0][0], current_vocabulary_size * embedding_dimension_size);

    printf("Dynamic model weights initialized successfully.\n");
}

void execute_layer_normalization_forward(const float* input_vector, const float* scale_parameters, const float* bias_parameters,
                                         float* output_vector, float* normalized_hat_vector, float* inverse_standard_deviation_output)
{
    float mean_value = 0.0f;
    for (int index = 0; index < embedding_dimension_size; index++) mean_value += input_vector[index];
    mean_value /= embedding_dimension_size;

    float variance_value = 0.0f;
    for (int index = 0; index < embedding_dimension_size; index++) {
        float difference = input_vector[index] - mean_value;
        variance_value += difference * difference;
    }
    variance_value /= embedding_dimension_size;

    float inverse_standard_deviation = 1.0f / sqrtf(variance_value + 1e-5f);
    *inverse_standard_deviation_output = inverse_standard_deviation;

    for (int index = 0; index < embedding_dimension_size; index++) {
        normalized_hat_vector[index] = (input_vector[index] - mean_value) * inverse_standard_deviation;
        output_vector[index] = normalized_hat_vector[index] * scale_parameters[index] + bias_parameters[index];
    }
}

void execute_layer_normalization_backward(const float* gradient_output_vector, const float* normalized_hat_vector, float inverse_standard_deviation,
                                          const float* scale_parameters, float* gradient_input_vector,
                                          float* gradient_scale_parameters, float* gradient_bias_parameters)
{
    for (int index = 0; index < embedding_dimension_size; index++) {
        gradient_scale_parameters[index] += gradient_output_vector[index] * normalized_hat_vector[index];
        gradient_bias_parameters[index]  += gradient_output_vector[index];
    }
    float gradient_normalized_hat[embedding_dimension_size];
    for (int index = 0; index < embedding_dimension_size; index++) gradient_normalized_hat[index] = gradient_output_vector[index] * scale_parameters[index];

    float mean_gradient_normalized_hat = 0.0f;
    for (int index = 0; index < embedding_dimension_size; index++) mean_gradient_normalized_hat += gradient_normalized_hat[index];
    mean_gradient_normalized_hat /= embedding_dimension_size;

    float mean_gradient_normalized_hat_times_hat = 0.0f;
    for (int index = 0; index < embedding_dimension_size; index++) mean_gradient_normalized_hat_times_hat += gradient_normalized_hat[index] * normalized_hat_vector[index];
    mean_gradient_normalized_hat_times_hat /= embedding_dimension_size;

    for (int index = 0; index < embedding_dimension_size; index++)
        gradient_input_vector[index] = inverse_standard_deviation * (gradient_normalized_hat[index] - mean_gradient_normalized_hat - normalized_hat_vector[index] * mean_gradient_normalized_hat_times_hat);
}

void reset_all_gradients()
{
    memset(token_embedding_gradient_buffer, 0, sizeof(token_embedding_gradient_buffer));
    memset(positional_embedding_gradient_buffer, 0, sizeof(positional_embedding_gradient_buffer));
    memset(query_projection_weight_gradient_matrix, 0, sizeof(query_projection_weight_gradient_matrix));
    memset(key_projection_weight_gradient_matrix, 0, sizeof(key_projection_weight_gradient_matrix));
    memset(value_projection_weight_gradient_matrix, 0, sizeof(value_projection_weight_gradient_matrix));
    memset(output_projection_weight_gradient_matrix, 0, sizeof(output_projection_weight_gradient_matrix));
    memset(feed_forward_expansion_weight_gradient_matrix, 0, sizeof(feed_forward_expansion_weight_gradient_matrix));
    memset(feed_forward_contraction_weight_gradient_matrix, 0, sizeof(feed_forward_contraction_weight_gradient_matrix));
    memset(attention_layer_normalization_scale_gradient_parameters, 0, sizeof(attention_layer_normalization_scale_gradient_parameters));
    memset(attention_layer_normalization_bias_gradient_parameters, 0, sizeof(attention_layer_normalization_bias_gradient_parameters));
    memset(feed_forward_layer_normalization_scale_gradient_parameters, 0, sizeof(feed_forward_layer_normalization_scale_gradient_parameters));
    memset(feed_forward_layer_normalization_bias_gradient_parameters, 0, sizeof(feed_forward_layer_normalization_bias_gradient_parameters));
    memset(final_layer_normalization_scale_gradient_parameters, 0, sizeof(final_layer_normalization_scale_gradient_parameters));
    memset(final_layer_normalization_bias_gradient_parameters, 0, sizeof(final_layer_normalization_bias_gradient_parameters));
    memset(unembedding_projection_weight_gradient_matrix, 0, sizeof(unembedding_projection_weight_gradient_matrix));
}

// ============================================================================
// FORWARD PASS
// ============================================================================
float execute_forward_pass(int* token_indices, int* target_indices, int sequence_length)
{
    for (int position = 0; position < sequence_length; position++)
        for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
            residual_stream_buffer[position][dimension] = token_embedding_parameters[token_indices[position]][dimension] + positional_embedding_parameters[position][dimension];

    float attention_scaling_factor = 1.0f / sqrtf((float)attention_head_dimension);

    for (int layer_index = 0; layer_index < total_transformer_layers; layer_index++)
    {
        for (int position = 0; position < sequence_length; position++)
            execute_layer_normalization_forward(residual_stream_buffer[position], attention_layer_normalization_scale_parameters[layer_index], attention_layer_normalization_bias_parameters[layer_index],
                                             saved_attention_normalized_values[layer_index][position], saved_attention_normalized_hat[layer_index][position], &saved_attention_inverse_standard_deviation[layer_index][position]);

        for (int head_index = 0; head_index < attention_head_count; head_index++) {
            for (int position = 0; position < sequence_length; position++) {
                for (int head_dimension_index = 0; head_dimension_index < attention_head_dimension; head_dimension_index++) {
                    float query_accumulator = 0, key_accumulator = 0, value_accumulator = 0;
                    const float* normalized_pointer = saved_attention_normalized_values[layer_index][position];
                    for (int dimension = 0; dimension < embedding_dimension_size; dimension++) {
                        query_accumulator += query_projection_weight_matrix[layer_index][head_index][head_dimension_index][dimension] * normalized_pointer[dimension];
                        key_accumulator   += key_projection_weight_matrix[layer_index][head_index][head_dimension_index][dimension] * normalized_pointer[dimension];
                        value_accumulator += value_projection_weight_matrix[layer_index][head_index][head_dimension_index][dimension] * normalized_pointer[dimension];
                    }
                    saved_query_vectors[layer_index][head_index][position][head_dimension_index] = query_accumulator;
                    saved_key_vectors[layer_index][head_index][position][head_dimension_index] = key_accumulator;
                    saved_value_vectors[layer_index][head_index][position][head_dimension_index] = value_accumulator;
                }
            }
        }

        for (int position = 0; position < sequence_length; position++)
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                saved_attention_output[layer_index][position][dimension] = 0.0f;

        for (int head_index = 0; head_index < attention_head_count; head_index++) {
            int head_offset = head_index * attention_head_dimension;
            for (int query_position = 0; query_position < sequence_length; query_position++) {
                float maximum_score_value = -1e30f;
                for (int key_position = 0; key_position <= query_position; key_position++) {
                    float dot_product_accumulator = 0.0f;
                    const float* query_pointer = saved_query_vectors[layer_index][head_index][query_position];
                    const float* key_pointer = saved_key_vectors[layer_index][head_index][key_position];
                    for (int head_dimension_index = 0; head_dimension_index < attention_head_dimension; head_dimension_index++)
                        dot_product_accumulator += query_pointer[head_dimension_index] * key_pointer[head_dimension_index];
                    float scaled_score = dot_product_accumulator * attention_scaling_factor;
                    saved_attention_weights[layer_index][head_index][query_position][key_position] = scaled_score;
                    if (scaled_score > maximum_score_value) maximum_score_value = scaled_score;
                }
                for (int key_position = query_position + 1; key_position < sequence_length; key_position++)
                    saved_attention_weights[layer_index][head_index][query_position][key_position] = -1e30f;

                float exponential_sum = 0.0f;
                for (int key_position = 0; key_position <= query_position; key_position++) {
                    float exponentiated_value = expf(saved_attention_weights[layer_index][head_index][query_position][key_position] - maximum_score_value);
                    saved_attention_weights[layer_index][head_index][query_position][key_position] = exponentiated_value;
                    exponential_sum += exponentiated_value;
                }
                float inverse_exponential_sum = 1.0f / exponential_sum;
                for (int key_position = 0; key_position <= query_position; key_position++) {
                    saved_attention_weights[layer_index][head_index][query_position][key_position] *= inverse_exponential_sum;
                }

                for (int key_position = 0; key_position <= query_position; key_position++) {
                    float attention_weight = saved_attention_weights[layer_index][head_index][query_position][key_position];
                    const float* value_pointer = saved_value_vectors[layer_index][head_index][key_position];
                    float* output_pointer = &saved_attention_output[layer_index][query_position][head_offset];
                    for (int head_dimension_index = 0; head_dimension_index < attention_head_dimension; head_dimension_index++)
                        output_pointer[head_dimension_index] += attention_weight * value_pointer[head_dimension_index];
                }
            }
        }

        for (int position = 0; position < sequence_length; position++) {
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++) {
                float projection_accumulator = 0.0f;
                const float* attention_output_pointer = saved_attention_output[layer_index][position];
                const float* output_weight_pointer = output_projection_weight_matrix[layer_index][dimension];
                for (int inner_dimension = 0; inner_dimension < embedding_dimension_size; inner_dimension++)
                    projection_accumulator += output_weight_pointer[inner_dimension] * attention_output_pointer[inner_dimension];
                residual_stream_buffer[position][dimension] += projection_accumulator;
            }
        }

        for (int position = 0; position < sequence_length; position++)
            execute_layer_normalization_forward(residual_stream_buffer[position], feed_forward_layer_normalization_scale_parameters[layer_index], feed_forward_layer_normalization_bias_parameters[layer_index],
                                             saved_feed_forward_normalized_values[layer_index][position], saved_feed_forward_normalized_hat[layer_index][position], &saved_feed_forward_inverse_standard_deviation[layer_index][position]);

        for (int position = 0; position < sequence_length; position++) {
            for (int hidden_index = 0; hidden_index < feed_forward_hidden_dimension_size; hidden_index++) {
                float expansion_accumulator = 0.0f;
                const float* normalized_pointer = saved_feed_forward_normalized_values[layer_index][position];
                const float* expansion_weight_pointer = feed_forward_expansion_weight_matrix[layer_index][hidden_index];
                for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                    expansion_accumulator += expansion_weight_pointer[dimension] * normalized_pointer[dimension];
                saved_feed_forward_pre_activation[layer_index][position][hidden_index] = expansion_accumulator; 
            }
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++) {
                float contraction_accumulator = 0.0f;
                const float* contraction_weight_pointer = feed_forward_contraction_weight_matrix[layer_index][dimension];
                for (int hidden_index = 0; hidden_index < feed_forward_hidden_dimension_size; hidden_index++) {
                    float gelu_activation_output = compute_gelu_activation(saved_feed_forward_pre_activation[layer_index][position][hidden_index]);
                    contraction_accumulator += contraction_weight_pointer[hidden_index] * gelu_activation_output;
                }
                residual_stream_buffer[position][dimension] += contraction_accumulator;
            }
        }
    }

    for (int position = 0; position < sequence_length; position++) {
        execute_layer_normalization_forward(residual_stream_buffer[position], final_layer_normalization_scale_parameters, final_layer_normalization_bias_parameters,
                                         saved_final_normalized_values[position], saved_final_normalized_hat[position], &saved_final_inverse_standard_deviation[position]);

        for (int vocabulary_index = 0; vocabulary_index < current_vocabulary_size; vocabulary_index++) {
            float logit_accumulator = 0.0f;
            const float* unembedding_pointer = unembedding_projection_weight_matrix[vocabulary_index];
            const float* final_normalized_pointer = saved_final_normalized_values[position];
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                logit_accumulator += unembedding_pointer[dimension] * final_normalized_pointer[dimension];
            vocabulary_logits_buffer[position][vocabulary_index] = logit_accumulator;
        }
    }

    float total_cross_entropy_loss = 0.0f;
    for (int position = 0; position < sequence_length; position++) {
        float maximum_logit = vocabulary_logits_buffer[position][0];
        for (int vocabulary_index = 1; vocabulary_index < current_vocabulary_size; vocabulary_index++)
            if (vocabulary_logits_buffer[position][vocabulary_index] > maximum_logit) maximum_logit = vocabulary_logits_buffer[position][vocabulary_index];
        float exponential_sum = 0.0f;
        for (int vocabulary_index = 0; vocabulary_index < current_vocabulary_size; vocabulary_index++) exponential_sum += expf(vocabulary_logits_buffer[position][vocabulary_index] - maximum_logit);
        total_cross_entropy_loss -= (vocabulary_logits_buffer[position][target_indices[position]] - (maximum_logit + logf(exponential_sum)));
    }
    return total_cross_entropy_loss / sequence_length;
}

// ============================================================================
// BACKWARD PASS
// ============================================================================
void execute_backward_pass(int* token_indices, int* target_indices, int sequence_length)
{
    float attention_scaling_factor = 1.0f / sqrtf((float)attention_head_dimension);

    for (int position = 0; position < sequence_length; position++) {
        float maximum_logit = vocabulary_logits_buffer[position][0];
        for (int vocabulary_index = 1; vocabulary_index < current_vocabulary_size; vocabulary_index++)
            if (vocabulary_logits_buffer[position][vocabulary_index] > maximum_logit) maximum_logit = vocabulary_logits_buffer[position][vocabulary_index];
        float exponential_sum = 0.0f;
        for (int vocabulary_index = 0; vocabulary_index < current_vocabulary_size; vocabulary_index++) {
            logits_gradient_buffer[position][vocabulary_index] = expf(vocabulary_logits_buffer[position][vocabulary_index] - maximum_logit);
            exponential_sum += logits_gradient_buffer[position][vocabulary_index];
        }
        for (int vocabulary_index = 0; vocabulary_index < current_vocabulary_size; vocabulary_index++) {
            logits_gradient_buffer[position][vocabulary_index] /= exponential_sum;
            if (vocabulary_index == target_indices[position]) logits_gradient_buffer[position][vocabulary_index] -= 1.0f;
            logits_gradient_buffer[position][vocabulary_index] /= sequence_length;
        }
    }

    float local_final_normalized_gradient[context_length_limit][embedding_dimension_size];
    memset(local_final_normalized_gradient, 0, sizeof(local_final_normalized_gradient));
    for (int position = 0; position < sequence_length; position++)
        for (int vocabulary_index = 0; vocabulary_index < current_vocabulary_size; vocabulary_index++) {
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++) {
                local_final_normalized_gradient[position][dimension] += logits_gradient_buffer[position][vocabulary_index] * unembedding_projection_weight_matrix[vocabulary_index][dimension];
                unembedding_projection_weight_gradient_matrix[vocabulary_index][dimension] += logits_gradient_buffer[position][vocabulary_index] * saved_final_normalized_values[position][dimension];
            }
        }

    memset(residual_gradient_buffer, 0, sizeof(residual_gradient_buffer));
    for (int position = 0; position < sequence_length; position++)
        execute_layer_normalization_backward(local_final_normalized_gradient[position], saved_final_normalized_hat[position], saved_final_inverse_standard_deviation[position],
                                          final_layer_normalization_scale_parameters, residual_gradient_buffer[position], final_layer_normalization_scale_gradient_parameters, final_layer_normalization_bias_parameters);

    for (int layer_index = total_transformer_layers - 1; layer_index >= 0; layer_index--)
    {
        for (int position = 0; position < sequence_length; position++) {
            float hidden_gradient_output[feed_forward_hidden_dimension_size];
            for (int hidden_index = 0; hidden_index < feed_forward_hidden_dimension_size; hidden_index++) {
                float gradient_accumulator = 0.0f;
                for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                    gradient_accumulator += residual_gradient_buffer[position][dimension] * feed_forward_contraction_weight_matrix[layer_index][dimension][hidden_index];
                hidden_gradient_output[hidden_index] = gradient_accumulator;
            }

            for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                for (int hidden_index = 0; hidden_index < feed_forward_hidden_dimension_size; hidden_index++) {
                    float gelu_evaluated_value = compute_gelu_activation(saved_feed_forward_pre_activation[layer_index][position][hidden_index]);
                    feed_forward_contraction_weight_gradient_matrix[layer_index][dimension][hidden_index] += residual_gradient_buffer[position][dimension] * gelu_evaluated_value;
                }

            float pre_gelu_gradient[feed_forward_hidden_dimension_size];
            for (int hidden_index = 0; hidden_index < feed_forward_hidden_dimension_size; hidden_index++)
                pre_gelu_gradient[hidden_index] = hidden_gradient_output[hidden_index] * compute_gelu_gradient(saved_feed_forward_pre_activation[layer_index][position][hidden_index]);

            float normalized_feed_forward_gradient_position[embedding_dimension_size];
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++) {
                float gradient_accumulator = 0.0f;
                for (int hidden_index = 0; hidden_index < feed_forward_hidden_dimension_size; hidden_index++)
                    gradient_accumulator += pre_gelu_gradient[hidden_index] * feed_forward_expansion_weight_matrix[layer_index][hidden_index][dimension];
                normalized_feed_forward_gradient_position[dimension] = gradient_accumulator;
            }

            for (int hidden_index = 0; hidden_index < feed_forward_hidden_dimension_size; hidden_index++)
                for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                    feed_forward_expansion_weight_gradient_matrix[layer_index][hidden_index][dimension] += pre_gelu_gradient[hidden_index] * saved_feed_forward_normalized_values[layer_index][position][dimension];

            float residual_from_feed_forward_normalization[embedding_dimension_size];
            execute_layer_normalization_backward(normalized_feed_forward_gradient_position, saved_feed_forward_normalized_hat[layer_index][position], saved_feed_forward_inverse_standard_deviation[layer_index][position],
                                              feed_forward_layer_normalization_scale_parameters[layer_index], residual_from_feed_forward_normalization, feed_forward_layer_normalization_scale_gradient_parameters[layer_index], feed_forward_layer_normalization_bias_parameters[layer_index]);

            for (int dimension = 0; dimension < embedding_dimension_size; dimension++) residual_gradient_buffer[position][dimension] += residual_from_feed_forward_normalization[dimension];
        }

        memset(attention_output_gradient_buffer, 0, sizeof(attention_output_gradient_buffer));
        for (int position = 0; position < sequence_length; position++) {
            for (int inner_dimension = 0; inner_dimension < embedding_dimension_size; inner_dimension++) {
                float gradient_accumulator = 0.0f;
                for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                    gradient_accumulator += residual_gradient_buffer[position][dimension] * output_projection_weight_matrix[layer_index][dimension][inner_dimension];
                attention_output_gradient_buffer[position][inner_dimension] = gradient_accumulator;
            }
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
                for (int inner_dimension = 0; inner_dimension < embedding_dimension_size; inner_dimension++)
                    output_projection_weight_gradient_matrix[layer_index][dimension][inner_dimension] += residual_gradient_buffer[position][dimension] * saved_attention_output[layer_index][position][inner_dimension];
        }

        memset(query_gradient_buffer, 0, sizeof(query_gradient_buffer));
        memset(key_gradient_buffer, 0, sizeof(key_gradient_buffer));
        memset(value_gradient_buffer, 0, sizeof(value_gradient_buffer));

        for (int head_index = 0; head_index < attention_head_count; head_index++) {
            int head_offset = head_index * attention_head_dimension;
            for (int query_position = 0; query_position < sequence_length; query_position++) {
                float attention_weight_gradients[context_length_limit];
                memset(attention_weight_gradients, 0, sizeof(attention_weight_gradients));
                for (int key_position = 0; key_position <= query_position; key_position++)
                    for (int head_dimension_index = 0; head_dimension_index < attention_head_dimension; head_dimension_index++) {
                        attention_weight_gradients[key_position] += attention_output_gradient_buffer[query_position][head_offset + head_dimension_index] * saved_value_vectors[layer_index][head_index][key_position][head_dimension_index];
                        value_gradient_buffer[head_index][key_position][head_dimension_index] += saved_attention_weights[layer_index][head_index][query_position][key_position] * attention_output_gradient_buffer[query_position][head_offset + head_dimension_index];
                    }

                float dot_product_accumulator = 0.0f;
                for (int key_position = 0; key_position <= query_position; key_position++)
                    dot_product_accumulator += attention_weight_gradients[key_position] * saved_attention_weights[layer_index][head_index][query_position][key_position];

                float score_gradients[context_length_limit];
                for (int key_position = 0; key_position <= query_position; key_position++)
                    score_gradients[key_position] = saved_attention_weights[layer_index][head_index][query_position][key_position] * (attention_weight_gradients[key_position] - dot_product_accumulator);

                for (int key_position = 0; key_position <= query_position; key_position++)
                    for (int head_dimension_index = 0; head_dimension_index < attention_head_dimension; head_dimension_index++) {
                        query_gradient_buffer[head_index][query_position][head_dimension_index] += score_gradients[key_position] * attention_scaling_factor * saved_key_vectors[layer_index][head_index][key_position][head_dimension_index];
                        key_gradient_buffer[head_index][key_position][head_dimension_index] += score_gradients[key_position] * attention_scaling_factor * saved_query_vectors[layer_index][head_index][query_position][head_dimension_index];
                    }
            }
        }

        memset(normalized_buffer, 0, sizeof(normalized_buffer));
        for (int head_index = 0; head_index < attention_head_count; head_index++)
            for (int position = 0; position < sequence_length; position++)
                for (int head_dimension_index = 0; head_dimension_index < attention_head_dimension; head_dimension_index++)
                    for (int dimension = 0; dimension < embedding_dimension_size; dimension++) {
                        normalized_buffer[position][dimension] += query_gradient_buffer[head_index][position][head_dimension_index] * query_projection_weight_matrix[layer_index][head_index][head_dimension_index][dimension];
                        normalized_buffer[position][dimension] += key_gradient_buffer[head_index][position][head_dimension_index] * key_projection_weight_matrix[layer_index][head_index][head_dimension_index][dimension];
                        normalized_buffer[position][dimension] += value_gradient_buffer[head_index][position][head_dimension_index] * value_projection_weight_matrix[layer_index][head_index][head_dimension_index][dimension];
                        query_projection_weight_gradient_matrix[layer_index][head_index][head_dimension_index][dimension] += query_gradient_buffer[head_index][position][head_dimension_index] * saved_attention_normalized_values[layer_index][position][dimension];
                        key_projection_weight_gradient_matrix[layer_index][head_index][head_dimension_index][dimension] += key_gradient_buffer[head_index][position][head_dimension_index] * saved_attention_normalized_values[layer_index][position][dimension];
                        value_projection_weight_gradient_matrix[layer_index][head_index][head_dimension_index][dimension] += value_gradient_buffer[head_index][position][head_dimension_index] * saved_attention_normalized_values[layer_index][position][dimension];
                    }

        for (int position = 0; position < sequence_length; position++) {
            float residual_from_attention_normalization[embedding_dimension_size];
            execute_layer_normalization_backward(normalized_buffer[position], saved_attention_normalized_hat[layer_index][position], saved_attention_inverse_standard_deviation[layer_index][position],
                                              attention_layer_normalization_scale_parameters[layer_index], residual_from_attention_normalization, attention_layer_normalization_scale_gradient_parameters[layer_index], attention_layer_normalization_bias_parameters[layer_index]);
            for (int dimension = 0; dimension < embedding_dimension_size; dimension++) residual_gradient_buffer[position][dimension] += residual_from_attention_normalization[dimension];
        }
    }

    for (int position = 0; position < sequence_length; position++)
        for (int dimension = 0; dimension < embedding_dimension_size; dimension++)
        {
            token_embedding_gradient_buffer[token_indices[position]][dimension] += residual_gradient_buffer[position][dimension];
            positional_embedding_gradient_buffer[position][dimension] += residual_gradient_buffer[position][dimension];
        }
}

// ============================================================================
// TOKENIZATION & CORPUS PARSING
// ============================================================================
std::vector<int> convert_prompt_to_token_vector(const std::string& prompt_string, const std::map<std::string, int>& word_to_id_mapping)
{
    std::vector<int> prompt_token_ids;
    std::stringstream string_stream(prompt_string);
    std::string raw_word;

    while (string_stream >> raw_word) {
        std::string current_word = "";
        for (char character : raw_word) {
            if (ispunct(character) && character != '\'' && character != '-') {
                if (!current_word.empty()) {
                    if (word_to_id_mapping.find(current_word) != word_to_id_mapping.end()) {
                        prompt_token_ids.push_back(word_to_id_mapping.at(current_word));
                    }
                    current_word = "";
                }
                std::string punctuation_string(1, character);
                if (word_to_id_mapping.find(punctuation_string) != word_to_id_mapping.end()) {
                    prompt_token_ids.push_back(word_to_id_mapping.at(punctuation_string));
                }
            } else {
                current_word += tolower(character);
            }
        }
        if (!current_word.empty()) {
            if (word_to_id_mapping.find(current_word) != word_to_id_mapping.end()) {
                prompt_token_ids.push_back(word_to_id_mapping.at(current_word));
            }
        }
    }
    return prompt_token_ids;
}

void save_model_weights_to_disk(const std::string& filename)
{
    std::ofstream file_stream(filename, std::ios::binary);
    if (!file_stream.is_open()) {
        printf("Error: Could not open file '%s' for saving weights.\n", filename.c_str());
        return;
    }

    file_stream.write((char*)token_embedding_parameters, sizeof(token_embedding_parameters));
    file_stream.write((char*)positional_embedding_parameters, sizeof(positional_embedding_parameters));
    file_stream.write((char*)query_projection_weight_matrix, sizeof(query_projection_weight_matrix));
    file_stream.write((char*)key_projection_weight_matrix, sizeof(key_projection_weight_matrix));
    file_stream.write((char*)value_projection_weight_matrix, sizeof(value_projection_weight_matrix));
    file_stream.write((char*)output_projection_weight_matrix, sizeof(output_projection_weight_matrix));
    file_stream.write((char*)feed_forward_expansion_weight_matrix, sizeof(feed_forward_expansion_weight_matrix));
    file_stream.write((char*)feed_forward_contraction_weight_matrix, sizeof(feed_forward_contraction_weight_matrix));
    file_stream.write((char*)attention_layer_normalization_scale_parameters, sizeof(attention_layer_normalization_scale_parameters));
    file_stream.write((char*)attention_layer_normalization_bias_parameters, sizeof(attention_layer_normalization_bias_parameters));
    file_stream.write((char*)feed_forward_layer_normalization_scale_parameters, sizeof(feed_forward_layer_normalization_scale_parameters));
    file_stream.write((char*)feed_forward_layer_normalization_bias_parameters, sizeof(feed_forward_layer_normalization_bias_parameters));
    file_stream.write((char*)final_layer_normalization_scale_parameters, sizeof(final_layer_normalization_scale_parameters));
    file_stream.write((char*)final_layer_normalization_bias_parameters, sizeof(final_layer_normalization_bias_parameters));
    file_stream.write((char*)unembedding_projection_weight_matrix, sizeof(unembedding_projection_weight_matrix));

    file_stream.close();
    printf("Model successfully saved to binary file '%s'.\n", filename.c_str());
}

void load_model_weights_from_disk(const std::string& filename)
{
    std::ifstream file_stream(filename, std::ios::binary);
    if (!file_stream.is_open()) {
        printf("Error: Could not open file '%s' for loading weights.\n", filename.c_str());
        return;
    }

    file_stream.read((char*)token_embedding_parameters, sizeof(token_embedding_parameters));
    file_stream.read((char*)positional_embedding_parameters, sizeof(positional_embedding_parameters));
    file_stream.read((char*)query_projection_weight_matrix, sizeof(query_projection_weight_matrix));
    file_stream.read((char*)key_projection_weight_matrix, sizeof(key_projection_weight_matrix));
    file_stream.read((char*)value_projection_weight_matrix, sizeof(value_projection_weight_matrix));
    file_stream.read((char*)output_projection_weight_matrix, sizeof(output_projection_weight_matrix));
    file_stream.read((char*)feed_forward_expansion_weight_matrix, sizeof(feed_forward_expansion_weight_matrix));
    file_stream.read((char*)feed_forward_contraction_weight_matrix, sizeof(feed_forward_contraction_weight_matrix));
    file_stream.read((char*)attention_layer_normalization_scale_parameters, sizeof(attention_layer_normalization_scale_parameters));
    file_stream.read((char*)attention_layer_normalization_bias_parameters, sizeof(attention_layer_normalization_bias_parameters));
    file_stream.read((char*)feed_forward_layer_normalization_scale_parameters, sizeof(feed_forward_layer_normalization_scale_parameters));
    file_stream.read((char*)feed_forward_layer_normalization_bias_parameters, sizeof(feed_forward_layer_normalization_bias_parameters));
    file_stream.read((char*)final_layer_normalization_scale_parameters, sizeof(final_layer_normalization_scale_parameters));
    file_stream.read((char*)final_layer_normalization_bias_parameters, sizeof(final_layer_normalization_bias_parameters));
    file_stream.read((char*)unembedding_projection_weight_matrix, sizeof(unembedding_projection_weight_matrix));

    file_stream.close();
    printf("Model successfully loaded from binary file '%s'.\n", filename.c_str());
}

// ============================================================================
// INFERENCE & GENERATION
// ============================================================================
void generate_autoregressive_tokens(const int* prompt_token_ids, int prompt_length, int max_new_tokens)
{
    std::vector<int> current_generation_tokens;
    for (int index = 0; index < prompt_length && index < context_length_limit; index++) {
        current_generation_tokens.push_back(prompt_token_ids[index]);
    }

    printf("\n--- Generating Words ---\nPrompt: ");
    for (int token_id : current_generation_tokens) {
        printf("%s ", vocabulary_word_strings[token_id]);
    }
    printf("\nOutput: ");
    for (int token_id : current_generation_tokens) {
        printf("%s ", vocabulary_word_strings[token_id]);
    }

    const float generation_temperature = 0.7f;
    const int top_k_sampling_limit = 20;
    const float repetition_penalty_value = 1.2f;

    for (int step = 0; step < max_new_tokens; step++) {
        int current_sequence_length = (int)current_generation_tokens.size();
        if (current_sequence_length > context_length_limit) break;

        execute_forward_pass(current_generation_tokens.data(), current_generation_tokens.data(), current_sequence_length);
        int last_position_index = current_sequence_length - 1;

        int lookback_window_start = std::max(0, current_sequence_length - 15);
        for (int index = lookback_window_start; index < current_sequence_length; index++) {
            int recent_token_id = current_generation_tokens[index];
            if (recent_token_id > 0 && recent_token_id < current_vocabulary_size) {
                vocabulary_logits_buffer[last_position_index][recent_token_id] -= repetition_penalty_value;
            }
        }

        for (int vocabulary_index = 0; vocabulary_index < current_vocabulary_size; vocabulary_index++) {
            vocabulary_logits_buffer[last_position_index][vocabulary_index] /= generation_temperature;
        }

        float maximum_logit_value = vocabulary_logits_buffer[last_position_index][0];
        for (int vocabulary_index = 1; vocabulary_index < current_vocabulary_size; vocabulary_index++) {
            if (vocabulary_logits_buffer[last_position_index][vocabulary_index] > maximum_logit_value) maximum_logit_value = vocabulary_logits_buffer[last_position_index][vocabulary_index];
        }

        std::vector<std::pair<float, int>> probability_pairs(current_vocabulary_size);
        float exponential_sum = 0.0f;
        for (int vocabulary_index = 0; vocabulary_index < current_vocabulary_size; vocabulary_index++) {
            float exponential_value = expf(vocabulary_logits_buffer[last_position_index][vocabulary_index] - maximum_logit_value);
            probability_pairs[vocabulary_index] = {exponential_value, vocabulary_index};
            exponential_sum += exponential_value;
        }

        std::sort(probability_pairs.rbegin(), probability_pairs.rend());

        int actual_top_k = std::min(top_k_sampling_limit, current_vocabulary_size);
        float top_k_cumulative_sum = 0.0f;
        for (int index = 0; index < actual_top_k; index++) {
            top_k_cumulative_sum += probability_pairs[index].first;
        }

        float random_sample_value = sample_uniform_range(0.0f, top_k_cumulative_sum);
        float cumulative_probability_accumulator = 0.0f;
        int chosen_token_id = probability_pairs[0].second;

        for (int index = 0; index < actual_top_k; index++) {
            cumulative_probability_accumulator += probability_pairs[index].first;
            if (random_sample_value <= cumulative_probability_accumulator) {
                chosen_token_id = probability_pairs[index].second;
                break;
            }
        }

        std::string chosen_token_string = vocabulary_word_strings[chosen_token_id];
        if (chosen_token_string == "." || chosen_token_string == "," || chosen_token_string == "?" || chosen_token_string == "!" || chosen_token_string == ":" || chosen_token_string == ";")
        {
            printf("%s", chosen_token_string.c_str());
        }
        else
        {
            printf(" %s", chosen_token_string.c_str());
        }

        current_generation_tokens.push_back(chosen_token_id);

        if (chosen_token_string == "<|endoftext|>") {
            printf("\n[Reached end of text]\n");
            break;
        }
    }
    printf("\n------------------------\n");
}

void save_token_frequency_report(
    const std::string& model_file_name,
    const std::vector<std::pair<std::string, int>>& sorted_vocabulary_list,
    const std::map<std::string, int>& word_to_id_mapping)
{
    std::string report_file_path = model_file_name + std::string("_tokens.txt");
    std::ofstream output_file_stream(report_file_path);
    if (!output_file_stream.is_open()) {
        printf("Failed to create token report '%s'\n", report_file_path.c_str());
        return;
    }

    output_file_stream << "TOKEN_ID\tFREQUENCY\tWORD\n";
    output_file_stream << "0\tN/A\t<unk>\n";

    for (const auto& pair_item : sorted_vocabulary_list) {
        auto map_iterator = word_to_id_mapping.find(pair_item.first);
        if (map_iterator == word_to_id_mapping.end())
            continue;

        output_file_stream << map_iterator->second
                           << '\t'
                           << pair_item.second
                           << '\t'
                           << pair_item.first
                           << '\n';
    }

    output_file_stream.close();
    printf("Token report saved to '%s'\n", report_file_path.c_str());
}

bool load_and_tokenize_corpus_file(const std::string& filename, std::vector<int>& training_word_ids, std::map<std::string, int>& word_to_id_mapping)
{
    std::string raw_corpus_text = "";
    std::ifstream file_stream(filename);

    if (file_stream.is_open()) {
        std::string line_string;
        while (std::getline(file_stream, line_string)) {
            raw_corpus_text += line_string + " ";
        }
        file_stream.close();
        printf("Successfully loaded training corpus from '%s'.\n", filename.c_str());
    } else {
        printf("'%s' not found. Falling back to default story corpus.\n", filename.c_str());
        raw_corpus_text = "I walk down a wide road... <|endoftext|> Once upon a time...";
    }

    std::vector<std::string> parsed_tokens_list;
    std::map<std::string, int> token_frequency_map;
    std::string current_word_buffer = "";

    for (size_t index = 0; index < raw_corpus_text.size(); ) {
        unsigned char character = raw_corpus_text[index];

        if (raw_corpus_text.compare(index, 13, "<|endoftext|>") == 0) {
            if (!current_word_buffer.empty()) {
                parsed_tokens_list.push_back(current_word_buffer);
                token_frequency_map[current_word_buffer]++;
                current_word_buffer = "";
            }
            parsed_tokens_list.push_back("<|endoftext|>");
            token_frequency_map["<|endoftext|>"]++;
            index += 13;
            continue;
        }

        if (isspace(character)) {
            if (!current_word_buffer.empty()) {
                parsed_tokens_list.push_back(current_word_buffer);
                token_frequency_map[current_word_buffer]++;
                current_word_buffer = "";
            }
            index++;
            continue;
        }

        if (index + 2 < raw_corpus_text.size() && 
            (unsigned char)raw_corpus_text[index] == 0xE2 && 
            (unsigned char)raw_corpus_text[index+1] == 0x80 && 
            ((unsigned char)raw_corpus_text[index+2] == 0x9C || (unsigned char)raw_corpus_text[index+2] == 0x9D || 
             (unsigned char)raw_corpus_text[index+2] == 0x98 || (unsigned char)raw_corpus_text[index+2] == 0x99)) {
            
            if (!current_word_buffer.empty()) {
                parsed_tokens_list.push_back(current_word_buffer);
                token_frequency_map[current_word_buffer]++;
                current_word_buffer = "";
            }

            std::string quote_character_string = "\"";
            unsigned char quote_type = raw_corpus_text[index+2];
            if (quote_type == 0x98 || quote_type == 0x99) quote_character_string = "'";

            parsed_tokens_list.push_back(quote_character_string);
            token_frequency_map[quote_character_string]++;
            index += 3;
            continue;
        }

        if (character < 128 && ispunct(character) && character != '\'' && character != '-') {
            if (!current_word_buffer.empty()) {
                parsed_tokens_list.push_back(current_word_buffer);
                token_frequency_map[current_word_buffer]++;
                current_word_buffer = "";
            }
            std::string punctuation_string(1, character);
            parsed_tokens_list.push_back(punctuation_string);
            token_frequency_map[punctuation_string]++;
            index++;
            continue;
        }

        if (character >= 128) {
            size_t character_length = 1;
            if ((character & 0xE0) == 0xC0) character_length = 2;
            else if ((character & 0xF0) == 0xE0) character_length = 3;
            else if ((character & 0xF8) == 0xF0) character_length = 4;

            if (index + character_length <= raw_corpus_text.size()) {
                current_word_buffer += raw_corpus_text.substr(index, character_length);
                index += character_length;
            } else {
                current_word_buffer += character;
                index++;
            }
        } else {
            current_word_buffer += tolower(character);
            index++;
        }
    }

    if (!current_word_buffer.empty()) {
        parsed_tokens_list.push_back(current_word_buffer);
        token_frequency_map[current_word_buffer]++;
    }

    std::vector<std::pair<std::string, int>> sorted_vocabulary_list(token_frequency_map.begin(), token_frequency_map.end());
    std::sort(sorted_vocabulary_list.begin(), sorted_vocabulary_list.end(), [](const auto& pair_a, const auto& pair_b) {
        return pair_a.second > pair_b.second;
    });

    word_to_id_mapping["<unk>"] = 0;
    snprintf(vocabulary_word_strings[0], sizeof(vocabulary_word_strings[0]), "%s", "<unk>");
    current_vocabulary_size = 1;

    bool has_end_of_text_token = false;
    for (const auto& pair_item : sorted_vocabulary_list) {
        if (pair_item.first == "<|endoftext|>") {
            has_end_of_text_token = true;
            break;
        }
    }

    if (has_end_of_text_token && current_vocabulary_size < maximum_vocabulary_size) {
        word_to_id_mapping["<|endoftext|>"] = current_vocabulary_size;
        snprintf(vocabulary_word_strings[current_vocabulary_size], sizeof(vocabulary_word_strings[current_vocabulary_size]), "%s", "<|endoftext|>");
        current_vocabulary_size++;
    }

    for (const auto& pair_item : sorted_vocabulary_list) {
        if (pair_item.first == "<|endoftext|>") continue;

        if (current_vocabulary_size < maximum_vocabulary_size) {
            word_to_id_mapping[pair_item.first] = current_vocabulary_size;
            snprintf(vocabulary_word_strings[current_vocabulary_size], sizeof(vocabulary_word_strings[current_vocabulary_size]), "%s", pair_item.first.c_str());
            current_vocabulary_size++;
        } else {
            break;
        }
    }

    for (const std::string& token_string : parsed_tokens_list) {
        if (word_to_id_mapping.find(token_string) != word_to_id_mapping.end()) {
            training_word_ids.push_back(word_to_id_mapping[token_string]);
        } else {
            training_word_ids.push_back(0);
        }
    }

    printf("Frequency-based Tokenization Complete. Unique Vocabulary Size: %d words/symbols.\n", current_vocabulary_size);
    save_token_frequency_report(model_file_path, sorted_vocabulary_list, word_to_id_mapping);
    return true;
}

void execute_model_inference(const std::string& prompt_string, const std::map<std::string, int>& word_to_id_mapping)
{
    auto prompt_token_ids = convert_prompt_to_token_vector(prompt_string, word_to_id_mapping);
    generate_autoregressive_tokens(prompt_token_ids.data(), prompt_token_ids.size(), maximum_new_tokens_to_generate);
}

// ============================================================================
// TRAINING LOOP
// ============================================================================
void execute_model_training(const std::vector<int>& training_word_ids)
{
    int batch_tokens[context_length_limit];
    int batch_targets[context_length_limit];
    
    for (int step = 0; step < total_training_steps; step++) {
        clock_t training_start_time = clock();
        reset_all_gradients();

        int random_start_offset = next_random() % (training_word_ids.size() - context_length_limit);
        for (int index = 0; index < context_length_limit; index++) {
            batch_tokens[index]  = training_word_ids[random_start_offset + index];
            batch_targets[index] = training_word_ids[random_start_offset + index + 1];
        }

        float current_learning_rate = calculate_learning_rate_schedule(step, total_training_steps);
        float computed_loss_value = execute_forward_pass(batch_tokens, batch_targets, context_length_limit);
        execute_backward_pass(batch_tokens, batch_targets, context_length_limit);
        execute_adamw_optimizer_step(current_learning_rate, step);

        clock_t training_end_time = clock();
        float elapsed_milliseconds = (float)(training_end_time - training_start_time) * 1000.0f / CLOCKS_PER_SEC;

        printf("step %d loss %.4f [%.2f ms]\n", step, computed_loss_value, elapsed_milliseconds);

        if (computed_loss_value <= early_stopping_loss_threshold) {
            printf("Early stopping triggered: Loss %.4f reached target threshold (<= %.1f)\n", computed_loss_value, early_stopping_loss_threshold);
            break;
        }
    }

    save_model_weights_to_disk(model_file_path);
}

bool load_vocabulary_from_tokens_file(
    const std::string& model_file_name,
    std::map<std::string, int>& word_to_id_mapping)
{
    std::string report_file_path = model_file_name + std::string("_tokens.txt");
    std::ifstream input_file_stream(report_file_path);
    if (!input_file_stream.is_open()) {
        return false;
    }

    std::string header_line;
    std::getline(input_file_stream, header_line); // Skip header: TOKEN_ID FREQUENCY WORD

    word_to_id_mapping.clear();
    current_vocabulary_size = 0;

    int token_id;
    std::string frequency_str, word_str;

    while (input_file_stream >> token_id >> frequency_str) {
        char tab_or_space;
        input_file_stream.get(tab_or_space);
        std::getline(input_file_stream, word_str);

        if (!word_str.empty() && word_str.back() == '\r') {
            word_str.pop_back();
        }

        if (token_id >= 0 && token_id < maximum_vocabulary_size) {
            word_to_id_mapping[word_str] = token_id;
            snprintf(vocabulary_word_strings[token_id], sizeof(vocabulary_word_strings[token_id]), "%s", word_str.c_str());
            if (token_id >= current_vocabulary_size) {
                current_vocabulary_size = token_id + 1;
            }
        }
    }

    input_file_stream.close();
    printf("Loaded vocabulary of size %d from existing token file '%s'.\n", current_vocabulary_size, report_file_path.c_str());
    return true;
}

int main(int argc, char** argv)
{
    std::string training_file_path = (argc > 1) ? argv[1] : "TinyStories-valid.txt";
    std::string user_prompt_string = (argc > 2) ? argv[2] : "A Frog and a";
    model_file_path = training_file_path + ".bin";

    std::map<std::string, int> word_to_id_mapping;
    std::vector<int> training_word_ids;

    // 1. Try loading vocabulary from existing token report first
    bool vocabulary_loaded = load_vocabulary_from_tokens_file(model_file_path, word_to_id_mapping);
    if (!vocabulary_loaded) {
        load_and_tokenize_corpus_file(training_file_path, training_word_ids, word_to_id_mapping);
    }

    initialize_model_weights();

    // 2. Check if model weights exist
    std::ifstream check_file_stream(model_file_path, std::ios::binary);
    bool model_file_exists = check_file_stream.is_open();

    if (model_file_exists) {
        check_file_stream.close();
        load_model_weights_from_disk(model_file_path);
    } else {
        // If training is required but tokens weren't converted to IDs yet
        if (training_word_ids.empty()) {
            load_and_tokenize_corpus_file(training_file_path, training_word_ids, word_to_id_mapping);
        }
        execute_model_training(training_word_ids);
    }

    execute_model_inference(user_prompt_string, word_to_id_mapping);
    return 0;
}