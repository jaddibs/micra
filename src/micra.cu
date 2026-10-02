#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <random>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define BLOCK_SIZE 128
#define TILE 16
#define MAX_DIM (1 << 20)

#define EPILOGUE_NONE 0
#define EPILOGUE_RELU 1
#define EPILOGUE_ACCUMULATE 2

__global__ void EmbeddingKernel(float *x, const float *wte, const float *wpe,
                                const int *tokens, int pos0, int count,
                                int n_embd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * n_embd) {
        int row = i / n_embd;
        int col = i - row * n_embd;
        x[i] = wte[tokens[row] * n_embd + col] +
               wpe[(pos0 + row) * n_embd + col];
    }
}

__global__ void RmsNormKernel(float *out, const float *x, int n)
{
    __shared__ float partial[BLOCK_SIZE];

    const float *x_row = x + (size_t) blockIdx.x * n;
    float *out_row = out + (size_t) blockIdx.x * n;

    float sum = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += x_row[i] * x_row[i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    // tree reduction. once it's down to one warp, __syncwarp is enough
    int stride = blockDim.x / 2;
    while (stride > 32) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
        stride /= 2;
    }
    if (threadIdx.x < 32) {
        while (stride > 0) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncwarp();
            stride /= 2;
        }
    }
    __syncthreads();

    float scale = rsqrtf(partial[0] / n + 1e-5f);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * scale;
    }
}

__global__ void LinearKernel(float *y, const float *w, const float *x,
                             int rows, int cols, int epilogue)
{
    __shared__ float partial[BLOCK_SIZE];
    int row = blockIdx.x;

    // read 4 floats at a time if the pointers are aligned for it
    float sum = 0;
    if (cols % 4 == 0 && (size_t) w % 16 == 0 && (size_t) x % 16 == 0) {
        const float4 *w_row = (const float4 *) (w + (size_t) row * cols);
        const float4 *x_vec = (const float4 *) x;
        for (int col = threadIdx.x; col < cols / 4; col += blockDim.x) {
            float4 wv = w_row[col];
            float4 xv = x_vec[col];
            sum += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
        }
    } else {
        for (int col = threadIdx.x; col < cols; col += blockDim.x) {
            sum += w[(size_t) row * cols + col] * x[col];
        }
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    int stride = blockDim.x / 2;
    while (stride > 32) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
        stride /= 2;
    }
    if (threadIdx.x < 32) {
        while (stride > 0) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncwarp();
            stride /= 2;
        }
    }

    if (threadIdx.x == 0) {
        float result = partial[0];
        if (epilogue == EPILOGUE_RELU) {
            result = fmaxf(result, 0.0f);
        } else if (epilogue == EPILOGUE_ACCUMULATE) {
            result += y[row];
        }
        y[row] = result;
    }
}

__global__ void GemmKernel(float *y, const float *w, const float *x,
                           int batch, int rows, int cols, int epilogue)
{
    __shared__ float x_tile[TILE][TILE];
    // extra column so reading down a column of w_tile doesn't bank conflict
    __shared__ float w_tile[TILE][TILE + 1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int tiles = cols / TILE;
    if (cols % TILE != 0) {
        tiles++;
    }

    float sum = 0;
    for (int tile = 0; tile < tiles; tile++) {
        int k = tile * TILE + threadIdx.x;

        if (row < batch && k < cols) {
            x_tile[threadIdx.y][threadIdx.x] = x[(size_t) row * cols + k];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = 0;
        }

        int w_row = blockIdx.x * TILE + threadIdx.y;
        if (w_row < rows && k < cols) {
            w_tile[threadIdx.y][threadIdx.x] = w[(size_t) w_row * cols + k];
        } else {
            w_tile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int j = 0; j < TILE; j++) {
            sum += x_tile[threadIdx.y][j] * w_tile[threadIdx.x][j];
        }
        __syncthreads();
    }

    if (row < batch && col < rows) {
        float result = sum;
        if (epilogue == EPILOGUE_RELU) {
            result = fmaxf(result, 0.0f);
        } else if (epilogue == EPILOGUE_ACCUMULATE) {
            result += y[(size_t) row * rows + col];
        }
        y[(size_t) row * rows + col] = result;
    }
}

__global__ void AttentionKernel(float *out, const float *q,
                                const float *k_cache, const float *v_cache,
                                int pos0, int n_embd, int head_dim)
{
    extern __shared__ float scores[];
    int head_start = blockIdx.x * head_dim;
    int pos = pos0 + blockIdx.y;
    const float *q_row = q + (size_t) blockIdx.y * n_embd;
    float *out_row = out + (size_t) blockIdx.y * n_embd;

    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        float score = 0;
        for (int i = 0; i < head_dim; i++) {
            score += k_cache[t * n_embd + head_start + i] *
                     q_row[head_start + i];
        }
        scores[t] = score / sqrtf((float) head_dim);
    }
    __syncthreads();

    // softmax on one thread. subtract the max first so expf can't overflow
    if (threadIdx.x == 0) {
        float max_score = scores[0];
        for (int t = 1; t <= pos; t++) {
            if (scores[t] > max_score) {
                max_score = scores[t];
            }
        }

        float total = 0;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            total += scores[t];
        }
        for (int t = 0; t <= pos; t++) {
            scores[t] /= total;
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float value = 0;
        for (int t = 0; t <= pos; t++) {
            value += scores[t] * v_cache[t * n_embd + head_start + i];
        }
        out_row[head_start + i] = value;
    }
}

__global__ void FlashAttentionKernel(float *out, const float *q,
                                     const float *k_cache, const float *v_cache,
                                     int pos0, int n_embd, int head_dim)
{
    __shared__ float partial[BLOCK_SIZE];
    int head_start = blockIdx.x * head_dim;
    int channel = threadIdx.x;
    int pos = pos0 + blockIdx.y;
    const float *q_row = q + (size_t) blockIdx.y * n_embd;
    float *out_row = out + (size_t) blockIdx.y * n_embd;

    float running_max = -INFINITY;
    float running_sum = 0;
    float acc = 0;

    for (int t = 0; t <= pos; t++) {
        float dot = 0;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            dot += k_cache[t * n_embd + head_start + i] * q_row[head_start + i];
        }
        partial[threadIdx.x] = dot;
        __syncthreads();

        int stride = blockDim.x / 2;
        while (stride > 32) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
            stride /= 2;
        }
        if (threadIdx.x < 32) {
            while (stride > 0) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
                __syncwarp();
                stride /= 2;
            }
        }
        __syncthreads();
        float score = partial[0] / sqrtf((float) head_dim);

        // online softmax: if the max moved, rescale what's been summed so far
        float new_max = fmaxf(running_max, score);
        float correction = expf(running_max - new_max);
        float weight = expf(score - new_max);
        running_sum = running_sum * correction + weight;
        if (channel < head_dim) {
            acc = acc * correction +
                  weight * v_cache[t * n_embd + head_start + channel];
        }
        running_max = new_max;
        __syncthreads();
    }

    if (channel < head_dim) {
        out_row[head_start + channel] = acc / running_sum;
    }
}

int blocksFor(int n)
{
    int blocks = n / BLOCK_SIZE;
    if (n % BLOCK_SIZE != 0) {
        blocks++;
    }
    return blocks;
}

int tilesFor(int n)
{
    int tiles = n / TILE;
    if (n % TILE != 0) {
        tiles++;
    }
    return tiles;
}

void embedding(float *x_d, const float *wte_d, const float *wpe_d,
               const int *tokens_d, int pos0, int count, int n_embd)
{
    EmbeddingKernel <<< blocksFor(count * n_embd), BLOCK_SIZE >>> (
        x_d, wte_d, wpe_d, tokens_d, pos0, count, n_embd);
    CUDA_CHECK(cudaGetLastError());
}

void rmsnorm(float *out_d, const float *x_d, int n, int rows = 1)
{
    RmsNormKernel <<< rows, BLOCK_SIZE >>> (out_d, x_d, n);
    CUDA_CHECK(cudaGetLastError());
}

void linear(float *y_d, const float *w_d, const float *x_d,
            int rows, int cols, int epilogue = EPILOGUE_NONE)
{
    LinearKernel <<< rows, BLOCK_SIZE >>> (y_d, w_d, x_d, rows, cols, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

void gemm(float *y_d, const float *w_d, const float *x_d,
          int batch, int rows, int cols, int epilogue = EPILOGUE_NONE)
{
    dim3 grid(tilesFor(rows), tilesFor(batch));
    dim3 block(TILE, TILE);
    GemmKernel <<< grid, block >>> (y_d, w_d, x_d, batch, rows, cols, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

void attention(float *out_d, const float *q_d, const float *k_cache_d,
               const float *v_cache_d, int pos0, int count,
               int n_embd, int n_head)
{
    int head_dim = n_embd / n_head;
    dim3 grid(n_head, count);
    size_t score_bytes = (size_t) (pos0 + count) * sizeof(float);
    AttentionKernel <<< grid, BLOCK_SIZE, score_bytes >>> (
        out_d, q_d, k_cache_d, v_cache_d, pos0, n_embd, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void flashAttention(float *out_d, const float *q_d,
                    const float *k_cache_d, const float *v_cache_d,
                    int pos0, int count, int n_embd, int n_head)
{
    int head_dim = n_embd / n_head;
    dim3 grid(n_head, count);
    FlashAttentionKernel <<< grid, BLOCK_SIZE >>> (
        out_d, q_d, k_cache_d, v_cache_d, pos0, n_embd, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

struct Config {
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t block_size;
    uint32_t vocab_size;
};

struct Layer {
    float *wq, *wk, *wv, *wo, *fc1, *fc2;
};

struct Model {
    Config config;
    std::vector<uint32_t> vocab;
    float *weights_d;
    float *wte, *wpe, *lm_head;
    std::vector<Layer> layers;
    float *x, *x_norm, *q, *x_attn, *hidden, *logits_d;
    float *k_cache, *v_cache;
    int *tokens;
};

void fail(const char *message)
{
    fprintf(stderr, "micra: %s\n", message);
    exit(1);
}

uint32_t readU32(const uint8_t *b)
{
    return (uint32_t) b[0] | (uint32_t) b[1] << 8 |
           (uint32_t) b[2] << 16 | (uint32_t) b[3] << 24;
}

size_t align32(size_t offset)
{
    return (offset + 31) / 32 * 32;
}

float *deviceAlloc(size_t count)
{
    float *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(float)));
    return ptr;
}

Model loadModel(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fail("cannot open checkpoint file");
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    std::vector<uint8_t> bytes(file_size);
    if (fread(bytes.data(), 1, file_size, file) != (size_t) file_size) {
        fail("cannot read checkpoint file");
    }
    fclose(file);

    // header fields and offsets are in docs/format.md
    if (file_size < 76) {
        fail("file smaller than the 76-byte header");
    }
    if (memcmp(bytes.data(), "MICRA\0\0\0", 8) != 0) {
        fail("bad magic, not a .micra file");
    }
    if (readU32(&bytes[8]) != 1) {
        fail("unsupported format version");
    }
    if (readU32(&bytes[12]) != 1) {
        fail("unsupported dtype, expected fp32");
    }
    for (int i = 0; i < 8; i++) {
        if (readU32(&bytes[44 + 4 * i]) != 0) {
            fail("reserved header words must be zero");
        }
    }

    Config c;
    c.n_layer = readU32(&bytes[16]);
    c.n_embd = readU32(&bytes[20]);
    c.n_head = readU32(&bytes[24]);
    c.block_size = readU32(&bytes[28]);
    c.vocab_size = readU32(&bytes[32]);
    uint32_t vocab_bytes = readU32(&bytes[36]);
    uint32_t tensor_count = readU32(&bytes[40]);

    uint32_t dims[5] = {c.n_layer, c.n_embd, c.n_head, c.block_size, c.vocab_size};
    for (int i = 0; i < 5; i++) {
        if (dims[i] == 0 || dims[i] > MAX_DIM) {
            fail("model dimension out of range");
        }
    }
    if (c.n_embd % c.n_head != 0) {
        fail("n_embd not divisible by n_head");
    }
    if (vocab_bytes != 4 + 4 * (c.vocab_size - 1)) {
        fail("vocab_bytes inconsistent with vocab_size");
    }
    if (tensor_count != 3 + 6 * c.n_layer) {
        fail("unexpected tensor count");
    }

    size_t d = c.n_embd;
    size_t params = 2 * c.vocab_size * d + c.block_size * d +
                    c.n_layer * 12 * d * d;
    size_t payload_offset = align32(76 + vocab_bytes);
    if ((size_t) file_size != payload_offset + 4 * params) {
        fail("file size does not match header");
    }

    uint32_t vocab_count = readU32(&bytes[76]);
    if (vocab_count != c.vocab_size - 1) {
        fail("vocab count inconsistent with vocab_size");
    }

    Model m;
    m.config = c;
    for (uint32_t i = 0; i < vocab_count; i++) {
        uint32_t ch = readU32(&bytes[80 + 4 * i]);
        if (ch > 127) {
            fail("only ASCII vocabularies are supported");
        }
        m.vocab.push_back(ch);
    }

    m.weights_d = deviceAlloc(params);
    CUDA_CHECK(cudaMemcpy(m.weights_d, bytes.data() + payload_offset,
                          params * sizeof(float), cudaMemcpyHostToDevice));

    float *p = m.weights_d;
    m.wte = p;
    p += (size_t) c.vocab_size * d;
    m.wpe = p;
    p += (size_t) c.block_size * d;
    for (uint32_t li = 0; li < c.n_layer; li++) {
        Layer layer;
        layer.wq = p; p += d * d;
        layer.wk = p; p += d * d;
        layer.wv = p; p += d * d;
        layer.wo = p; p += d * d;
        layer.fc1 = p; p += 4 * d * d;
        layer.fc2 = p; p += 4 * d * d;
        m.layers.push_back(layer);
    }
    m.lm_head = p;

    m.x = deviceAlloc((size_t) c.block_size * d);
    m.x_norm = deviceAlloc((size_t) c.block_size * d);
    m.q = deviceAlloc((size_t) c.block_size * d);
    m.x_attn = deviceAlloc((size_t) c.block_size * d);
    m.hidden = deviceAlloc((size_t) c.block_size * 4 * d);
    m.logits_d = deviceAlloc(c.vocab_size);
    m.k_cache = deviceAlloc((size_t) c.n_layer * c.block_size * d);
    m.v_cache = deviceAlloc((size_t) c.n_layer * c.block_size * d);
    CUDA_CHECK(cudaMalloc(&m.tokens, c.block_size * sizeof(int)));
    return m;
}

void freeModel(Model &m)
{
    cudaFree(m.weights_d);
    cudaFree(m.x);
    cudaFree(m.x_norm);
    cudaFree(m.q);
    cudaFree(m.x_attn);
    cudaFree(m.hidden);
    cudaFree(m.logits_d);
    cudaFree(m.k_cache);
    cudaFree(m.v_cache);
    cudaFree(m.tokens);
}

void forward(Model &m, int token, int pos, bool use_flash, float *logits_out)
{
    Config &c = m.config;
    int d = c.n_embd;

    CUDA_CHECK(cudaMemcpy(m.tokens + pos, &token, sizeof(int),
                          cudaMemcpyHostToDevice));
    embedding(m.x, m.wte, m.wpe, m.tokens + pos, pos, 1, d);
    rmsnorm(m.x, m.x, d);

    for (uint32_t li = 0; li < c.n_layer; li++) {
        Layer &layer = m.layers[li];
        float *k_cache = m.k_cache + (size_t) li * c.block_size * d;
        float *v_cache = m.v_cache + (size_t) li * c.block_size * d;

        rmsnorm(m.x_norm, m.x, d);
        linear(m.q, layer.wq, m.x_norm, d, d);
        // k and v go straight into this position's row of the cache
        linear(k_cache + (size_t) pos * d, layer.wk, m.x_norm, d, d);
        linear(v_cache + (size_t) pos * d, layer.wv, m.x_norm, d, d);
        if (use_flash) {
            flashAttention(m.x_attn, m.q, k_cache, v_cache, pos, 1, d,
                           c.n_head);
        } else {
            attention(m.x_attn, m.q, k_cache, v_cache, pos, 1, d, c.n_head);
        }
        linear(m.x, layer.wo, m.x_attn, d, d, EPILOGUE_ACCUMULATE);

        rmsnorm(m.x_norm, m.x, d);
        linear(m.hidden, layer.fc1, m.x_norm, 4 * d, d, EPILOGUE_RELU);
        linear(m.x, layer.fc2, m.hidden, d, 4 * d, EPILOGUE_ACCUMULATE);
    }

    linear(m.logits_d, m.lm_head, m.x, c.vocab_size, d);
    CUDA_CHECK(cudaMemcpy(logits_out, m.logits_d,
                          c.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
}

// prompt goes through as one batch, so gemm where forward() uses linear
void forwardPrefill(Model &m, const int *tokens, int count,
                    bool use_flash, float *logits_out)
{
    Config &c = m.config;
    int d = c.n_embd;

    CUDA_CHECK(cudaMemcpy(m.tokens, tokens, count * sizeof(int),
                          cudaMemcpyHostToDevice));
    embedding(m.x, m.wte, m.wpe, m.tokens, 0, count, d);
    rmsnorm(m.x, m.x, d, count);

    for (uint32_t li = 0; li < c.n_layer; li++) {
        Layer &layer = m.layers[li];
        float *k_cache = m.k_cache + (size_t) li * c.block_size * d;
        float *v_cache = m.v_cache + (size_t) li * c.block_size * d;

        rmsnorm(m.x_norm, m.x, d, count);
        gemm(m.q, layer.wq, m.x_norm, count, d, d);
        gemm(k_cache, layer.wk, m.x_norm, count, d, d);
        gemm(v_cache, layer.wv, m.x_norm, count, d, d);
        if (use_flash) {
            flashAttention(m.x_attn, m.q, k_cache, v_cache, 0, count, d,
                           c.n_head);
        } else {
            attention(m.x_attn, m.q, k_cache, v_cache, 0, count, d, c.n_head);
        }
        gemm(m.x, layer.wo, m.x_attn, count, d, d, EPILOGUE_ACCUMULATE);

        rmsnorm(m.x_norm, m.x, d, count);
        gemm(m.hidden, layer.fc1, m.x_norm, count, 4 * d, d, EPILOGUE_RELU);
        gemm(m.x, layer.fc2, m.hidden, count, d, 4 * d, EPILOGUE_ACCUMULATE);
    }

    linear(m.logits_d, m.lm_head, m.x + (size_t) (count - 1) * d,
           c.vocab_size, d);
    CUDA_CHECK(cudaMemcpy(logits_out, m.logits_d,
                          c.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
}

int sampleToken(const std::vector<float> &logits, bool greedy,
                float temperature, std::mt19937 &rng)
{
    int vocab_size = logits.size();

    if (greedy) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        return best;
    }

    float max_logit = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    std::vector<float> probs(vocab_size);
    float total = 0;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        total += probs[i];
    }

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float r = uniform(rng) * total;
    float cumulative = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumulative += probs[i];
        if (r < cumulative) {
            return i;
        }
    }
    return vocab_size - 1;
}

void usage()
{
    fprintf(stderr,
            "usage: micra checkpoint.micra [options]\n"
            "  --prompt STR          starting text (default empty)\n"
            "  --max-new-tokens N    cap on generated tokens (default block_size)\n"
            "  --samples N           number of sequences to generate (default 1)\n"
            "  --greedy              pick argmax instead of sampling\n"
            "  --temperature F       sampling temperature (default 0.5)\n"
            "  --seed N              rng seed (default 42)\n"
            "  --flash               use the online-softmax attention kernel\n"
            "  --verify              check prefill against one-token decode, then exit\n");
    exit(1);
}

int main(int argc, char **argv)
{
    const char *checkpoint_path = NULL;
    const char *prompt = "";
    int max_new_tokens = -1;
    int num_samples = 1;
    bool greedy = false;
    bool use_flash = false;
    bool verify = false;
    float temperature = 0.5f;
    unsigned int seed = 42;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            num_samples = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--greedy") == 0) {
            greedy = true;
        } else if (strcmp(argv[i], "--flash") == 0) {
            use_flash = true;
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int) atoi(argv[++i]);
        } else if (argv[i][0] != '-' && checkpoint_path == NULL) {
            checkpoint_path = argv[i];
        } else {
            usage();
        }
    }
    if (checkpoint_path == NULL) {
        usage();
    }

    Model model = loadModel(checkpoint_path);
    Config &c = model.config;
    printf("loaded %s: n_layer=%u n_embd=%u n_head=%u block_size=%u vocab_size=%u\n",
           checkpoint_path, c.n_layer, c.n_embd, c.n_head, c.block_size,
           c.vocab_size);
    if (use_flash && c.n_embd / c.n_head > BLOCK_SIZE) {
        fail("--flash supports head_dim up to 128");
    }

    std::vector<int> prompt_tokens;
    for (size_t j = 0; prompt[j] != '\0'; j++) {
        uint32_t ch = (unsigned char) prompt[j];
        int id = -1;
        for (size_t i = 0; i < model.vocab.size(); i++) {
            if (model.vocab[i] == ch) {
                id = i;
                break;
            }
        }
        if (id < 0) {
            fail("prompt contains a character outside the model vocabulary");
        }
        prompt_tokens.push_back(id);
    }
    if (prompt_tokens.size() >= c.block_size) {
        fail("prompt longer than block_size");
    }

    int bos = c.vocab_size - 1;
    int d = c.n_embd;
    std::mt19937 rng(seed);
    std::vector<float> logits(c.vocab_size);

    std::vector<int> prefill_tokens;
    prefill_tokens.push_back(bos);
    for (int t : prompt_tokens) {
        prefill_tokens.push_back(t);
    }
    int prefill_count = prefill_tokens.size();

    if (verify) {
        size_t cache_floats = (size_t) c.n_layer * c.block_size * d;
        std::vector<float> logits_seq(c.vocab_size);
        std::vector<float> k1(cache_floats), v1(cache_floats);
        std::vector<float> k2(cache_floats), v2(cache_floats);

        forwardPrefill(model, prefill_tokens.data(), prefill_count, use_flash,
                       logits.data());
        CUDA_CHECK(cudaMemcpy(k1.data(), model.k_cache,
                              cache_floats * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v1.data(), model.v_cache,
                              cache_floats * sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (int pos = 0; pos < prefill_count; pos++) {
            forward(model, prefill_tokens[pos], pos, use_flash,
                    logits_seq.data());
        }
        CUDA_CHECK(cudaMemcpy(k2.data(), model.k_cache,
                              cache_floats * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v2.data(), model.v_cache,
                              cache_floats * sizeof(float),
                              cudaMemcpyDeviceToHost));

        float max_diff = 0;
        float max_val = 0;
        for (uint32_t li = 0; li < c.n_layer; li++) {
            for (int i = 0; i < prefill_count * d; i++) {
                size_t idx = (size_t) li * c.block_size * d + i;
                max_diff = fmaxf(max_diff, fabsf(k1[idx] - k2[idx]));
                max_diff = fmaxf(max_diff, fabsf(v1[idx] - v2[idx]));
                max_val = fmaxf(max_val, fabsf(k2[idx]));
                max_val = fmaxf(max_val, fabsf(v2[idx]));
            }
        }
        for (uint32_t i = 0; i < c.vocab_size; i++) {
            max_diff = fmaxf(max_diff, fabsf(logits[i] - logits_seq[i]));
            max_val = fmaxf(max_val, fabsf(logits_seq[i]));
        }
        // relative, since the absolute diffs get bigger as the values do
        float relative = max_diff / fmaxf(max_val, 1e-6f);
        printf("prefill vs decode over %d positions: max diff %g on values "
               "up to %g (relative %g) -> %s\n",
               prefill_count, max_diff, max_val, relative,
               relative < 1e-4f ? "PASS" : "FAIL");
        freeModel(model);
        return relative < 1e-4f ? 0 : 1;
    }

    int prefill_total = 0, decode_total = 0;
    double prefill_ms = 0, decode_ms = 0;

    for (int s = 0; s < num_samples; s++) {
        std::string text(prompt);
        int generated = 0;

        auto t0 = std::chrono::steady_clock::now();
        forwardPrefill(model, prefill_tokens.data(), prefill_count, use_flash,
                       logits.data());
        auto t1 = std::chrono::steady_clock::now();
        prefill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        prefill_total += prefill_count;

        for (uint32_t pos = prefill_count; ; pos++) {
            if (max_new_tokens >= 0 && generated >= max_new_tokens) {
                break;
            }
            int token = sampleToken(logits, greedy, temperature, rng);
            // BOS is also the stop token
            if (token == bos) {
                break;
            }
            text += (char) model.vocab[token];
            generated++;
            if (pos >= c.block_size) {
                break;
            }
            forward(model, token, pos, use_flash, logits.data());
            decode_total++;
        }
        auto t2 = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();

        printf("sample %2d: %s\n", s + 1, text.c_str());
    }

    printf("prefill: %d tokens in %.2f ms (%.3f ms/token)\n",
           prefill_total, prefill_ms, prefill_ms / prefill_total);
    if (decode_total > 0) {
        printf("decode:  %d tokens in %.2f ms (%.3f ms/token)\n",
               decode_total, decode_ms, decode_ms / decode_total);
    }

    freeModel(model);
    return 0;
}
