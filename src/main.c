/*
 * NumLLM — Vrai Mini-LLM Transformer (architecture llama2.c, poids INT8 en Flash XIP)
 * Tourne 100% hors-ligne sur NumWorks (N0110/N0115/N0120 avec Nwagra, < 6 Mo).
 * Zero syscall libc : compatible 100% avec my.numworks.com/apps.
 */

#include <eadk.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "model_weights.h"

const char eadk_app_name[] __attribute__((section(".rodata.eadk_app_name"))) = "NumLLM";
const uint32_t eadk_api_level __attribute__((section(".rodata.eadk_api_level"))) = 0;

/* --- Couleurs RGB565 --- */
#define COL_BG       0x0861
#define COL_HEADER   0x10A3
#define COL_INPUT    0x18E4
#define COL_USER_BG  0x1A68
#define COL_AI_BG    0x18C3
#define COL_TEXT     0xFFFF
#define COL_ACCENT   0xFBE0
#define COL_GREEN    0x3FE6

/* --- Buffers RAM statiques (< 110 Ko au total) --- */
static float x[LLM_DIM];
static float xb[LLM_DIM];
static float xb2[LLM_DIM];
static float hb[LLM_HIDDEN];
static float hb2[LLM_HIDDEN];
static float q[LLM_DIM];
static float k[LLM_DIM];
static float v[LLM_DIM];
static float att[LLM_HEADS * LLM_SEQ];
static float logits[LLM_VOCAB];

/* KV-Cache quantifie en int8 pour tenir dans la RAM de la NumWorks */
static int8_t key_cache[LLM_LAYERS * LLM_SEQ * LLM_DIM];
static int8_t val_cache[LLM_LAYERS * LLM_SEQ * LLM_DIM];
static float key_scale[LLM_LAYERS * LLM_SEQ];
static float val_scale[LLM_LAYERS * LLM_SEQ];

/* --- Operations mathematiques du Transformer --- */
static void rmsnorm(float *o, const float *in, const float *w, int size) {
    float ss = 0.0f;
    for (int i = 0; i < size; i++) ss += in[i] * in[i];
    ss = 1.0f / sqrtf(ss / (float)size + 1e-5f);
    for (int i = 0; i < size; i++) o[i] = w[i] * (ss * in[i]);
}

static void softmax(float *arr, int size) {
    float max_v = arr[0];
    for (int i = 1; i < size; i++) if (arr[i] > max_v) max_v = arr[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        arr[i] = expf(arr[i] - max_v);
        sum += arr[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < size; i++) arr[i] *= inv;
}

/* MatMul INT8 lu directement depuis la Flash (XIP) sans jamais copier en RAM */
static void matmul_q8(float *xout, const float *xin, const int8_t *w, float scale, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        const int8_t *row = w + i * n;
        for (int j = 0; j < n; j++) {
            val += (float)row[j] * xin[j];
        }
        xout[i] = val * scale;
    }
}

static void quantize_vec(const float *in, int8_t *out, float *scale, int n) {
    float m = 1e-6f;
    for (int i = 0; i < n; i++) {
        float a = in[i] < 0 ? -in[i] : in[i];
        if (a > m) m = a;
    }
    *scale = m / 127.0f;
    float inv = 127.0f / m;
    for (int i = 0; i < n; i++) {
        int qv = (int)(in[i] * inv);
        if (qv > 127) qv = 127;
        if (qv < -127) qv = -127;
        out[i] = (int8_t)qv;
    }
}

static float *transformer_forward(int token, int pos) {
    int head_size = LLM_DIM / LLM_HEADS;
    if (token < 0 || token >= LLM_VOCAB) token = 0;

    /* Embedding lu depuis la Flash */
    const int8_t *emb = W_EMBED + token * LLM_DIM;
    for (int i = 0; i < LLM_DIM; i++) x[i] = (float)emb[i] * S_EMBED;

    for (int l = 0; l < LLM_LAYERS; l++) {
        rmsnorm(xb, x, W_RMS_ATT + l * LLM_DIM, LLM_DIM);

        matmul_q8(q, xb, W_Q + l * LLM_DIM * LLM_DIM, S_Q[l], LLM_DIM, LLM_DIM);
        matmul_q8(k, xb, W_K + l * LLM_DIM * LLM_DIM, S_K[l], LLM_DIM, LLM_DIM);
        matmul_q8(v, xb, W_V + l * LLM_DIM * LLM_DIM, S_V[l], LLM_DIM, LLM_DIM);

        /* RoPE (Rotary Positional Embedding) */
        for (int i = 0; i < LLM_DIM; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)head_size);
            float val = (float)pos * freq;
            float fcr = cosf(val), fci = sinf(val);
            float q0 = q[i], q1 = q[i + 1];
            q[i] = q0 * fcr - q1 * fci;
            q[i + 1] = q0 * fci + q1 * fcr;
            float k0 = k[i], k1 = k[i + 1];
            k[i] = k0 * fcr - k1 * fci;
            k[i + 1] = k0 * fci + k1 * fcr;
        }

        /* Stockage dans le KV-Cache INT8 */
        int cache_idx = l * LLM_SEQ + pos;
        quantize_vec(k, key_cache + cache_idx * LLM_DIM, &key_scale[cache_idx], LLM_DIM);
        quantize_vec(v, val_cache + cache_idx * LLM_DIM, &val_scale[cache_idx], LLM_DIM);

        /* Multi-Head Self-Attention */
        for (int h = 0; h < LLM_HEADS; h++) {
            float *qh = q + h * head_size;
            float *atth = att + h * LLM_SEQ;
            for (int t = 0; t <= pos; t++) {
                int cidx = l * LLM_SEQ + t;
                const int8_t *kh = key_cache + cidx * LLM_DIM + h * head_size;
                float ks = key_scale[cidx];
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += qh[i] * ((float)kh[i] * ks);
                atth[t] = score / sqrtf((float)head_size);
            }
            softmax(atth, pos + 1);

            float *xbh = xb + h * head_size;
            for (int i = 0; i < head_size; i++) xbh[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                int cidx = l * LLM_SEQ + t;
                const int8_t *vh = val_cache + cidx * LLM_DIM + h * head_size;
                float vs = val_scale[cidx] * atth[t];
                for (int i = 0; i < head_size; i++) xbh[i] += (float)vh[i] * vs;
            }
        }

        matmul_q8(xb2, xb, W_O + l * LLM_DIM * LLM_DIM, S_O[l], LLM_DIM, LLM_DIM);
        for (int i = 0; i < LLM_DIM; i++) x[i] += xb2[i];

        /* SwiGLU FFN */
        rmsnorm(xb, x, W_RMS_FFN + l * LLM_DIM, LLM_DIM);
        matmul_q8(hb, xb, W_W1 + l * LLM_HIDDEN * LLM_DIM, S_W1[l], LLM_DIM, LLM_HIDDEN);
        matmul_q8(hb2, xb, W_W3 + l * LLM_HIDDEN * LLM_DIM, S_W3[l], LLM_DIM, LLM_HIDDEN);
        for (int i = 0; i < LLM_HIDDEN; i++) {
            float val = hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            hb[i] = val * hb2[i];
        }
        matmul_q8(xb, hb, W_W2 + l * LLM_DIM * LLM_HIDDEN, S_W2[l], LLM_HIDDEN, LLM_DIM);
        for (int i = 0; i < LLM_DIM; i++) x[i] += xb[i];
    }

    rmsnorm(x, x, W_RMS_FINAL, LLM_DIM);
    matmul_q8(logits, x, W_EMBED, S_EMBED, LLM_DIM, LLM_VOCAB);
    return logits;
}

static int argmax(const float *arr, int n) {
    int best = 0;
    float best_v = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > best_v) { best_v = arr[i]; best = i; }
    }
    return best;
}

/* --- Interface NumChat --- */
#define MAX_LINES 60
#define LINE_CHARS 42
#define INPUT_MAX 56

typedef struct { char text[LINE_CHARS + 1]; bool is_user; } chat_line_t;
static chat_line_t history[MAX_LINES];
static int line_count = 0;
static char input_buf[INPUT_MAX + 1];
static int input_len = 0;

typedef struct { eadk_key_t key; char ch; } key_map_t;
static const key_map_t KEY_MAP[] = {
    {eadk_key_exp, 'a'}, {eadk_key_ln, 'b'}, {eadk_key_log, 'c'},
    {eadk_key_imaginary, 'd'}, {eadk_key_comma, 'e'}, {eadk_key_power, 'f'},
    {eadk_key_sine, 'g'}, {eadk_key_cosine, 'h'}, {eadk_key_tangent, 'i'},
    {eadk_key_pi, 'j'}, {eadk_key_sqrt, 'k'}, {eadk_key_square, 'l'},
    {eadk_key_seven, 'm'}, {eadk_key_eight, 'n'}, {eadk_key_nine, 'o'},
    {eadk_key_left_parenthesis, 'p'}, {eadk_key_right_parenthesis, 'q'},
    {eadk_key_four, 'r'}, {eadk_key_five, 's'}, {eadk_key_six, 't'},
    {eadk_key_multiplication, 'u'}, {eadk_key_division, 'v'},
    {eadk_key_one, 'w'}, {eadk_key_two, 'x'}, {eadk_key_three, 'y'},
    {eadk_key_plus, 'z'}, {eadk_key_zero, ' '}, {eadk_key_dot, '.'},
};

static void push_line(const char *s, bool is_user) {
    if (line_count >= MAX_LINES) {
        memmove(&history[0], &history[1], (MAX_LINES - 1) * sizeof(chat_line_t));
        line_count = MAX_LINES - 1;
    }
    strncpy(history[line_count].text, s, LINE_CHARS);
    history[line_count].text[LINE_CHARS] = '\0';
    history[line_count].is_user = is_user;
    line_count++;
}

static void draw_ui(const char *status) {
    eadk_display_push_rect_uniform((eadk_rect_t){0, 0, 320, 240}, COL_BG);
    eadk_display_push_rect_uniform((eadk_rect_t){0, 0, 320, 22}, COL_HEADER);
    eadk_display_draw_string("NumLLM (Transformer INT8)", (eadk_point_t){6, 4}, false, COL_ACCENT, COL_HEADER);
    eadk_display_draw_string(status, (eadk_point_t){235, 4}, false, COL_GREEN, COL_HEADER);

    int start = line_count > 11 ? line_count - 11 : 0;
    int y = 26;
    for (int i = start; i < line_count; i++) {
        eadk_color_t bg = history[i].is_user ? COL_USER_BG : COL_AI_BG;
        eadk_color_t fg = history[i].is_user ? COL_GREEN : COL_TEXT;
        eadk_display_push_rect_uniform((eadk_rect_t){4, (uint16_t)y, 312, 16}, bg);
        eadk_display_draw_string(history[i].text, (eadk_point_t){8, (uint16_t)(y + 2)}, false, fg, bg);
        y += 17;
    }

    eadk_display_push_rect_uniform((eadk_rect_t){0, 214, 320, 26}, COL_INPUT);
    char prompt[INPUT_MAX + 5];
    prompt[0] = '>'; prompt[1] = ' ';
    strncpy(prompt + 2, input_buf, INPUT_MAX);
    size_t l = strlen(prompt);
    prompt[l] = '_'; prompt[l + 1] = '\0';
    eadk_display_draw_string(prompt, (eadk_point_t){6, 220}, false, COL_TEXT, COL_INPUT);
}

static void generate_reply(const char *user_prompt) {
    draw_ui("Calcul...");
    int pos = 0;
    /* Prefill du prompt caractere par caractere (ASCII/UTF-8) */
    for (int i = 0; user_prompt[i] && pos < LLM_SEQ - 2; i++) {
        int tok = (unsigned char)user_prompt[i];
        if (tok >= LLM_VOCAB) tok = 32;
        transformer_forward(tok, pos++);
    }
    float *out = transformer_forward('\n', pos++);

    char reply[LINE_CHARS + 1];
    int rlen = 0;
    reply[0] = '\0';

    /* Generation autoregressive token par token */
    for (int step = 0; step < 64 && pos < LLM_SEQ; step++) {
        int next_tok = argmax(out, LLM_VOCAB);
        if (next_tok == 0 || next_tok == '\n') break;
        if (next_tok >= 32 && next_tok < 127) {
            reply[rlen++] = (char)next_tok;
            reply[rlen] = '\0';
            if (rlen >= LINE_CHARS - 1) {
                push_line(reply, false);
                rlen = 0;
                reply[0] = '\0';
                draw_ui("Gen...");
            }
        }
        out = transformer_forward(next_tok, pos++);
    }
    if (rlen > 0) push_line(reply, false);
    draw_ui("Pret");
}

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    push_line("NumLLM pret ! (Transformer INT8 en Flash)", false);
    draw_ui("Pret");

    eadk_keyboard_state_t prev = eadk_keyboard_scan();
    while (true) {
        eadk_keyboard_state_t now = eadk_keyboard_scan();
        bool dirty = false;
        if (eadk_keyboard_key_down(now, eadk_key_back) && !eadk_keyboard_key_down(prev, eadk_key_back)) break;
        if (eadk_keyboard_key_down(now, eadk_key_backspace) && !eadk_keyboard_key_down(prev, eadk_key_backspace)) {
            if (input_len > 0) { input_buf[--input_len] = '\0'; dirty = true; }
        }
        if ((eadk_keyboard_key_down(now, eadk_key_exe) && !eadk_keyboard_key_down(prev, eadk_key_exe)) ||
            (eadk_keyboard_key_down(now, eadk_key_ok) && !eadk_keyboard_key_down(prev, eadk_key_ok))) {
            if (input_len > 0) {
                push_line(input_buf, true);
                char q[INPUT_MAX + 1];
                strcpy(q, input_buf);
                input_len = 0; input_buf[0] = '\0';
                generate_reply(q);
                dirty = false;
            }
        }
        for (size_t i = 0; i < sizeof(KEY_MAP) / sizeof(KEY_MAP[0]); i++) {
            if (eadk_keyboard_key_down(now, KEY_MAP[i].key) && !eadk_keyboard_key_down(prev, KEY_MAP[i].key)) {
                if (input_len < INPUT_MAX) {
                    input_buf[input_len++] = KEY_MAP[i].ch;
                    input_buf[input_len] = '\0';
                    dirty = true;
                }
            }
        }
        if (dirty) draw_ui("Pret");
        prev = now;
        eadk_timing_msleep(20);
    }
    return 0;
}
