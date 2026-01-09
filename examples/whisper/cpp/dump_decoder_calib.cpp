// Generate calibration dataset for Whisper decoder quantization
// Requires: libsndfile
// Build: g++ -o dump_decoder_calib dump_decoder_calib.cpp -lsndfile -std=c++17

#include <sndfile.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// =========================
// Config (Whisper defaults)
// =========================
static constexpr int SAMPLE_RATE   = 16000;
static constexpr int N_FFT         = 400;
static constexpr int HOP_LENGTH    = 160;
static constexpr int WIN_LENGTH    = 400;
static constexpr int N_MELS        = 80;
static constexpr float PI_F        = 3.14159265358979323846f;

// user controls this (20s -> 2000 frames; 30s -> 3000 frames)
static int CHUNK_SECONDS = 20;
static int MAX_AUDIO_LENGTH = 20 * SAMPLE_RATE;
static int ENCODER_INPUT_SIZE = (20 * SAMPLE_RATE) / HOP_LENGTH;
static int ENCODER_OUTPUT_SIZE = 20 * 50 * 512; // 1000 * 512 for 20s

// =========================
// FP32 to FP16 conversion
// =========================
static uint16_t float32_to_float16(float value) {
    uint32_t f32;
    std::memcpy(&f32, &value, sizeof(float));
    
    uint16_t sign = (f32 >> 16) & 0x8000;
    int32_t exponent = ((f32 >> 23) & 0xff) - 127;
    uint32_t mantissa = f32 & 0x7fffff;
    
    // Handle special cases
    if (exponent == 128) { // Inf or NaN
        return sign | 0x7c00 | (mantissa != 0 ? 0x0200 : 0);
    }
    if (exponent > 15) { // Overflow to infinity
        return sign | 0x7c00;
    }
    if (exponent < -14) { // Underflow
        if (exponent < -24) return sign; // Too small, return signed zero
        // Denormalized number
        mantissa = (mantissa | 0x800000) >> (1 - (exponent + 14));
        return sign | (mantissa >> 13);
    }
    
    // Normalized number
    uint16_t fp16_exp = (exponent + 15) << 10;
    uint16_t fp16_mantissa = mantissa >> 13;
    
    // Rounding
    if ((mantissa & 0x1fff) > 0x1000) {
        fp16_mantissa++;
    }
    
    return sign | fp16_exp | fp16_mantissa;
}

// =========================
// Minimal audio buffer type
// =========================
typedef struct audio_buffer_t {
    float* data = nullptr;
    int num_frames = 0;
} audio_buffer_t;

// =========================
// Utility: Hann window
// =========================
static std::vector<float> hann_window(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) {
        w[i] = 0.5f - 0.5f * std::cos(2.0f * PI_F * i / (n - 1));
    }
    return w;
}

// =========================
// Utility: Hz <-> Mel (HTK)
// =========================
static inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}
static inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// =========================
// Build mel filterbank
// =========================
static std::vector<float> build_mel_filters() {
    const int n_freqs = N_FFT / 2 + 1;
    std::vector<float> filters(N_MELS * n_freqs, 0.0f);

    const float f_min = 0.0f;
    const float f_max = SAMPLE_RATE / 2.0f;
    const float m_min = hz_to_mel(f_min);
    const float m_max = hz_to_mel(f_max);

    std::vector<float> m_pts(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        m_pts[i] = m_min + (m_max - m_min) * i / (N_MELS + 1);
    }

    std::vector<float> f_pts(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        f_pts[i] = mel_to_hz(m_pts[i]);
    }

    std::vector<int> bins(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        float bin = (N_FFT + 1) * f_pts[i] / SAMPLE_RATE;
        int b = static_cast<int>(std::floor(bin));
        if (b < 0) b = 0;
        if (b > n_freqs - 1) b = n_freqs - 1;
        bins[i] = b;
    }

    for (int m = 0; m < N_MELS; ++m) {
        int left = bins[m];
        int center = bins[m + 1];
        int right = bins[m + 2];

        if (center == left) center = left + 1;
        if (right == center) right = center + 1;
        if (right > n_freqs - 1) right = n_freqs - 1;

        for (int k = left; k < center && k < n_freqs; ++k) {
            float v = (k - left) / float(center - left);
            filters[m * n_freqs + k] = std::max(v, 0.0f);
        }
        for (int k = center; k < right && k < n_freqs; ++k) {
            float v = (right - k) / float(right - center);
            filters[m * n_freqs + k] = std::max(v, 0.0f);
        }
    }

    return filters;
}

// =========================
// Simple linear resampler
// =========================
static std::vector<float> resample_linear(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in_sr == out_sr) return in;
    const double ratio = double(out_sr) / double(in_sr);
    const size_t out_len = static_cast<size_t>(std::llround(in.size() * ratio));
    std::vector<float> out(out_len);

    for (size_t i = 0; i < out_len; ++i) {
        double src_pos = double(i) / ratio;
        size_t j = (size_t)std::floor(src_pos);
        double t = src_pos - double(j);
        float a = in[std::min(j, in.size() - 1)];
        float b = in[std::min(j + 1, in.size() - 1)];
        out[i] = (float)((1.0 - t) * a + t * b);
    }
    return out;
}

// =========================
// Read audio via libsndfile
// =========================
static bool read_audio_mono(const std::string& path, std::vector<float>& audio_out, int& sr_out) {
    SF_INFO info;
    std::memset(&info, 0, sizeof(info));
    SNDFILE* f = sf_open(path.c_str(), SFM_READ, &info);
    if (!f) {
        std::cerr << "Failed to open audio: " << path << "\n";
        return false;
    }

    sr_out = info.samplerate;
    const int ch = info.channels;
    const sf_count_t nframes = info.frames;
    std::vector<float> interleaved((size_t)nframes * (size_t)ch);

    sf_count_t nread = sf_readf_float(f, interleaved.data(), nframes);
    sf_close(f);

    if (nread != nframes) {
        std::cerr << "Warning: short read: " << path << "\n";
    }

    audio_out.resize((size_t)nread);
    for (sf_count_t i = 0; i < nread; ++i) {
        double s = 0.0;
        for (int c = 0; c < ch; ++c) s += interleaved[(size_t)i * ch + c];
        audio_out[(size_t)i] = (float)(s / ch);
    }
    return true;
}

// =========================
// Write .npy (FP16) v1.0
// Shape: (1, seq_len, hidden_dim)
// =========================
static bool write_npy_fp16_3d(const std::string& path, const float* data, 
                               int d0, int d1, int d2) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;

    // Convert to FP16
    const size_t total_elements = (size_t)d0 * (size_t)d1 * (size_t)d2;
    std::vector<uint16_t> fp16_data(total_elements);
    for (size_t i = 0; i < total_elements; i++) {
        fp16_data[i] = float32_to_float16(data[i]);
    }

    // NPY header
    os.write("\x93NUMPY", 6);
    uint8_t ver[2] = {1, 0};
    os.write((char*)ver, 2);

    std::string header = "{'descr': '<f2', 'fortran_order': False, 'shape': (";
    header += std::to_string(d0) + ", " + std::to_string(d1) + ", " + std::to_string(d2) + "), }";

    int header_len = (int)header.size() + 1;
    int pad = 16 - ((10 + header_len) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');

    uint16_t hlen = (uint16_t)header.size();
    os.write((char*)&hlen, 2);
    os.write(header.data(), header.size());

    os.write((const char*)fp16_data.data(), total_elements * sizeof(uint16_t));
    return true;
}

// =========================
// Write .npy (INT64) v1.0
// Shape: (1, num_tokens)
// =========================
static bool write_npy_int64_2d(const std::string& path, const int64_t* data, 
                                int d0, int d1) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;

    os.write("\x93NUMPY", 6);
    uint8_t ver[2] = {1, 0};
    os.write((char*)ver, 2);

    std::string header = "{'descr': '<i8', 'fortran_order': False, 'shape': (";
    header += std::to_string(d0) + ", " + std::to_string(d1) + "), }";

    int header_len = (int)header.size() + 1;
    int pad = 16 - ((10 + header_len) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');

    uint16_t hlen = (uint16_t)header.size();
    os.write((char*)&hlen, 2);
    os.write(header.data(), header.size());

    const size_t n = (size_t)d0 * (size_t)d1;
    os.write((const char*)data, n * sizeof(int64_t));
    return true;
}

// =========================
// pad_x_mel
// =========================
static void pad_x_mel(const std::vector<float>& cur_x_mel,
                      int rows, int cols,
                      std::vector<float>& x_mel,
                      int cols_pad) {
    x_mel.assign((size_t)rows * (size_t)cols_pad, 0.0f);
    const int copy_cols = std::min(cols, cols_pad);
    for (int r = 0; r < rows; ++r) {
        const float* src = cur_x_mel.data() + (size_t)r * (size_t)cols;
        float* dst = x_mel.data() + (size_t)r * (size_t)cols_pad;
        std::memcpy(dst, src, (size_t)copy_cols * sizeof(float));
    }
}

// =========================
// log_mel_spectrogram
// =========================
static void log_mel_spectrogram(const float* audio, int audio_len,
                               int num_frames_of_stfts,
                               const float* mel_filters,
                               std::vector<float>& x_mel_out) {
    const int n_freqs = N_FFT / 2 + 1;
    const int frames = num_frames_of_stfts - 1;
    x_mel_out.assign((size_t)N_MELS * (size_t)frames, 0.0f);

    static std::vector<float> window = hann_window(WIN_LENGTH);

    std::vector<float> frame(WIN_LENGTH);
    std::vector<float> mag(n_freqs);

    for (int t = 0; t < frames; ++t) {
        const int start = t * HOP_LENGTH;

        for (int i = 0; i < WIN_LENGTH; ++i) {
            int idx = start + i;
            float s = (idx >= 0 && idx < audio_len) ? audio[idx] : 0.0f;
            frame[i] = s * window[i];
        }

        for (int k = 0; k < n_freqs; ++k) {
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < N_FFT; ++n) {
                float x = (n < WIN_LENGTH) ? frame[n] : 0.0f;
                float ang = -2.0f * PI_F * k * n / N_FFT;
                re += x * std::cos(ang);
                im += x * std::sin(ang);
            }
            mag[k] = re * re + im * im;
        }

        for (int m = 0; m < N_MELS; ++m) {
            double s = 0.0;
            const float* filt = mel_filters + (size_t)m * (size_t)n_freqs;
            for (int k = 0; k < n_freqs; ++k) s += (double)mag[k] * (double)filt[k];
            float v = (float)std::log10(std::max(s, 1e-10));
            x_mel_out[(size_t)m * (size_t)frames + (size_t)t] = v;
        }
    }

    float vmax = -1e9f;
    for (float v : x_mel_out) vmax = std::max(vmax, v);
    const float vmin = vmax - 8.0f;
    for (float& v : x_mel_out) v = std::max(v, vmin);
    for (float& v : x_mel_out) v = (v + 4.0f) / 4.0f;
}

// =========================
// audio_preprocess (from your code)
// =========================
void audio_preprocess(audio_buffer_t *audio, float *mel_filters, std::vector<float> &x_mel)
{
    int audio_length = audio->num_frames;
    std::vector<float> ori_audio_data(audio->data, audio->data + audio_length);

    if (audio_length >= MAX_AUDIO_LENGTH)
    {
        std::vector<float> trim_audio_data(MAX_AUDIO_LENGTH);
        std::copy(ori_audio_data.begin(), ori_audio_data.begin() + MAX_AUDIO_LENGTH, trim_audio_data.begin());
        int cur_num_frames_of_stfts = MAX_AUDIO_LENGTH / HOP_LENGTH + 1;
        log_mel_spectrogram(trim_audio_data.data(), MAX_AUDIO_LENGTH, cur_num_frames_of_stfts, mel_filters, x_mel);
    }
    else
    {
        int cur_num_frames_of_stfts = audio_length / HOP_LENGTH + 1;
        int x_mel_rows = N_MELS;
        int x_mel_cols = cur_num_frames_of_stfts - 1;
        int x_mel_cols_pad = MAX_AUDIO_LENGTH / HOP_LENGTH;
        std::vector<float> cur_x_mel(x_mel_rows * x_mel_cols, 0.0f);
        log_mel_spectrogram(ori_audio_data.data(), audio_length, cur_num_frames_of_stfts, mel_filters, cur_x_mel);
        pad_x_mel(cur_x_mel, x_mel_rows, x_mel_cols, x_mel, x_mel_cols_pad);
    }
}

// =========================
// Fake encoder: generate random FP32 output
// For real quantization, you should run actual encoder inference
// =========================
static void fake_encoder_output(std::vector<float>& encoder_out) {
    // For real usage, replace this with actual encoder inference
    // Here we just generate random values for demonstration
    static std::mt19937 rng(12345);
    static std::normal_distribution<float> dist(0.0f, 1.0f);
    
    encoder_out.resize(ENCODER_OUTPUT_SIZE);
    for (size_t i = 0; i < encoder_out.size(); ++i) {
        encoder_out[i] = dist(rng);
    }
}

// =========================
// Arg parsing
// =========================
struct Args {
    std::string librispeech_dir;
    std::string out_dir;
    int max_samples = 500;
    int seed = 1234;
    int chunk_seconds = 20;
    bool use_random = false; // If true, use random encoder output; if false, expect real encoder
};

static void usage(const char* prog) {
    std::cerr
        << "Usage:\n  " << prog
        << " --librispeech_dir <dir> --out_dir <dir> [--chunk_seconds 20|30]"
        << " [--max_samples N] [--seed S] [--use_random]\n"
        << "\n  --use_random: Use random encoder output (for testing)\n"
        << "                Otherwise expects real encoder inference (not implemented here)\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--librispeech_dir" && i + 1 < argc) a.librispeech_dir = argv[++i];
        else if (k == "--out_dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (k == "--max_samples" && i + 1 < argc) a.max_samples = std::stoi(argv[++i]);
        else if (k == "--seed" && i + 1 < argc) a.seed = std::stoi(argv[++i]);
        else if (k == "--chunk_seconds" && i + 1 < argc) a.chunk_seconds = std::stoi(argv[++i]);
        else if (k == "--use_random") a.use_random = true;
        else {
            usage(argv[0]);
            return false;
        }
    }
    if (a.librispeech_dir.empty() || a.out_dir.empty()) {
        usage(argv[0]);
        return false;
    }
    if (a.chunk_seconds != 20 && a.chunk_seconds != 30) {
        std::cerr << "chunk_seconds must be 20 or 30.\n";
        return false;
    }
    return true;
}

// =========================
// Main
// =========================
int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) return 1;

    CHUNK_SECONDS = args.chunk_seconds;
    MAX_AUDIO_LENGTH = CHUNK_SECONDS * SAMPLE_RATE;
    ENCODER_INPUT_SIZE = MAX_AUDIO_LENGTH / HOP_LENGTH;
    ENCODER_OUTPUT_SIZE = CHUNK_SECONDS * 50 * 512; // seq_len = chunk_seconds * 50

    const fs::path in_root = fs::path(args.librispeech_dir);
    const fs::path out_root = fs::path(args.out_dir);
    fs::create_directories(out_root);

    // collect flac files
    std::vector<fs::path> flacs;
    for (auto& p : fs::recursive_directory_iterator(in_root)) {
        if (!p.is_regular_file()) continue;
        if (p.path().extension() == ".flac") flacs.push_back(p.path());
    }
    if (flacs.empty()) {
        std::cerr << "No .flac found under: " << in_root << "\n";
        return 2;
    }

    // shuffle
    std::mt19937 rng(args.seed);
    std::shuffle(flacs.begin(), flacs.end(), rng);
    if ((int)flacs.size() > args.max_samples) flacs.resize(args.max_samples);

    // mel filters
    std::vector<float> mel_filters_vec = build_mel_filters();
    float* mel_filters = mel_filters_vec.data();

    // Fixed tokens (decoder input prompt): [1, 12]
    // Typical Whisper decoder prompt: [sot, task, notimestamps, ...]
    int64_t fixed_tokens[12] = {
        50258, 50259, 50359, 50363,  // typical whisper start tokens
        50364, 50365, 50366, 50367,
        50368, 50369, 50370, 50371
    };

    // open dataset.txt
    std::ofstream ds(out_root / "dataset.txt");
    if (!ds) {
        std::cerr << "Failed to open dataset.txt for writing\n";
        return 3;
    }

    // process
    int kept = 0;
    for (size_t i = 0; i < flacs.size(); ++i) {
        const std::string path = flacs[i].string();

        std::vector<float> audio;
        int sr = 0;
        if (!read_audio_mono(path, audio, sr)) continue;

        if (sr != SAMPLE_RATE) {
            audio = resample_linear(audio, sr, SAMPLE_RATE);
            sr = SAMPLE_RATE;
        }

        audio_buffer_t buf;
        buf.data = audio.data();
        buf.num_frames = (int)audio.size();

        // Generate mel spectrogram (not used for decoder, but ensures audio is valid)
        std::vector<float> x_mel;
        audio_preprocess(&buf, mel_filters, x_mel);

        if ((int)x_mel.size() != N_MELS * ENCODER_INPUT_SIZE) {
            std::cerr << "Skip (unexpected mel size): " << path << "\n";
            continue;
        }

        // Generate encoder output (fake or real)
        std::vector<float> encoder_output;
        if (args.use_random) {
            fake_encoder_output(encoder_output);
        } else {
            // TODO: Here you should call your real encoder inference
            // For now, we'll use fake data with a warning
            std::cerr << "WARNING: Using random encoder output. For real quantization,\n"
                      << "         implement actual encoder inference here.\n";
            fake_encoder_output(encoder_output);
        }

        // Save tokens.npy: shape [1, 12], type INT64
        const std::string tokens_name = (out_root / ("tokens_" + std::to_string(kept) + ".npy")).string();
        if (!write_npy_int64_2d(tokens_name, fixed_tokens, 1, 12)) {
            std::cerr << "Failed to write tokens npy: " << tokens_name << "\n";
            continue;
        }

        // Save audio.npy: shape [1, seq_len, 512], type FP16
        const int seq_len = CHUNK_SECONDS * 50;
        const std::string audio_name = (out_root / ("audio_" + std::to_string(kept) + ".npy")).string();
        if (!write_npy_fp16_3d(audio_name, encoder_output.data(), 1, seq_len, 512)) {
            std::cerr << "Failed to write audio npy: " << audio_name << "\n";
            continue;
        }

        // Write to dataset.txt: tokens.npy audio.npy
        ds << fs::absolute(tokens_name).string() << " " 
           << fs::absolute(audio_name).string() << "\n";
        kept++;

        if (kept % 50 == 0) {
            std::cout << "Generated " << kept << " samples...\n";
        }
    }

    std::cout << "\nDone. Generated " << kept << " decoder calibration samples.\n";
    std::cout << "dataset.txt: " << fs::absolute(out_root / "dataset.txt") << "\n";
    std::cout << "Expected decoder input shapes:\n";
    std::cout << "  tokens: (1, 12) INT64\n";
    std::cout << "  audio:  (1, " << (CHUNK_SECONDS * 50) << ", 512) FP16\n";
    return 0;
}
