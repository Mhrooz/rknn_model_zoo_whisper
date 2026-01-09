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
static constexpr int N_FFT         = 400;   // whisper uses 400
static constexpr int HOP_LENGTH    = 160;   // whisper uses 160
static constexpr int WIN_LENGTH    = 400;   // whisper uses 400
static constexpr int N_MELS        = 80;
static constexpr float PI_F        = 3.14159265358979323846f;

// user controls this (20s -> 2000 frames; 30s -> 3000 frames)
static int CHUNK_SECONDS = 20;
static int MAX_AUDIO_LENGTH = 20 * SAMPLE_RATE; // in samples
static int ENCODER_INPUT_SIZE = (20 * SAMPLE_RATE) / HOP_LENGTH; // frames (2000)

// =========================
// Minimal audio buffer type
// =========================
typedef struct audio_buffer_t {
    float* data = nullptr;
    int num_frames = 0; // number of samples
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
// mel_filters shape: [N_MELS, (N_FFT/2 + 1)]
// =========================
static std::vector<float> build_mel_filters() {
    const int n_freqs = N_FFT / 2 + 1;
    std::vector<float> filters(N_MELS * n_freqs, 0.0f);

    const float f_min = 0.0f;
    const float f_max = SAMPLE_RATE / 2.0f;

    const float m_min = hz_to_mel(f_min);
    const float m_max = hz_to_mel(f_max);

    // mel points
    std::vector<float> m_pts(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        m_pts[i] = m_min + (m_max - m_min) * i / (N_MELS + 1);
    }

    // hz points
    std::vector<float> f_pts(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        f_pts[i] = mel_to_hz(m_pts[i]);
    }

    // bin points
    std::vector<int> bins(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; ++i) {
        // map frequency to FFT bin
        float bin = (N_FFT + 1) * f_pts[i] / SAMPLE_RATE;
        int b = static_cast<int>(std::floor(bin));
        if (b < 0) b = 0;
        if (b > n_freqs - 1) b = n_freqs - 1;
        bins[i] = b;
    }

    // triangular filters
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
// Simple linear resampler (only if needed)
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
// Read audio via libsndfile (FLAC supported)
// Returns mono float in [-1,1], sr in out_sr
// =========================
static bool read_audio_mono(const std::string& path, std::vector<float>& audio_out, int& sr_out) {
    SF_INFO info;
    std::memset(&info, 0, sizeof(info));
    SNDFILE* f = sf_open(path.c_str(), SFM_READ, &info);
    if (!f) {
        std::cerr << "Failed to open audio: " << path << " err=" << sf_strerror(nullptr) << "\n";
        return false;
    }

    sr_out = info.samplerate;
    const int ch = info.channels;
    const sf_count_t nframes = info.frames;
    std::vector<float> interleaved((size_t)nframes * (size_t)ch);

    sf_count_t nread = sf_readf_float(f, interleaved.data(), nframes);
    sf_close(f);

    if (nread != nframes) {
        std::cerr << "Warning: short read: " << path << " read=" << nread << " expect=" << nframes << "\n";
    }

    // downmix to mono
    audio_out.resize((size_t)nread);
    for (sf_count_t i = 0; i < nread; ++i) {
        double s = 0.0;
        for (int c = 0; c < ch; ++c) s += interleaved[(size_t)i * ch + c];
        audio_out[(size_t)i] = (float)(s / ch);
    }
    return true;
}

// =========================
// Write .npy (float32) v1.0
// Shape here: (1, 80, frames)
// =========================
static bool write_npy_f32_3d(const std::string& path, const float* data, int d0, int d1, int d2) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;

    // magic + version
    os.write("\x93NUMPY", 6);
    uint8_t ver[2] = {1, 0};
    os.write((char*)ver, 2);

    // header dict
    // little-endian float32: <f4, fortran_order False
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    header += std::to_string(d0) + ", " + std::to_string(d1) + ", " + std::to_string(d2) + "), }";

    // pad header to 16-byte alignment incl. newline
    // header_len stored as uint16 (little-endian)
    int header_len = (int)header.size() + 1; // + '\n'
    int pad = 16 - ((10 + header_len) % 16);
    if (pad == 16) pad = 0;
    header.append(pad, ' ');
    header.push_back('\n');

    uint16_t hlen = (uint16_t)header.size();
    os.write((char*)&hlen, 2);
    os.write(header.data(), header.size());

    // data
    const size_t n = (size_t)d0 * (size_t)d1 * (size_t)d2;
    os.write((const char*)data, n * sizeof(float));
    return true;
}

// =========================
// pad_x_mel: pad mel to fixed frames (ENCODER_INPUT_SIZE)
// cur_x_mel shape: [rows=80, cols=cur_frames]
// out x_mel shape: [rows=80, cols_pad=ENCODER_INPUT_SIZE]
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
// log_mel_spectrogram (Whisper-like; not bit-exact to PyTorch, but consistent with itself)
// Produces x_mel in row-major [80, frames], frames = num_frames_of_stfts - 1
// =========================
static void log_mel_spectrogram(const float* audio, int audio_len,
                               int num_frames_of_stfts,
                               const float* mel_filters,
                               std::vector<float>& x_mel_out) {
    const int n_freqs = N_FFT / 2 + 1;
    const int frames = num_frames_of_stfts - 1; // matches your audio_preprocess usage
    x_mel_out.assign((size_t)N_MELS * (size_t)frames, 0.0f);

    static std::vector<float> window = hann_window(WIN_LENGTH);

    // For each frame: take WIN_LENGTH samples starting at i*HOP_LENGTH, zero-pad if out of range
    std::vector<float> frame(WIN_LENGTH);
    std::vector<float> mag(n_freqs);

    for (int t = 0; t < frames; ++t) {
        const int start = t * HOP_LENGTH;

        for (int i = 0; i < WIN_LENGTH; ++i) {
            int idx = start + i;
            float s = (idx >= 0 && idx < audio_len) ? audio[idx] : 0.0f;
            frame[i] = s * window[i];
        }

        // naive DFT for real signal at needed bins (N_FFT=400, frames <= 3000, still OK for dataset gen)
        // If you need speed, replace with FFT library (FFTW).
        for (int k = 0; k < n_freqs; ++k) {
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < N_FFT; ++n) {
                float x = (n < WIN_LENGTH) ? frame[n] : 0.0f;
                float ang = -2.0f * PI_F * k * n / N_FFT;
                re += x * std::cos(ang);
                im += x * std::sin(ang);
            }
            float p = re * re + im * im; // power
            mag[k] = p;
        }

        // mel projection: mel[m] = sum_k mag[k] * filter[m,k]
        for (int m = 0; m < N_MELS; ++m) {
            double s = 0.0;
            const float* filt = mel_filters + (size_t)m * (size_t)n_freqs;
            for (int k = 0; k < n_freqs; ++k) s += (double)mag[k] * (double)filt[k];
            // log compression
            float v = (float)std::log10(std::max(s, 1e-10));
            x_mel_out[(size_t)m * (size_t)frames + (size_t)t] = v;
        }
    }

    // Whisper-like dynamic range clamp + normalize (critical for quantization stability)
    // This step is typically what makes mel ranges "small" and quantization-friendly.
    // Clamp relative to max
    float vmax = -1e9f;
    for (float v : x_mel_out) vmax = std::max(vmax, v);
    const float vmin = vmax - 8.0f;
    for (float& v : x_mel_out) v = std::max(v, vmin);

    // Normalize (keep same transform across calibration/inference)
    for (float& v : x_mel_out) v = (v + 4.0f) / 4.0f;
}

// =========================
// Your provided audio_preprocess (kept as-is)
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
// Arg parsing
// =========================
struct Args {
    std::string librispeech_dir;
    std::string out_dir;
    int max_samples = 500;
    int seed = 1234;
    int chunk_seconds = 20;
};

static void usage(const char* prog) {
    std::cerr
        << "Usage:\n  " << prog
        << " --librispeech_dir <dir> --out_dir <dir> [--chunk_seconds 20|30] [--max_samples N] [--seed S]\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--librispeech_dir" && i + 1 < argc) a.librispeech_dir = argv[++i];
        else if (k == "--out_dir" && i + 1 < argc) a.out_dir = argv[++i];
        else if (k == "--max_samples" && i + 1 < argc) a.max_samples = std::stoi(argv[++i]);
        else if (k == "--seed" && i + 1 < argc) a.seed = std::stoi(argv[++i]);
        else if (k == "--chunk_seconds" && i + 1 < argc) a.chunk_seconds = std::stoi(argv[++i]);
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

        // resample if necessary
        if (sr != SAMPLE_RATE) {
            audio = resample_linear(audio, sr, SAMPLE_RATE);
            sr = SAMPLE_RATE;
        }

        audio_buffer_t buf;
        buf.data = audio.data();
        buf.num_frames = (int)audio.size();

        std::vector<float> x_mel;
        audio_preprocess(&buf, mel_filters, x_mel);

        // Expect fixed size: 80 * ENCODER_INPUT_SIZE
        if ((int)x_mel.size() != N_MELS * ENCODER_INPUT_SIZE) {
            std::cerr << "Skip (unexpected mel size): " << path
                      << " got=" << x_mel.size()
                      << " expect=" << (N_MELS * ENCODER_INPUT_SIZE) << "\n";
            continue;
        }

        // Save as (1,80,frames)
        const std::string npy_name = (out_root / ("mel_" + std::to_string(kept) + ".npy")).string();
        if (!write_npy_f32_3d(npy_name, x_mel.data(), 1, N_MELS, ENCODER_INPUT_SIZE)) {
            std::cerr << "Failed to write npy: " << npy_name << "\n";
            continue;
        }

        // absolute path to dataset.txt
        ds << fs::absolute(npy_name).string() << "\n";
        kept++;

        if (kept % 50 == 0) {
            std::cout << "Generated " << kept << " samples...\n";
        }
    }

    std::cout << "Done. Generated " << kept << " samples.\n";
    std::cout << "dataset.txt: " << fs::absolute(out_root / "dataset.txt") << "\n";
    std::cout << "Expected model input shape: (1, " << N_MELS << ", " << ENCODER_INPUT_SIZE << ")\n";
    return 0;
}
