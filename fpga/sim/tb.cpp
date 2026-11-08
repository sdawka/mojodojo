// Verilator co-simulation harness for the generated systolic tile.
//
// Replays a JSONL golden trace written by `pixi run trace -- --format jsonl`
// (DESIGN.md section 6) against the Verilated RTL. For every line: drive the
// inputs with clk low, eval(), compare EVERY output field (and the dbg st/cnt
// registers) with the trace, then raise clk and eval() (the rising edge).
//
// Build-time macros, from a per-configuration tb_config.h written by sim/cosim.sh:
//   TOP      Verilated class, e.g. Vsystolic_n4_i8_a32
//   TOPHDR   its header, e.g. "Vsystolic_n4_i8_a32.h"
//   TOPROOT  its root-class header, e.g. "Vsystolic_n4_i8_a32___024root.h"
//   DBG_FSM / DBG_CNT  rootp member names of the fsm / cnt registers
//   TILE_N, IN_BITS, ACC_BITS
//
// Usage: tb <trace.jsonl> [--quiet]
// Exit: 0 all cycles match, 1 mismatch (first one reported), 2 usage/IO/parse.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "tb_config.h"
#include "verilated.h"
#include TOPHDR
#include TOPROOT

namespace {

// ---- minimal parser for the fixed trace schema ----
bool find_key(const std::string& s, size_t from, const char* key, size_t& pos) {
    std::string pat = std::string("\"") + key + "\":";
    size_t p = s.find(pat, from);
    if (p == std::string::npos) return false;
    pos = p + pat.size();
    return true;
}

bool parse_int(const std::string& s, size_t& pos, int64_t& v) {
    char* end = nullptr;
    v = std::strtoll(s.c_str() + pos, &end, 10);
    if (end == s.c_str() + pos) return false;
    pos = static_cast<size_t>(end - s.c_str());
    return true;
}

bool get_int(const std::string& s, size_t from, const char* key, int64_t& v) {
    size_t p;
    return find_key(s, from, key, p) && parse_int(s, p, v);
}

bool get_arr(const std::string& s, size_t from, const char* key,
             std::vector<int64_t>& out) {
    size_t p;
    if (!find_key(s, from, key, p) || s[p] != '[') return false;
    ++p;
    out.clear();
    while (p < s.size() && s[p] != ']') {
        int64_t v;
        if (!parse_int(s, p, v)) return false;
        out.push_back(v);
        if (s[p] == ',') ++p;
    }
    return p < s.size();
}

struct Line {
    int64_t cyc, rst, start, k_len, in_valid, out_ready;
    std::vector<int64_t> a_col, b_row;
    int64_t idle, in_ready, out_valid, out_last;
    std::vector<int64_t> out_row;
    int64_t st, cnt;
    bool has_dbg;
};

bool parse_line(const std::string& s, Line& l) {
    size_t in_pos, out_pos, dbg_pos;
    if (!get_int(s, 0, "cyc", l.cyc)) return false;
    if (!find_key(s, 0, "in", in_pos) || !find_key(s, 0, "out", out_pos)) return false;
    bool ok = get_int(s, in_pos, "rst", l.rst) && get_int(s, in_pos, "start", l.start) &&
              get_int(s, in_pos, "k_len", l.k_len) &&
              get_int(s, in_pos, "in_valid", l.in_valid) &&
              get_int(s, in_pos, "out_ready", l.out_ready) &&
              get_arr(s, in_pos, "a_col", l.a_col) && get_arr(s, in_pos, "b_row", l.b_row) &&
              get_int(s, out_pos, "idle", l.idle) &&
              get_int(s, out_pos, "in_ready", l.in_ready) &&
              get_int(s, out_pos, "out_valid", l.out_valid) &&
              get_int(s, out_pos, "out_last", l.out_last) &&
              get_arr(s, out_pos, "out_row", l.out_row);
    l.has_dbg = find_key(s, 0, "dbg", dbg_pos) && get_int(s, dbg_pos, "st", l.st) &&
                get_int(s, dbg_pos, "cnt", l.cnt);
    return ok && l.a_col.size() == TILE_N && l.b_row.size() == TILE_N &&
           l.out_row.size() == TILE_N;
}

// ---- packed-bus access, element 0 in the low bits ----
template <typename T>
void put_field(T& sig, int lsb, int width, int64_t v) {
    for (int b = 0; b < width; ++b) {
        T bit = static_cast<T>((static_cast<uint64_t>(v) >> b) & 1u);
        sig = static_cast<T>((sig & ~(static_cast<T>(1) << (lsb + b))) | (bit << (lsb + b)));
    }
}
template <std::size_t W>
void put_field(VlWide<W>& sig, int lsb, int width, int64_t v) {
    for (int b = 0; b < width; ++b) {
        int pos = lsb + b;
        uint32_t bit = static_cast<uint32_t>((static_cast<uint64_t>(v) >> b) & 1u);
        sig[pos / 32] = (sig[pos / 32] & ~(1u << (pos % 32))) | (bit << (pos % 32));
    }
}
template <typename T>
int64_t get_field(const T& sig, int lsb, int width) {
    uint64_t u = 0;
    for (int b = 0; b < width; ++b) u |= static_cast<uint64_t>((sig >> (lsb + b)) & 1u) << b;
    if (width < 64 && (u >> (width - 1)) & 1u) u |= ~((uint64_t{1} << width) - 1);  // sext
    return static_cast<int64_t>(u);
}
template <std::size_t W>
int64_t get_field(const VlWide<W>& sig, int lsb, int width) {
    uint64_t u = 0;
    for (int b = 0; b < width; ++b) {
        int pos = lsb + b;
        u |= static_cast<uint64_t>((sig[pos / 32] >> (pos % 32)) & 1u) << b;
    }
    if (width < 64 && (u >> (width - 1)) & 1u) u |= ~((uint64_t{1} << width) - 1);
    return static_cast<int64_t>(u);
}

std::string vec_str(const std::vector<int64_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s trace.jsonl [--quiet]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    bool quiet = argc > 2 && std::strcmp(argv[2], "--quiet") == 0;
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 2;
    }

    VerilatedContext ctx;
    TOP top(&ctx);
    top.clk = 0;
    top.eval();

    std::string text, prev_text;
    Line l{};
    long n = 0, lineno = 0;
    while (std::getline(f, text)) {
        ++lineno;
        if (text.empty() || text[0] == '#') continue;
        if (!parse_line(text, l)) {
            std::fprintf(stderr, "%s:%ld: cannot parse trace line\n", path, lineno);
            return 2;
        }
        if (l.cyc != n) {
            std::fprintf(stderr, "%s:%ld: expected cyc %ld, got %lld\n", path, lineno, n,
                         static_cast<long long>(l.cyc));
            return 2;
        }
        // Drive inputs for this cycle (clk low), settle combinational logic.
        top.rst = l.rst;
        top.start = l.start;
        top.k_len = static_cast<uint16_t>(l.k_len);
        top.in_valid = l.in_valid;
        top.out_ready = l.out_ready;
        for (int i = 0; i < TILE_N; ++i) {
            put_field(top.a_col, i * IN_BITS, IN_BITS, l.a_col[i]);
            put_field(top.b_row, i * IN_BITS, IN_BITS, l.b_row[i]);
        }
        top.clk = 0;
        top.eval();

        // Compare every output field, plus the dbg registers.
        std::string err;
        auto chk = [&](const char* name, int64_t got, int64_t want) {
            if (got != want && err.empty())
                err = std::string(name) + ": rtl=" + std::to_string(got) +
                      " trace=" + std::to_string(want);
        };
        chk("idle", top.idle, l.idle);
        chk("in_ready", top.in_ready, l.in_ready);
        chk("out_valid", top.out_valid, l.out_valid);
        chk("out_last", top.out_last, l.out_last);
        std::vector<int64_t> row(TILE_N);
        for (int j = 0; j < TILE_N; ++j) {
            row[j] = get_field(top.out_row, j * ACC_BITS, ACC_BITS);
            std::string nm = "out_row[" + std::to_string(j) + "]";
            chk(nm.c_str(), row[j], l.out_row[j]);
        }
        int64_t st = top.rootp->DBG_FSM, cnt = top.rootp->DBG_CNT;
        if (l.has_dbg) {
            chk("dbg.st", st, l.st);
            chk("dbg.cnt", cnt, l.cnt);
        }
        if (!err.empty()) {
            std::fprintf(stderr,
                         "MISMATCH %s at cyc %ld (line %ld): %s\n"
                         "  rtl : idle=%d in_ready=%d out_valid=%d out_last=%d st=%lld cnt=%lld "
                         "out_row=%s\n"
                         "  this: %s\n  prev: %s\n",
                         path, n, lineno, err.c_str(), top.idle, top.in_ready, top.out_valid,
                         top.out_last, static_cast<long long>(st), static_cast<long long>(cnt),
                         vec_str(row).c_str(), text.c_str(),
                         prev_text.empty() ? "(none)" : prev_text.c_str());
            return 1;
        }
        // Rising edge.
        top.clk = 1;
        top.eval();
        ++n;
        prev_text.swap(text);
    }
    top.final();
    if (n == 0) {
        std::fprintf(stderr, "%s: empty trace\n", path);
        return 2;
    }
    if (!quiet) std::printf("PASS %s: %ld cycles, N=%d\n", path, n, TILE_N);
    return 0;
}
