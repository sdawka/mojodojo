// Verilator harness for tile v2 (fpga/tile2/INTERFACE.md).
//
// Replays a register-level program written by tile2/prog.mojo (the Mojo
// bus-level model) against the generated tile2_n{N}. Acts as a Wishbone
// classic master: for each op it idles `gap` cycles (cyc=stb=0, checking
// ack stays low), then drives one request cycle (checks ack=0 and the irq
// level the model expects) and one ACK cycle (checks ack=1 and, for reads,
// wb_dat_r). Cycle counting starts after one initial reset cycle; at END the
// total must equal the model's. Any difference in timing (BUSY length,
// CYCLES, DONE/irq edges) shows up as a read or irq mismatch.
//
// Build-time macros (tb_config.h): TOP, TOPHDR.
// Usage: tile2_tb <prog.txt> [--quiet]
// Exit: 0 pass, 1 mismatch, 2 usage/IO/parse.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "tb_config.h"
#include "verilated.h"
#include TOPHDR

namespace {

TOP* dut;
uint64_t cyc = 0;  // cycles since the end of the initial reset
int line_no = 0;
std::string cur_line;

void tick() {
    dut->clk = 1;
    dut->eval();
    dut->clk = 0;
    dut->eval();
}

[[noreturn]] void fail(const char* what, uint64_t got, uint64_t exp) {
    std::printf("MISMATCH line %d at cyc %llu: %s: rtl=0x%llx expected=0x%llx\n  %s\n",
                line_no, (unsigned long long)cyc, what, (unsigned long long)got,
                (unsigned long long)exp, cur_line.c_str());
    std::exit(1);
}

void idle(int n) {
    dut->wb_cyc = 0;
    dut->wb_stb = 0;
    dut->wb_we = 0;
    for (int i = 0; i < n; ++i) {
        dut->eval();
        if (dut->wb_ack) fail("spurious ack", dut->wb_ack, 0);
        tick();
        ++cyc;
    }
}

uint32_t access(bool we, uint32_t addr, uint32_t data, int sel, int irq_exp) {
    dut->wb_cyc = 1;
    dut->wb_stb = 1;
    dut->wb_we = we;
    dut->wb_adr = (addr >> 2) & 0x7fff;
    dut->wb_dat_w = data;
    dut->wb_sel = sel;
    dut->eval();
    if (dut->wb_ack) fail("ack in request cycle", 1, 0);
    if (dut->irq != irq_exp) fail("irq", dut->irq, irq_exp);
    tick();
    ++cyc;
    dut->eval();
    if (!dut->wb_ack) fail("no ack in second cycle", 0, 1);
    uint32_t r = dut->wb_dat_r;
    tick();
    ++cyc;
    dut->wb_cyc = 0;
    dut->wb_stb = 0;
    dut->wb_we = 0;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <prog.txt> [--quiet]\n", argv[0]);
        return 2;
    }
    bool quiet = argc > 2 && std::strcmp(argv[2], "--quiet") == 0;
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    Verilated::commandArgs(1, argv);
    dut = new TOP;
    dut->clk = 0;
    dut->rst = 1;
    dut->eval();
    tick();  // initial reset cycle (not counted)
    dut->rst = 0;

    uint64_t n_rd = 0, n_wr = 0, runs = 0;
    bool ended = false;
    while (std::getline(in, cur_line)) {
        ++line_no;
        if (cur_line.empty() || cur_line[0] == '#') continue;
        std::istringstream ss(cur_line);
        std::string op;
        ss >> op;
        if (op == "W") {
            int gap, sel, irq;
            std::string a, d;
            ss >> gap >> a >> d >> sel >> irq;
            idle(gap);
            access(true, std::stoul(a, nullptr, 16), std::stoul(d, nullptr, 16), sel, irq);
            ++n_wr;
        } else if (op == "R") {
            int gap, irq;
            std::string a, e;
            ss >> gap >> a >> e >> irq;
            idle(gap);
            uint32_t exp = std::stoul(e, nullptr, 16);
            uint32_t got = access(false, std::stoul(a, nullptr, 16), 0, 0xf, irq);
            if (got != exp) fail("wb_dat_r", got, exp);
            ++n_rd;
        } else if (op == "X") {
            int gap;
            ss >> gap;
            idle(gap);
            dut->rst = 1;
            dut->eval();
            tick();
            ++cyc;
            dut->rst = 0;
        } else if (op == "END") {
            unsigned long long total;
            ss >> total >> runs;
            if (cyc != total) fail("total cycles", cyc, total);
            ended = true;
            break;
        } else {
            std::fprintf(stderr, "parse error line %d: %s\n", line_no, cur_line.c_str());
            return 2;
        }
    }
    if (!ended) {
        std::fprintf(stderr, "missing END\n");
        return 2;
    }
    if (!quiet)
        std::printf("PASS %s: %llu cycles, %llu writes, %llu reads, %llu runs\n", argv[1],
                    (unsigned long long)cyc, (unsigned long long)n_wr,
                    (unsigned long long)n_rd, (unsigned long long)runs);
    else
        std::printf("%llu %llu %llu %llu\n", (unsigned long long)cyc,
                    (unsigned long long)n_wr, (unsigned long long)n_rd,
                    (unsigned long long)runs);
    delete dut;
    return 0;
}
