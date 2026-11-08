"""A tiny netlist IR and its Verilog-2005 emitter.

Everything in this file is ordinary Mojo code. `rtlgen.systolic` calls it from
`comptime` declarations, so the compiler's interpreter runs it during
compilation and the resulting `String` is baked into the binary as a constant.
None of these functions are present in the compiled `rtlgen` executable.

The IR covers the datapath only: ports, combinational wires (one continuous
assignment each) and clocked registers with a common shape:

    always @(posedge clk)
      if (clear)       r <= 0;
      else if (adv)    r <= adv_expr;     # empty expr: no update under adv
      else if (shift)  r <= shift_expr;   # empty expr: no update under shift

Control logic (the FSM) is passed through as a pre-rendered block.
"""

comptime DIR_IN = 0
comptime DIR_OUT = 1


@fieldwise_init
struct Port(Copyable, Movable):
    var name: String
    var dir: Int  # DIR_IN / DIR_OUT
    var width: Int


@fieldwise_init
struct Wire(Copyable, Movable):
    var name: String
    var width: Int
    var signed: Bool
    var expr: String
    var role: String  # free-form tag, used by comptime structural checks


@fieldwise_init
struct Reg(Copyable, Movable):
    var name: String
    var width: Int
    var signed: Bool
    var adv_expr: String
    var shift_expr: String
    var role: String


struct Module(Copyable, Movable):
    var name: String
    var header: List[String]  # comment lines at the top of the file
    var ports: List[Port]
    var params: List[String]  # rendered `localparam` lines
    var wires: List[Wire]
    var regs: List[Reg]
    var control: String  # pre-rendered control block (FSM)
    var clear: String  # datapath synchronous-clear condition
    var adv: String  # datapath clock-enable condition
    var shift: String  # drain-shift condition

    def __init__(out self, name: String):
        self.name = name
        self.header = List[String]()
        self.ports = List[Port]()
        self.params = List[String]()
        self.wires = List[Wire]()
        self.regs = List[Reg]()
        self.control = String()
        self.clear = String()
        self.adv = String()
        self.shift = String()

    def reg_bits(self, role: String) -> Int:
        var total = 0
        for r in self.regs:
            if r.role == role:
                total += r.width
        return total

    def count_wires(self, role: String) -> Int:
        var total = 0
        for w in self.wires:
            if w.role == role:
                total += 1
        return total

    def count_regs(self, role: String) -> Int:
        var total = 0
        for r in self.regs:
            if r.role == role:
                total += 1
        return total


def range_decl(width: Int, signed: Bool) -> String:
    var s = String("signed ") if signed else String()
    if width == 1 and not signed:
        return s^
    return String(s, "[", width - 1, ":0] ")


def emit_verilog(m: Module) -> String:
    """Render a `Module` as one Verilog-2005 file."""
    var v = String()
    for line in m.header:
        v += String("// ", line, "\n")
    v += "`default_nettype none\n\n"
    v += String("module ", m.name, " (\n")
    var first = True
    for p in m.ports:
        if not first:
            v += ",\n"
        first = False
        var dir = String("input  wire ") if p.dir == DIR_IN else String(
            "output wire "
        )
        v += String("    ", dir, range_decl(p.width, False), p.name)
    v += "\n);\n\n"
    for p in m.params:
        v += String("    ", p, "\n")
    v += "\n"
    v += m.control
    v += "\n    // ---- datapath registers ----\n"
    # The spec keeps a_reg in the last column and b_reg in the last row even
    # though nothing reads them (synthesis prunes them); silence the lint.
    v += "    /* verilator lint_off UNUSEDSIGNAL */\n"
    for r in m.regs:
        v += String("    reg ", range_decl(r.width, r.signed), r.name, ";\n")
    v += "    /* verilator lint_on UNUSEDSIGNAL */\n"
    v += "\n    // ---- datapath wires ----\n"
    for w in m.wires:
        v += String(
            "    wire ", range_decl(w.width, w.signed), w.name, " = ", w.expr, ";\n"
        )
    v += "\n    always @(posedge clk) begin\n"
    v += String("        if (", m.clear, ") begin\n")
    for r in m.regs:
        v += String("            ", r.name, " <= ", r.width, "'d0;\n")
    v += "        end else begin\n"
    v += String("            if (", m.adv, ") begin\n")
    for r in m.regs:
        if r.adv_expr.byte_length() > 0:
            v += String("                ", r.name, " <= ", r.adv_expr, ";\n")
    v += String("            end else if (", m.shift, ") begin\n")
    for r in m.regs:
        if r.shift_expr.byte_length() > 0:
            v += String("                ", r.name, " <= ", r.shift_expr, ";\n")
    v += "            end\n"
    v += "        end\n"
    v += "    end\n\nendmodule\n\n`default_nettype wire\n"
    return v^
