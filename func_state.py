import ast
from collections import Iterable
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Callable, Generator

from op_code import OpCode, MAXARG_sBx, _ENV
from vm.lua_value import LuaValue, UpValue


class LocalVarInfo:
    def __init__(self,
                 name: str,
                 prev: Optional["LocalVarInfo"],
                 scope_level: int,
                 slot: int,
                 ):
        self.prev = prev
        self.name = name
        self.scope_level = scope_level
        self.slot = slot
        self.start_pc = 0
        self.end_pc = 0
        self.captured: bool = False


class UpValueInfo:
    def __init__(self,
                 loc_var_slot: int,
                 up_value_index: int,
                 index: int,
                 ):
        self.loc_var_slot = loc_var_slot
        self.up_value_index = up_value_index
        self.index = index


class FuncState:
    arith_and_bitwise_binops = {
        ast.Add: OpCode.ADD,
        ast.Sub: OpCode.SUB,
        ast.Mult: OpCode.MUL,
        ast.Mod: OpCode.MOD,
        ast.Pow: OpCode.POW,
        ast.Div: OpCode.DIV,
        ast.FloorDiv: OpCode.IDIV,
        ast.BitAnd: OpCode.BAND,
        ast.BitOr: OpCode.BOR,
        ast.BitXor: OpCode.BXOR,
        ast.LShift: OpCode.SHL,
        ast.RShift: OpCode.SHR
    }

    def __init__(self,
                 parent: Optional["FuncState"] = None,
                 params_num: int = 0,
                 is_vararg: bool = True,
                 ):
        self.parent = parent
        self.sub_funcs: List["FuncState"] = []
        self.used_regs: int = 0
        self.max_regs: int = 0
        self.scope_level: int = 0
        self.local_vars: List[LocalVarInfo] = []
        self.local_names: Dict[str, Optional[LocalVarInfo]] = {}
        self.up_values: Dict[str, UpValueInfo] = {}
        self.constants: Dict[Any, int] = {}
        self.breaks: List[Optional[List[int]]] = [None]
        self.continues: List[Optional[List[int]]] = [None]
        self.insts: List[int] = []
        self.params_num = params_num
        self.is_vararg: bool = is_vararg

    # Code

    @property
    def pc(self):
        return len(self.insts) - 1

    def fill_sbx(self, pc: int, sbx: int):
        i = self.insts[pc]
        i = ((i << 18) & 0xFFFFFFFF) >> 18  # clear sBx
        i = (i | (sbx + MAXARG_sBx) << 14) & 0xFFFFFFFF  # reset sBx
        self.insts[pc] = i

    # Constants

    def index_of_constant(self, k: Any, opcode_index=True) -> int:
        flag = 0x100 if opcode_index else 0
        return flag + self.constants.setdefault(k, len(self.constants))

    # Registers

    def incr_regs(self, n: int = 1) -> int:
        if self.used_regs + n >= 255:
            raise Exception("too many registers")
        self.used_regs += n
        if self.used_regs > self.max_regs:
            self.max_regs = self.used_regs
        return self.used_regs - n

    def decr_regs(self, n: int = 1):
        if n < 0 or self.used_regs - n < 0:
            raise ValueError("n value is outside the valid range")
        self.used_regs -= n

    @contextmanager
    def assign_reg(self) -> Generator[int, None, None]:
        yield self.incr_regs()
        self.decr_regs()

    # Lexical scope

    def enter_scope(self, is_for: bool):
        self.scope_level += 1
        if is_for:
            self.breaks.append([])
        else:
            self.breaks.append(None)

    def leave_scope(self, fill_c_or_b_jmp=True):
        if fill_c_or_b_jmp:
            self.fill_c_or_b_jmp(self.breaks)

        self.scope_level -= 1
        for v in self.local_names.values():
            if v.scope_level > self.scope_level:
                self.remove_local_var(v)

    @contextmanager
    def scope(self, is_for=False):
        self.enter_scope(is_for)
        yield
        if not is_for:
            self.close_opened_up_values()
        self.leave_scope()

    # Local var

    def add_local_var(self, name: str) -> int:
        prev = self.local_names.get(name)
        slot = self.incr_regs()
        new_var = LocalVarInfo(name, prev, self.scope_level, slot)
        self.local_vars.append(new_var)
        self.local_names[name] = new_var
        return slot

    def remove_local_var(self, loc_var: LocalVarInfo):
        self.decr_regs()
        while loc_var.prev and loc_var.prev.scope_level == loc_var.scope_level:
            self.decr_regs()
            loc_var = loc_var.prev
        if loc_var.prev is None:
            self.local_names.pop(loc_var.name)
        else:
            self.local_names[loc_var.name] = loc_var.prev

    def slot_of_local_var(self, name):
        if name in self.local_names:
            return self.local_names[name].slot
        return -1

    # Up value

    def index_of_up_value(self, name):
        if name in self.up_values:
            return self.up_values[name].index

        if self.parent is not None:
            idx = len(self.up_values)
            if name in self.parent.local_names:
                local_var = self.parent.local_names[name]
                self.up_values[name] = UpValueInfo(local_var.slot, -1, idx)
                local_var.captured = True
                return idx

            up_value_idx = self.parent.index_of_up_value(name)
            if up_value_idx >= 0:
                self.up_values[name] = UpValueInfo(-1, up_value_idx, idx)
                return idx

        return -1

    def close_opened_up_values(self):
        a = self.get_jmp_arg_a()
        if a > 0:
            self.emit_jmp(a, 0)

    # Jmp

    def get_jmp_arg_a(self) -> int:
        has_captured_loc_var = False
        min_slot_of_loc_var = self.max_regs
        for loc_var in self.local_names.values():
            while loc_var and loc_var.scope_level == self.scope_level:
                if loc_var.captured:
                    has_captured_loc_var = True
                if loc_var.slot < min_slot_of_loc_var and not loc_var.name.startswith("("):
                    min_slot_of_loc_var = loc_var.slot
                loc_var = loc_var.prev
        if has_captured_loc_var:
            return min_slot_of_loc_var + 1
        return 0

    def add_break_jmp(self):
        for i in range(self.scope_level, -1, -1):
            if self.breaks[i] is not None:
                pc = self.emit_jmp(0, 0)
                self.breaks[i].append(pc)
                return
        raise Exception("<break> not inside a loop!")

    def add_continue_jmp(self):
        for i in range(self.scope_level, -1, -1):
            if self.continues[i] is not None:
                pc = self.emit_jmp(0, 0)
                self.continues[i].append(pc)
                return
        raise Exception("<continue> not inside a loop!")

    def fill_c_or_b_jmp(self, record: list):
        pending_jmps = record.pop()
        a = self.get_jmp_arg_a()
        for pc in pending_jmps:
            sbx = self.pc - pc
            i = ((sbx + MAXARG_sBx) << 14) | (a << 6) | OpCode.JMP
            self.insts[pc] = i  # replace

    # Emit

    def emit_abc(self, opcode, a, b, c):
        # print("%5s %8d %8d %8d %8d" % ('ABC', opcode, a, b, c))
        i = ((b << 23) | (c << 14) | (a << 6) | opcode) & 0xffffffff
        self.insts.append(i)

    def emit_a_bx(self, opcode, a, bx):
        # print("%5s %8d %8d %8d" % ('ABx', opcode, a, bx))
        i = ((bx << 14) | (a << 6) | opcode) & 0xffffffff
        self.insts.append(i)

    def emit_as_bx(self, opcode, a, sbx):
        # print("%5s %8d %8d %8d" % ('AsBx', opcode, a, sbx))
        i = (((sbx + MAXARG_sBx) << 14) | (a << 6) | opcode) & 0xffffffff
        self.insts.append(i)

    def emit_ax(self, opcode, ax):
        # print("%5s %8d %8d" % ('AX', opcode, ax))
        i = ((ax << 6) | opcode) & 0xffffffff
        self.insts.append(i)

    def emit_move(self, a, b):
        self.emit_abc(OpCode.MOVE, a, b, 0)

    def emit_load_nil(self, a, n):
        self.emit_abc(OpCode.LOADNIL, a, n - 1, 0)

    def emit_load_bool(self, a, b, c):
        self.emit_abc(OpCode.LOADBOOL, a, b, c)

    def emit_load_k(self, a, k):
        idx = self.index_of_constant(k)
        if idx < (1 << 18):
            self.emit_a_bx(OpCode.LOADK, a, idx)
        else:
            self.emit_a_bx(OpCode.LOADKX, a, 0)
            self.emit_ax(OpCode.EXTRAARG, idx)

    def emit_vararg(self, a, n):
        self.emit_abc(OpCode.VARARG, a, n + 1, 0)

    def emit_closure(self, a, bx):
        self.emit_a_bx(OpCode.CLOSURE, a, bx)

    def emit_new_table(self, a, narr, nrec):
        self.emit_abc(OpCode.NEWTABLE, a, LuaValue.int2fb(narr), LuaValue.int2fb(nrec))

    def emit_set_list(self, a, b, c):
        self.emit_abc(OpCode.SETLIST, a, b, c)

    def emit_get_table(self, a, b, c):
        self.emit_abc(OpCode.GETTABLE, a, b, c)

    def emit_set_table(self, a, b, c):
        self.emit_abc(OpCode.SETTABLE, a, b, c)

    def emit_get_upval(self, a, b):
        self.emit_abc(OpCode.GETUPVAL, a, b, 0)

    def emit_set_upval(self, a, b):
        self.emit_abc(OpCode.SETUPVAL, a, b, 0)

    def emit_get_tabup(self, a, b, c):
        self.emit_abc(OpCode.GETTABUP, a, b, c)

    def emit_set_tabup(self, a, b, c):
        self.emit_abc(OpCode.SETTABUP, a, b, c)

    def emit_call(self, a, nargs, nret):
        self.emit_abc(OpCode.CALL, a, nargs + 1, nret + 1)

    def emit_tail_call(self, a, nargs):
        self.emit_abc(OpCode.TAILCALL, a, nargs + 1, 0)

    def emit_return(self, a, n):
        self.emit_abc(OpCode.RETURN, a, n + 1, 0)

    def emit_self(self, a, b, c):
        self.emit_abc(OpCode.SELF, a, b, c)

    def emit_jmp(self, a, sbx):
        self.emit_as_bx(OpCode.JMP, a, sbx)
        return len(self.insts) - 1

    def emit_test(self, a, c):
        self.emit_abc(OpCode.TEST, a, 0, c)

    def emit_test_set(self, a, b, c):
        self.emit_abc(OpCode.TESTSET, a, b, c)

    def emit_for_prep(self, a, sbx):
        self.emit_as_bx(OpCode.FORPREP, a, sbx)
        return len(self.insts) - 1

    def emit_for_loop(self, a, sbx):
        self.emit_as_bx(OpCode.FORLOOP, a, sbx)
        return len(self.insts) - 1

    def emit_tfor_call(self, a, c):
        self.emit_abc(OpCode.TFORCALL, a, 0, c)

    def emit_tfor_loop(self, a, sbx):
        self.emit_as_bx(OpCode.TFORLOOP, a, sbx)

    def emit_len(self, a, b):
        self.emit_abc(OpCode.LEN, a, b, 0)

    def emit_unary_op(self, op, a, b):
        if isinstance(op, ast.Not):
            self.emit_abc(OpCode.NOT, a, b, 0)
        elif isinstance(op, ast.Invert):
            self.emit_abc(OpCode.BNOT, a, b, 0)
        elif isinstance(op, ast.USub):
            self.emit_abc(OpCode.UNM, a, b, 0)
        #    self.emit_abc(OpCode.LEN, a, b, 0)

    def emit_binary_op(self, op, a, b, c):
        if op in FuncState.arith_and_bitwise_binops:
            self.emit_abc(FuncState.arith_and_bitwise_binops[op], a, b, c)
        else:
            if op == ast.Eq:
                self.emit_abc(OpCode.EQ, 1, b, c)
            elif op == ast.NotEq:
                self.emit_abc(OpCode.EQ, 0, b, c)
            elif op == ast.Lt:
                self.emit_abc(OpCode.LT, 1, b, c)
            elif op == ast.Gt:
                self.emit_abc(OpCode.LT, 1, c, b)
            elif op == ast.LtE:
                self.emit_abc(OpCode.LE, 1, b, c)
            elif op == ast.GtE:
                self.emit_abc(OpCode.LE, 1, c, b)

            self.emit_jmp(0, 1)
            self.emit_load_bool(a, 0, 1)
            self.emit_load_bool(a, 1, 0)

    # Output

    def get_up_values(self) -> List[Optional[UpValue]]:
        up_values: List[Optional[UpValue]] = [None for _ in range(len(self.up_values))]
        for _, up_value_info in self.up_values.items():
            if up_value_info.loc_var_slot >= 0:
                up_value = UpValue(True, up_value_info.loc_var_slot)
            else:
                up_value = UpValue(False, up_value_info.up_value_index)
            up_values[up_value_info.index] = up_value
        return up_values

    def get_up_value_names(self) -> List[str]:
        names = ['' for _ in range(len(self.up_values))]
        for name, up_value_info in self.up_values.items():
            names[up_value_info.index] = name
        return names

    def get_constants(self) -> Dict[Any, int]:
        consts = {}
        for k, idx in self.constants.items():
            consts[idx] = k
        return consts

    # Expression

    @contextmanager
    def concat(self, length: int, a: int):

        yield

        c = self.used_regs - 1
        b = c - length + 1
        self.emit_abc(OpCode.CONCAT, a, b, c)

    # Statement

    @contextmanager
    def do_stat(self):
        self.enter_scope(False)

        yield

        self.close_opened_up_values()
        self.leave_scope(self.pc - 1)

    @contextmanager
    def for_num(self, var_name: str):
        self.add_local_var("(for index)")
        self.add_local_var("(for limit)")
        self.add_local_var("(for step)")
        self.add_local_var(var_name)
        a = self.used_regs - 4
        pc_for_prep = self.emit_for_prep(a, 0)
        yield
        self.fill_c_or_b_jmp(self.continues)
        self.close_opened_up_values()
        pc_for_loop = self.emit_for_loop(a, 0)

        self.fill_sbx(pc_for_prep, pc_for_loop - pc_for_prep - 1)
        self.fill_sbx(pc_for_loop, pc_for_prep - pc_for_loop)

        self.leave_scope()

    @contextmanager
    def for_in(self, *var_names):
        self.add_local_var("(for generator)")
        self.add_local_var("(for state)")
        self.add_local_var("(for control)")
        for name in var_names:
            self.add_local_var(name)
        pc_jmp_to_tfc = self.emit_jmp(0, 0)

        yield

        self.fill_c_or_b_jmp(self.continues)
        self.close_opened_up_values()
        self.fill_sbx(pc_jmp_to_tfc, self.pc - pc_jmp_to_tfc)

        r = self.slot_of_local_var("(for generator)")
        self.emit_tfor_call(r, len(var_names))
        self.emit_tfor_loop(r + 2, pc_jmp_to_tfc - self.pc - 1)

        self.leave_scope()

    def while_do(self, process_test: Callable[[int], None], process_body: Callable, process_orelse: Callable = None):
        pc_before_exp = self.pc

        with self.assign_reg() as r:
            process_test(r)

        self.emit_test(r, 0)
        pc_jmp_to_end = self.emit_jmp(0, 0)
        self.enter_scope(True)

        process_body()

        self.fill_c_or_b_jmp(self.continues)
        self.close_opened_up_values()
        self.emit_jmp(0, pc_before_exp - self.pc - 1)

        if process_orelse:
            self.leave_scope(False)
            process_orelse()
            self.fill_c_or_b_jmp(self.breaks)
        else:
            self.leave_scope()
        self.fill_sbx(pc_jmp_to_end, self.pc - pc_jmp_to_end)

    def do_if(self, process_test, process_body):
        process_test_list = process_test if isinstance(process_test, Iterable) else [process_test]
        process_body_list = process_body if isinstance(process_body, Iterable) else [process_test]

        pc_jmp_to_ends = []
        pc_jmp_to_next_exp = -1

        for i, process_test in enumerate(process_test_list):
            if pc_jmp_to_next_exp >= 0:
                self.fill_sbx(pc_jmp_to_next_exp, self.pc - pc_jmp_to_next_exp)

            with self.assign_reg() as r:
                process_test(r)
                self.emit_test(r, 0)
                pc_jmp_to_next_exp = self.emit_jmp(0, 0)

            self.enter_scope(False)
            process_body_list[i]()
            self.close_opened_up_values()
            self.leave_scope()

            if i < len(process_test_list) - 1:
                pc_jmp_to_ends.append(self.emit_jmp(0, 0))
            else:
                pc_jmp_to_ends.append(pc_jmp_to_next_exp)

        for pc in pc_jmp_to_ends:
            self.fill_sbx(pc, self.pc - pc)

    # Helper

    def get_base_api(self, name: str, r: int):
        paths = name.split(".")
        self.emit_get_tabup(r, self.index_of_up_value(_ENV), self.index_of_constant(paths.pop(0)))
        for item in paths:
            self.emit_get_table(r, r, self.index_of_constant(item))

    @contextmanager
    def set_meta_table(self, name: str, r: int):
        setmetatable = self.get_base_api("setmetatable", r)

        yield

        self.get_base_api("python.class.mt", self.incr_regs())
        self.emit_call(setmetatable, 3, 1)
