import ast
from functools import wraps
from typing import Optional
from op_code import _ENV
from compiler.func_state import FuncState


class Context:
    def __init__(self, fs: FuncState, r: Optional[int] = None, n: int = 0):
        self.fs = fs
        self.r = r
        self.n = n


class ClassBody(Context):
    def __init__(self, fs: FuncState, cls_name: str, r: Optional[int] = None, n: int = 0):
        self.cls_name = cls_name
        super().__init__(fs, r, n)


def load_ctx(fn):
    @wraps(fn)
    def wrapper(self, node):
        self.fs: FuncState = node._ctx.fs
        self.r = node._ctx.r
        self.n = node._ctx.n
        fn(self, node)

    return wrapper


class NodeVisitor(ast.NodeVisitor):
    node_to_magic_method = {
        # Unary operators
        ast.UAdd: "__pos__",
        ast.USub: "__neg__",
        ast.Invert: "__invert__",
        # Normal arithmetic operators
        ast.Add: "__add__",
        ast.Sub: "__sub__",
        ast.Mult: "__mul__",
        ast.Div: "__div__",
        ast.FloorDiv: "__floordiv__",
        ast.Mod: "__mod__",
        ast.Pow: "__pow__",
        ast.LShift: "__lshift__",
        ast.RShift: "__rshift__",
        ast.BitOr: "__or__",
        ast.BitXor: "__xor__",
        ast.BitAnd: "__and__",
    }

    def generic_visit(self, node):
        raise NotImplementedError(node)

    # Top level node

    def visit_Module(self, node):
        node._ctx = Context(FuncState(), 0)

        ctx = node._ctx
        fs = FuncState(ctx.fs)
        fs.add_local_var("_ENV")
        fs.sub_funcs.append(fs)

        for child in node.body:
            child._ctx = Context(fs)
            self.visit(child)

        fs.leave_scope()
        fs.emit_return(0, 0)
        bx = len(fs.sub_funcs) - 1
        fs.emit_closure(node._ctx.cur_reg, bx)

    # Statement

    @load_ctx
    def visit_Assign(self, node):
        pass

    @load_ctx
    def visit_AnnAssign(self, node):
        pass

    @load_ctx
    def visit_AugAssign(self, node):
        pass

    @load_ctx
    def visit_Delete(self, node):
        for target in node.targets:
            pass  # TODO

    @load_ctx
    def visit_Assert(self, node):
        assert_func = self.fs.incr_regs()
        self.fs.emit_get_tabup(assert_func, self.fs.index_of_up_value("_ENV"), self.fs.index_of_constant("assert"))

        with self.fs.incr_regs() as ret, self.fs.incr_regs() as msg:
            node.test._ctx = Context(self.fs, ret)
            self.visit_Expr(node.test)
            if node.msg:
                self.fs.emit_load_k(msg, self.fs.index_of_constant(node.msg))
            else:
                self.fs.emit_load_nil(msg, 1)

        self.fs.emit_call(assert_func, 3, 1)

    @load_ctx
    def visit_Pass(self, node):
        self.fs.emit_jmp(0, 0)

    @load_ctx
    def visit_Raise(self, node):
        if node.exec:
            with self.fs.assign_reg() as ret:
                node.iter.ret = Context(self.fs, ret)
                self.visit_Expr(node.exec)

        if node.cause:
            raise NotImplementedError()

        error_func = self.fs.incr_regs()
        self.fs.emit_get_tabup(error_func, self.fs.index_of_up_value(_ENV), self.fs.index_of_constant("error"))
        self.fs.emit_call(error_func, 2, 1)

    # Literals

    @load_ctx
    def visit_Constant(self, node):
        """
        This class is available in the ast module from Python 3.6,
        but it isn’t produced by parsing code until Python 3.8.
        """
        pass

    @load_ctx
    def visit_Num(self, node):
        self.fs.get_base_api(f"python.type.{type(node.n)}", self.r)
        self.fs.emit_load_k(self.fs.incr_regs(), self.fs.index_of_constant(node.n))
        self.fs.emit_call(self.r, 2, 2)

    @load_ctx
    def visit_Str(self, node):
        self.fs.get_base_api("python.type.str", self.r)
        self.fs.emit_load_k(self.fs.incr_regs(), self.fs.index_of_constant(node.n))
        self.fs.emit_call(self.r, 2, 2)

    # Variables

    @load_ctx
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.fs.emit_move(self.r, self.fs.slot_of_local_var(node.id))
        elif isinstance(node.ctx, ast.Store):
            return self.fs.slot_of_local_var(node.id)
        elif isinstance(node.ctx, ast.Del):
            self.fs.emit_load_nil(self.fs.slot_of_local_var(node.id), 1)

    @load_ctx
    def visit_Starred(self, node):
        if isinstance(node.value, ast.Name):
            if isinstance(node.ctx, ast.Store):
                return self.fs.slot_of_local_var(node.value.id)
            if isinstance(node.ctx, ast.Load):
                with self.fs.assign_reg() as unpack:
                    self.fs.emit_get_tabup(unpack, self.fs.index_of_up_value(_ENV), self.fs.index_of_constant("unpack"))
                    self.fs.emit_call(unpack, 2, 0)
        else:
            raise NotImplementedError(node.value)

    @load_ctx
    def visit_JoinedStr(self, node):
        self.fs.get_base_api("python.type.str", self.r)

        joined_str = self.fs.incr_regs()
        for value in node.values:
            if isinstance(value, ast.Str):
                self.emit_load_k(self.fs.incr_regs(), self.fs.index_of_constant(value.s))
            elif isinstance(value, ast.FormattedValue):
                with self.fs.assign_reg() as python, self.fs.assign_reg() as py_format:
                    self.fs.emit_get_tabup(python, self.fs.index_of_up_value(_ENV), self.fs.index_of_constant("python"))
                    self.fs.emit_get_table(py_format, python, self.fs.index_of_constant("format"))
                    with self.fs.assign_reg() as ret:
                        value._ctx = Context(self.fs, ret)
                        self.visit(value)
                    if value.conversion != -1:
                        raise NotImplementedError(value.conversion)
                    if value.format_spec:
                        value.format_spec._ctx = Context(self.fs)
                        self.visit(value.format_spec)
                self.emit_call(py_format, 0, 2)
            else:
                raise NotImplementedError(value)
        length = len(node.values)
        self.fs.decr_regs(length)
        self.fs.concat(length, joined_str)

        self.fs.emit_call(self.r, 2, 2)

    # Expressions

    @load_ctx
    def visit_Expr(self, node):
        value = node.value
        node._ctx.r = self.fs.incr_regs()
        value._ctx = node._ctx
        self.visit(value)

    @load_ctx
    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            bl = self.fs.incr_regs()
            self.get_base_api("python.builtin.bool", bl)

            node.operand._ctx = Context(self.fs, self.fs.incr_regs())
            self.visit(node.operand)

            self.fs.emit_call(bl, 2, 2)
            self.fs.emit_unary_op(node.op, self.r, self.r)
        else:
            node.operand._ctx = Context(self.fs, self.r)
            self.visit(node.operand)
            self.fs.emit_self(self.r, self.r, self.fs.index_of_constant(self.node_to_magic_method[node.op]))
            self.fs.emit_call(self.r, 2, 2)

    @load_ctx
    def visit_BinOp(self, node):
        node.left._ctx = Context(self.fs, self.r)
        self.visit(node.left)
        node.right._ctx = Context(self.fs, self.fs.incr_regs())
        self.visit(node.right)
        self.fs.emit_self(self.r, self.r, self.fs.index_of_constant(self.node_to_magic_method[node.op]))
        self.fs.emit_call(self.r, 3, 2)

    @load_ctx
    def visit_BoolOp(self, node):
        pass

    @load_ctx
    def visit_Compare(self, node):
        pass

    @load_ctx
    def visit_Call(self, node):
        self.visit(node.func)

        nargs = len(node.args)
        for arg in node.args:
            tmp_reg = self.fs.incr_regs()
            arg._ctx = Context(self.fs, tmp_reg)
            self.visit(arg)

        if node.keywords:
            nargs += 1
            with self.fs.assign_reg() as kwargs:
                self.fs.emit_new_table(kwargs, 0, 0)
                for keyword in node.keywords:
                    with self.fs.assign_reg() as ret:
                        value = keyword.value
                        value._ctx = Context(self.fs, ret)
                        self.visit_Expr(value)
                        if keyword.arg:
                            self.fs.emit_set_table(kwargs, self.fs.index_of_constant(keyword.arg), ret)
                        else:
                            with self.fs.scope(True):
                                with self.fs.assign_reg() as pairs, self.fs.assign_reg() as t:
                                    self.fs.emit_get_tabup(pairs, self.fs.index_of_up_value("_ENV"), self.fs.index_of_constant("pairs"))
                                    self.fs.emit_move(t, kwargs)
                                    self.fs.emit_call(pairs, 2, 0)
                                with self.fs.for_in("k", "v"):
                                    self.fs.emit_set_table(kwargs, self.fs.slot_of_local_var("k"), self.fs.slot_of_local_var("v"))

        self.fs.emit_call(self.r, nargs + 1, self.n)

    @load_ctx
    def visit_IfExp(self, node):
        pass

    @load_ctx
    def visit_Attribute(self, node):
        with self.fi.assign_reg() as value:
            node.value._ctx = Context(self.fs, value)
            self.visit(node.value)
            if isinstance(node.ctx, ast.Load):
                self.fi.emit_get_table(self.cur_reg, value, self.fi.index_of_constant(node.attr))
            elif isinstance(node.ctx, ast.Store):
                self.fi.emit_set_table(self.cur_reg, self.fi.index_of_constant(node.attr), value)
            else:
                raise NotImplementedError(node.value)

    @load_ctx
    def visit_Subscript(self, node):
        with self.fs.assign_reg() as value:
            node.value._ctx = Context(self.fs, value)
            self.visit(node.value)
            self.fs.emit_self(value, value, self.fs.index_of_constant("__getitem__"))
            node.slice._ctx = Context(self.fs, self.fs.incr_regs())
            self.visit(node.slice)
            self.emit_call(value, 3, self.n)

    @load_ctx
    def visit_Slice(self, node):
        self.fs.get_base_api("python.type.slice", self.r)
        for attr_name in ("lower", "upper", "step"):
            attr = getattr(node, attr_name)
            if attr:
                attr._ctx = Context(self.fs, self.fs.incr_regs())
                self.visit(attr)
            else:
                none_type = self.fs.incr_regs()
                self.fs.get_base_api("python.type.NoneType", none_type)
                self.fs.emit_call(none_type, 1, 2)
        self.fs.emit_call(self.r, 4, 2)

    @load_ctx
    def visit_ExtSlice(self, node):
        self.fs.get_base_api("python.type.tuple")
        for dim in node.dims:
            dim._ctx = Context(self.fs, self.fs.incr_regs())
            self.visit(dim)
        self.fs.emit_call(self.r, len(node.dims), 2)

    # Control flow

    @load_ctx
    def visit_While(self, node):
        def process_test(r):
            node.test._ctx = Context(self.fs, r)
            self.visit_Expr(node.test)

        def process_body():
            for child in node.body:
                child._ctx = Context(self.fs)
                self.visit(child)

        def process_orelse():
            for child in node.orelse:
                child._ctx = Context(self.fs)
                self.visit(child)

        self.fs.while_do(process_test, process_body, process_orelse)

    @load_ctx
    def visit_If(self, node):
        def process_test(r):
            node.test._ctx = Context(self.fs, r)
            self.visit_Expr(node.test)

        def process_body():
            for child in node.body:
                child._ctx = Context(self.fs)
                self.visit(child)

        if node.orelse:
            def process_orelse():
                for child in node.body:
                    child._ctx = Context(self.fs)
                    self.visit(child)

            self.fs.do_if(process_test, (process_body, process_orelse))
        else:
            self.fs.do_if(process_test, process_body)

    @load_ctx
    def visit_For(self, node):
        var_names = []
        if isinstance(node.target, ast.Tuple):
            for name in node.target.elts:
                if not isinstance(name, ast.Name):
                    raise NotImplementedError()
                var_names.append(name.id)
        elif isinstance(node.target, ast.Name):
            var_names.append(node.target.id)
        else:
            raise NotImplementedError()

        with self.fs.scope(True):
            with self.fs.assign_reg() as iterable:
                # if isinstance(node.iter, ast.Name):
                #     fs.emit_move(iterable, fs.slot_of_local_var(node.iter.id))
                node.iter._ctx = Context(self.fs, iterable)
                self.visit_Expr(node.iter)
                self.fs.emit_self(iterable, iterable, self.fs.index_of_constant("__iter"))
                self.fs.emit_call(iterable, 2, 0)
            with self.fs.for_in(var_names):
                for child in node.body:
                    child._ctx = Context(self.fs)
                    self.visit(child)

    @load_ctx
    def visit_Break(self, node):
        self.fs.add_break_jmp()

    @load_ctx
    def visit_Continue(self, node):
        self.fs.add_continue_jmp()

    # Function and class definitions

    @load_ctx
    def visit_FunctionDef(self, node):
        ctx: Context = node._ctx
        if ctx.fs.parent.parent:
            ctx.cur_reg = ctx.fs.add_local_var(node.name)
        fs = FuncState(ctx.fs)
        ctx.fs.sub_funcs.append(fs)

        arguments = node.args
        args = arguments.args
        if args:
            args_len, defaults_len = len(arguments.args), len(arguments.defaults) if arguments.defaults else 0
            for i, arg in enumerate(arguments.args):
                slot = fs.add_local_var(arg.arg)
                if i > args_len - defaults_len:
                    fs.emit_test(slot, 1)
                    fs.emit_jmp(slot, 1)
                    tmp_ci = fs.index_of_constant(arguments.defaults[i - (args_len + defaults_len)].n, True)
                    fs.emit_load_k(slot, tmp_ci)

        vararg = arguments.vararg
        if vararg:
            fs.is_vararg = True
            tail = 0
            kwonlyarg_slots = []
            kwarg_slot = None
            if arguments.kwonlyargs:
                tail += len(arguments.kwonlyargs)
                for arg in arguments.kwonlyargs:
                    kwonlyarg_slots.append(fs.add_local_var(arg.arg))
            if arguments.kwarg:
                tail += 1
                kwarg_slot = fs.add_local_var(arguments.kwarg.arg)
            fs.add_local_var(vararg.arg)

            with fs.assign_reg() as vararg_list, fs.assign_reg() as separate:
                fs.emit_new_table(vararg_list, 0, 0)
                fs.emit_vararg(vararg_list, 0)
                fs.emit_set_list(vararg_list, 0, 1)
                fs.emit_len(separate, vararg_list)

                with fs.scope(True):
                    fs.emit_load_k(fs.incr_regs(), fs.index_of_constant(1))
                    for_limit = fs.incr_regs()
                    fs.emit_move(for_limit, separate)
                    fs.emit_binary_op(ast.Sub, for_limit, for_limit, fs.index_of_constant(tail))
                    fs.emit_load_k(fs.incr_regs(), fs.index_of_constant(1))
                    fs.decr_regs(3)
                    with fs.for_num("i"):
                        v = fs.incr_regs()
                        fs.emit_get_table(v, vararg_list, fs.slot_of_local_var("i"))
                        fs.emit_set_table(fs.slot_of_local_var(vararg.arg), fs.slot_of_local_var("i"), v)

                if arguments.kwonlyargs:
                    kwonlyargs_len, kw_defaults_len = len(arguments.kwonlyargs), len(arguments.kw_defaults) if arguments.kw_defaults else 0
                    for i, slot in enumerate(kwonlyarg_slots):
                        fs.emit_binary_op(ast.Add, separate, separate, fs.index_of_constant(1, True))
                        if i > kwonlyargs_len - kw_defaults_len:
                            tmp_reg = fs.incr_regs()
                            fs.emit_get_table(tmp_reg, vararg_list, separate)
                            fs.emit_test_set(slot, tmp_reg, 1)
                            fs.emit_jmp(slot, 1)
                            tmp_ci = fs.index_of_constant(arguments.kw_defaults[i - (kwonlyargs_len + kw_defaults_len)].n, True)
                            fs.emit_load_k(slot, tmp_ci)
                            break
                        fs.emit_get_table(slot, vararg_list, separate)

                if arguments.kwarg:
                    fs.emit_binary_op(ast.Add, separate, separate, fs.index_of_constant(1, True))
                    fs.emit_get_table(kwarg_slot, vararg_list, separate)
        else:
            fs.add_local_var(arguments.kwarg.arg)

        for child in node.body:
            child._ctx = Context(fs)
            self.visit(child)

        fs.leave_scope()
        fs.emit_return(0, 0)
        bx = len(ctx.fs.sub_funcs) - 1
        fs.emit_closure(ctx.cur_reg, bx)

    @load_ctx
    def visit_Return(self, node):
        pass

    @load_ctx
    def visit_Lambda(self, node):
        pass

    @load_ctx
    def visit_ClassDef(self, node):
        setmetatable = self.fs.get_base_api("setmetatable")
        klass = self.fs.add_local_var(node.name)

        cur_reg = self.fs.used_regs
        bases = self.fs.incr_regs()
        self.emit_get_table(bases, klass, self.fs.index_of_constant("bases"))
        for base in node.bases:
            base._ctx = Context(self.fs, self.fs.incr_regs())
            self.visit(base)
        self.emit_set_list(bases, len(node.bases), 1)
        self.fs.used_regs = cur_reg

        for attr_name in ("keywords", "decorator_list"):
            with self.fs.assign_reg() as attr_reg:
                attr = getattr(node, attr_name)
                self.emit_new_table(attr_reg, 0, len(attr))
                for item in attr:
                    with self.fs.assign_reg() as ret:
                        item.value._ctx = Context(self.fs, ret)
                        self.visit(item.value)
                        self.emit_set_table(attr_reg, self.fs.index_of_constant(item.arg), ret)
                self.emit_set_table(klass, self.fs.index_of_constant(f"({attr_name})"), attr_reg)

        with self.fs.do_stat():
            for child in node.body:
                child._ctx = ClassBody(self.fs, node.name)
                self.visit(child)

        self.fs.get_base_api("python.class.mt")
        self.fs.emit_call(setmetatable, 3, 1)

    @load_ctx
    def visit_Yield(self, node):
        with self.fs.assign_reg() as coroutine, self.fs.assign_reg() as lyield:
            self.fs.emit_get_tabup(coroutine, self.fs.index_of_up_value(_ENV), self.fs.index_of_constant("coroutine"))
            self.fs.emit_get_table(lyield, coroutine, self.fs.index_of_constant("yield"))
            if node.value:
                with self.fs.assign_reg() as ret:
                    node.value._ctx = Context(self.fs, ret)
                    self.visit(node)
            self.emit_call(lyield, 2, 1)

    # Async and await
