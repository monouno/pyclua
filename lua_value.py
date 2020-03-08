class UpValue:
    def __init__(self, in_stack: bool, idx: int):
        self.in_stack = in_stack
        self.idx = idx


class LuaValue:
    @staticmethod
    def fb2int(val):
        if val < 8:
            return val
        return ((val & 7) + 9) << ((val >> 3) - 1)

    @staticmethod
    def int2fb(val):
        e = 0
        if val < 8:
            return val

        while val >= (8 << 4):
            val = (val + 0xf) >> 4
            e += 4

        while val >= (8 << 1):
            val = (val + 1) >> 1
            e += 1

        return ((e + 1) << 3) | (val - 8)
