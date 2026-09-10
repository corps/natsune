from types import SimpleNamespace

from natsune.adapters import ParValueAdapter, adapter_from_type
from natsune.frontend.ir import render_function
from natsune.special_forms import Par
from tests.frontend.helpers import build_ir_for


def _inet_function(arg_types=(int,), return_type=int):
    compiler = SimpleNamespace(
        args_adapter=ParValueAdapter([adapter_from_type(te) for te in arg_types]),
        return_annot=return_type,
    )

    def make(x):
        return 0, 0

    setattr(make, "__inet__", compiler)
    return make


def _render(snippet: str, **kwargs) -> str:
    ir, _ = build_ir_for(snippet, **kwargs)
    return render_function(ir)


def test_golden_basic():
    assert (
        _render("""
        def f(a: int, b: str) -> int:
            c = 10
            d = a
            return c
        """)
        == (
            "(function f\n"
            "  (param a va)\n"
            "  (param b va)\n"
            "  (returns va)\n"
            "  (body\n"
            "    (usage c :write d :write a :read)\n"
            "    (assign\n"
            "      (targets\n"
            "        (name c :va))\n"
            "      (value\n"
            "        (const 10 :va)))\n"
            "    (assign\n"
            "      (targets\n"
            "        (name d :va))\n"
            "      (value\n"
            "        (var a :va)))\n"
            "    (return !\n"
            "      (var c :va))))\n"
        )
    )


def test_golden_dynamic_with_captures():
    assert (
        _render(
            """
        def f(b: int) -> int:
            print(b * 2, make(3), b_global)
            return 0
        """,
            namespace={"make": _inet_function(), "b_global": 1},
        )
        == (
            "(function f\n"
            "  (param b va)\n"
            "  (returns va)\n"
            "  (body\n"
            "    (usage b :read)\n"
            "    (exprstmt\n"
            "      (dynamic :va\n"
            '        (source "print(b * 2, __natsune_0__, b_global)")\n'
            "        (captures\n"
            "          (b\n"
            "            (var b :va))\n"
            "          (__natsune_0__\n"
            "            (call-inet :arity 1 :sigs (va) :ret va\n"
            "              (const 3 :va))))))\n"
            "    (return !\n"
            "      (const 0 :va))))\n"
        )
    )


def test_golden_control_flow():
    assert (
        _render(
            """
        def f(c: int) -> int:
            if c:
                a = 1
            else:
                a = 2
            while c:
                break
            for i in xs:
                print(i)
            return a
        """,
            namespace={"xs": [1]},
        )
        == (
            "(function f\n"
            "  (param c va)\n"
            "  (returns va)\n"
            "  (body\n"
            "    (usage c :read a :write i :write)\n"
            "    (if *\n"
            "      (test\n"
            "        (var c :va))\n"
            "      (then\n"
            "        (usage a :write)\n"
            "        (assign\n"
            "          (targets\n"
            "            (name a :va))\n"
            "          (value\n"
            "            (const 1 :va))))\n"
            "      (else\n"
            "        (usage a :write)\n"
            "        (assign\n"
            "          (targets\n"
            "            (name a :va))\n"
            "          (value\n"
            "            (const 2 :va)))))\n"
            "    (while *\n"
            "      (test\n"
            "        (var c :va))\n"
            "      (body\n"
            "        (break !))\n"
            "      (orelse))\n"
            "    (for *\n"
            "      (target\n"
            "        (name i :va))\n"
            "      (iter\n"
            "        (var xs :global :va))\n"
            "      (body\n"
            "        (usage i :read)\n"
            "        (exprstmt\n"
            "          (dynamic :va\n"
            '            (source "print(i)")\n'
            "            (captures\n"
            "              (i\n"
            "                (var i :va))))))\n"
            "      (orelse))\n"
            "    (return !\n"
            "      (var a :va))))\n"
        )
    )


def test_golden_par_index_and_augassign():
    # p is Par[int, str]: p[1] alone is copyable, but reads go through a
    # readout of the whole Par with neighbors closed, so ANY p-read
    # linearizes the cell — both the p[0] += 3 and the p[1] read report
    # :write (read_independently is VALUE-leaf-only for now; element
    # pass-through is the marked post-cutover improvement, §6).
    assert (
        _render(
            """
        def f(p: Par[int, str]) -> int:
            p[1] += 3
            return p[0]
        """,
            namespace={"Par": Par},
        )
        == (
            "(function f\n"
            "  (param p par[va, va])\n"
            "  (returns va)\n"
            "  (body\n"
            "    (usage p :write)\n"
            "    (augassign +\n"
            "      (target\n"
            "        (dynamic\n"
            '          (source "__natsune_0__")\n'
            "          (captures\n"
            "            (__natsune_0__\n"
            "              (par-index 1 :va\n"
            "                (var p :par[va, va]))))))\n"
            "      (value\n"
            "        (const 3 :va)))\n"
            "    (return !\n"
            "      (par-index 0 :va\n"
            "        (var p :par[va, va])))))\n"
        )
    )


def test_render_is_deterministic():
    snippet = """
        def f(b: int) -> int:
            print(b, make(1))
            return b
        """
    ir1, _ = build_ir_for(snippet, namespace={"make": _inet_function()})
    ir2, _ = build_ir_for(snippet, namespace={"make": _inet_function()})

    assert render_function(ir1) == render_function(ir2)
    assert render_function(ir1) == render_function(ir1)


def test_include_positions_flag():
    ir, _ = build_ir_for("""
        def f() -> int:
            return 1
        """)

    plain = render_function(ir)
    positioned = render_function(ir, include_positions=True)

    assert "@" not in plain
    assert "@1:0" in positioned  # the function node
    assert "@2:4" in positioned  # the return statement
    assert positioned != plain
