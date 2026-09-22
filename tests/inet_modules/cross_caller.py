"""Cross-module caller — see cross_callee."""

from natsune.inet import inet

from tests.inet_modules.cross_callee import double


@inet
def quadruple(x: int) -> int:
    return double(double(x))
