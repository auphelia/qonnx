import numpy as np

from qonnx.custom_op.general.intquant import int_quant


def _f32_ulp(x, n):
    """Return the float32 that is n ULPs away from x (walks the raw bit pattern)."""
    xi = np.float32(x).view(np.int32)
    return (xi + np.int32(n)).view(np.float32)


def _find_fp32_division_tie(scale, level, span=1 << 15):
    """Construct a float32 value ``v`` whose quotient ``v / scale`` collapses to
    the exact half-way tie ``level + 0.5`` in float32, while in float64 it lands
    strictly on the ``level + 1`` side of that tie.

    For such a value the naive float32 path rounds (round-half-to-even) to the
    even neighbour ``level``, whereas the mathematically-correct level is
    ``level + 1``. Returns the candidate with the float64 quotient furthest onto
    the correct side, or ``None`` if none is found in the search window.
    """
    scale32 = np.float32(scale)
    tie = np.float32(level) + np.float32(0.5)
    base = np.float32(np.float64(tie) * np.float64(scale32))
    best = None
    best_q64 = -np.inf
    for k in range(-span, span):
        v = _f32_ulp(base, k)
        if np.float32(v) / scale32 == tie:
            q64 = np.float64(v) / np.float64(scale32)
            # tie is negative; the "correct" side (level + 1) is toward zero
            if q64 > np.float64(tie) and q64 > best_q64:
                best = v
                best_q64 = q64
    return best


def test_int_quant_rounds_in_fp64_off_fp32_tie():
    """Regression: int_quant must round using a float64 quotient so a value that
    only collapses to a rounding tie in float32 is still sent to the correct
    integer level (mirrors the SigLIP QONNX->FINN one-level bit-flip).

    The scale below is the per-tensor activation scale from the SigLIP layer-0
    K-projection that first exposed the flip; with it a float32 value exists
    whose ``v / scale`` collapses to exactly ``-9.5`` (rounding to ``-10``) while
    the float64 quotient is ``-9.4999997`` (correctly rounding to ``-9``).
    """
    scale = np.float32(0.13991454)
    level = -10  # integer level the buggy float32 tie would round to

    v = _find_fp32_division_tie(scale, level)
    assert v is not None, "could not construct a float32 division tie for this scale"

    # sanity: the constructed value really is a float32-only tie
    assert np.float32(v) / scale == np.float32(level) + np.float32(0.5)
    assert np.float64(v) / np.float64(scale) > np.float64(level) + 0.5

    inp = np.array([v], dtype=np.float32)
    scale_arr = np.array([scale], dtype=np.float32)
    zeropt = np.array([0.0], dtype=np.float32)
    bitwidth = np.array(8.0)

    out = int_quant(inp, scale_arr, zeropt, bitwidth, signed=True, narrow=False, rounding_mode="ROUND")

    # recover the integer level the quantizer actually selected
    got_level = int(np.round(np.float64(out[0]) / np.float64(scale)))
    assert got_level == level + 1
    # output dtype is preserved as float32
    assert out.dtype == np.float32
