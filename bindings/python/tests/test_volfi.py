import math
import numpy as np
import volfi


def test_reference_values():
    h = np.array([0, 0.01, 0.1, 0.5, 1.0])
    c = np.array([0.1, 0.01, 0.001, 0.05, 0.2])
    t = np.array([1, 0.5, 2, 1.25, 0.75])
    w = np.array([0.063163096373724928, 0.0013032051608100883, 0.0033012778463021937, 0.24189968319602106, 1.4308324613409955])
    iv = np.array([0.25132269371014815, 0.051053014814212261, 0.040628055862311413, 0.43990879345248018, 1.381222869460728])
    ctx = volfi.ctx(h)
    assert np.allclose(ctx.w(c), w, rtol=0, atol=1e-14)
    assert np.allclose(ctx.iv(c, t), iv, rtol=0, atol=1e-14)
    assert np.allclose(volfi.w_otm(h, c), w, rtol=0, atol=1e-14)
    assert np.allclose(volfi.iv_otm(h, c, t), iv, rtol=0, atol=1e-14)


def test_call_helpers():
    assert abs(volfi.w_otm([1.0], [0.01])[0] - 0.28357398093281544) < 1e-14
    assert abs(volfi.iv_otm([1.0], [0.01], [2.0])[0] - 0.37654613325116981) < 1e-14
    assert abs(volfi.w_call_norm([0.1], [0.01])[0] - 0.010983412121237294) < 1e-14
    assert abs(volfi.w_call_norm([-0.1], [0.1])[0] - 0.0072973603266672377) < 1e-14
    assert abs(volfi.iv_call([100], [105], [0.98], [1.2], [7])[0] - 0.21108908075172517) < 1e-14
    assert volfi.version() == "0.1.8"
