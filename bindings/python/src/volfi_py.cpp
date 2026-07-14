// Python binding for the volfi v0.2.0 routed implied-volatility inverter.
//
// Array inputs are inverted through the vectorized grid-batch driver, so a NumPy call gets
// the SIMD path; both scalar and array paths return bit-identical results (the module must
// be built with -ffp-contract=off, which setup.py does). Conventions match the C++ library:
// h = |log(K/F)| >= 0, c = undiscounted OTM-call price / F in (0,1), and the returned
// w = v^2 = (sigma*sqrt(T))^2 is the total implied variance.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <volfi/volfi_annulus_all.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace va = volfi_annulus;

// A broadcast-aware float64 view: length 1 broadcasts against length n.
struct dvec {
  py::array_t<double, py::array::c_style | py::array::forcecast> a;
  const double* p;
  py::ssize_t n;
  explicit dvec(py::object x)
      : a(py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(x)) {
    if (!a) throw std::runtime_error("input cannot be converted to a float64 array");
    auto b = a.request();
    p = static_cast<const double*>(b.ptr);
    n = b.size;
  }
  double operator()(py::ssize_t i) const { return p[n == 1 ? 0 : i]; }
};

static py::ssize_t bcast(std::initializer_list<py::ssize_t> lens) {
  py::ssize_t n = 1;
  for (py::ssize_t l : lens) {
    if (l < 1) throw std::runtime_error("inputs must be non-empty");
    n = std::max(n, l);
  }
  for (py::ssize_t l : lens)
    if (l != 1 && l != n) throw std::runtime_error("input lengths must be 1 or equal");
  return n;
}

static inline void need_h(double h) {
  if (!std::isfinite(h) || h <= 0.0)
    throw std::runtime_error("h must be finite and positive (h = |log(K/F)|)");
}
static inline void need_c(double c) {
  if (!std::isfinite(c) || c <= 0.0 || c >= 1.0)
    throw std::runtime_error("c must be finite and strictly inside (0, 1)");
}
static inline void need_pos(double x, const char* nm) {
  if (!std::isfinite(x) || x <= 0.0)
    throw std::runtime_error(std::string(nm) + " must be finite and positive");
}

// ---- total implied variance w, via the vectorized grid-batch driver ---------------------
static py::array_t<double> implied_variance(py::object h0, py::object c0) {
  dvec h(h0), c(c0);
  py::ssize_t n = bcast({h.n, c.n});
  std::vector<double> hb(n), cb(n);
  for (py::ssize_t i = 0; i < n; ++i) {
    need_h(h(i)); need_c(c(i));
    hb[i] = h(i); cb[i] = c(i);
  }
  py::array_t<double> out(n);
  va::implied_variance_grid_batch(hb.data(), cb.data(), out.mutable_data(), static_cast<int>(n));
  return out;
}

// ---- total volatility sigma = sqrt(w / T) ------------------------------------------------
static py::array_t<double> implied_volatility(py::object h0, py::object c0, py::object t0) {
  dvec h(h0), c(c0), t(t0);
  py::ssize_t n = bcast({h.n, c.n, t.n});
  std::vector<double> hb(n), cb(n);
  for (py::ssize_t i = 0; i < n; ++i) {
    need_h(h(i)); need_c(c(i)); need_pos(t(i), "t");
    hb[i] = h(i); cb[i] = c(i);
  }
  py::array_t<double> out(n);
  double* w = out.mutable_data();
  va::implied_variance_grid_batch(hb.data(), cb.data(), w, static_cast<int>(n));
  for (py::ssize_t i = 0; i < n; ++i) w[i] = std::sqrt(w[i] / t(i));
  return out;
}

// ---- checked variant: returns (w, status) without raising on bad input -------------------
static py::tuple implied_variance_checked(py::object h0, py::object c0) {
  dvec h(h0), c(c0);
  py::ssize_t n = bcast({h.n, c.n});
  py::array_t<double> w(n);
  py::array_t<int> st(n);
  double* wp = w.mutable_data();
  int* sp = st.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    va::iv_status s;
    wp[i] = va::implied_variance_otm_checked(h(i), c(i), &s);
    sp[i] = static_cast<int>(s);
  }
  return py::make_tuple(w, st);
}

// ---- from raw option data: put-call parity flip to the OTM twin handled internally -------
static py::array_t<double> implied_volatility_from_option(py::object F0, py::object K0,
                                                          py::object price0, py::object t0,
                                                          py::object is_call0) {
  dvec F(F0), K(K0), price(price0), t(t0);
  auto ic = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(is_call0);
  if (!ic) throw std::runtime_error("is_call cannot be interpreted as a boolean array");
  auto icb = ic.request();
  const double* icp = static_cast<const double*>(icb.ptr);
  py::ssize_t icn = icb.size;
  py::ssize_t n = bcast({F.n, K.n, price.n, t.n, icn});
  py::array_t<double> out(n);
  double* y = out.mutable_data();
  for (py::ssize_t i = 0; i < n; ++i) {
    need_pos(F(i), "F"); need_pos(K(i), "K"); need_pos(t(i), "t");
    bool call = icp[icn == 1 ? 0 : i] != 0.0;
    y[i] = va::implied_volatility(F(i), K(i), price(i), t(i), call, nullptr);
  }
  return out;
}

// ---- streaming warm restart: re-invert from previous variances ---------------------------
static py::array_t<double> implied_variance_warm(py::object h0, py::object c0, py::object wprev0,
                                                 int steps) {
  dvec h(h0), c(c0), wprev(wprev0);
  py::ssize_t n = bcast({h.n, c.n, wprev.n});
  std::vector<double> hb(n), cb(n), pb(n);
  for (py::ssize_t i = 0; i < n; ++i) {
    need_h(h(i)); need_c(c(i));
    hb[i] = h(i); cb[i] = c(i); pb[i] = wprev(i);
  }
  py::array_t<double> out(n);
  va::implied_variance_warm_batch(hb.data(), cb.data(), pb.data(), out.mutable_data(),
                                  static_cast<int>(n), steps);
  return out;
}

PYBIND11_MODULE(_volfi, m) {
  m.doc() = "volfi v0.2.0: routed, vectorizable Black-Scholes implied-volatility inverter";

  py::enum_<va::iv_status>(m, "iv_status", "Input-classification status of a checked inversion")
      .value("ok", va::iv_status::ok)
      .value("below_intrinsic", va::iv_status::below_intrinsic)
      .value("above_max", va::iv_status::above_max)
      .value("bad_input", va::iv_status::bad_input)
      .value("out_of_domain", va::iv_status::out_of_domain)
      .value("near_saturation", va::iv_status::near_saturation);

  m.def("implied_variance", &implied_variance, py::arg("h"), py::arg("c"),
        "Total implied variance w = v^2 from (h, c); vectorized over arrays.");
  m.def("implied_volatility", &implied_volatility, py::arg("h"), py::arg("c"), py::arg("t"),
        "Total volatility sigma = sqrt(w/T) from (h, c, T).");
  m.def("implied_variance_checked", &implied_variance_checked, py::arg("h"), py::arg("c"),
        "Return (w, status) arrays; never raises. status values match the iv_status enum.");
  m.def("implied_volatility_from_option", &implied_volatility_from_option, py::arg("F"),
        py::arg("K"), py::arg("price"), py::arg("t"), py::arg("is_call"),
        "sigma from raw (F, K, price, T, is_call); ITM options and puts are parity-projected.");
  m.def("implied_variance_warm", &implied_variance_warm, py::arg("h"), py::arg("c"),
        py::arg("w_prev"), py::arg("steps") = 2,
        "Streaming warm restart from previous variances (few-ULP contract; see README).");
  m.def("version", []() { return "0.2.0"; });
}
