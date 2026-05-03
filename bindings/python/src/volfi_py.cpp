#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#ifndef __GNUC__
#ifndef __builtin_expect
#define __builtin_expect(x, y) (x)
#endif
#endif
#include <volfi/volfi.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

struct dvec {
  py::array_t<double, py::array::c_style | py::array::forcecast> a;
  const double* p;
  py::ssize_t n;
  explicit dvec(py::object x): a(py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(x)) {
    if (!a) throw std::runtime_error("input cannot be converted to float64 array");
    auto b = a.request();
    p = static_cast<const double*>(b.ptr);
    n = b.size;
  }
  double operator()(py::ssize_t i) const { return p[n == 1 ? 0 : i]; }
};

static py::ssize_t nout(py::ssize_t a, py::ssize_t b) {
  if (a < 1 || b < 1) throw std::runtime_error("inputs must be non-empty");
  py::ssize_t n = a > b ? a : b;
  if ((a != 1 && a != n) || (b != 1 && b != n)) throw std::runtime_error("lengths must be 1 or equal");
  return n;
}

static py::ssize_t nout3(py::ssize_t a, py::ssize_t b, py::ssize_t c) {
  if (a < 1 || b < 1 || c < 1) throw std::runtime_error("inputs must be non-empty");
  py::ssize_t n = std::max(a, std::max(b, c));
  if ((a != 1 && a != n) || (b != 1 && b != n) || (c != 1 && c != n)) throw std::runtime_error("lengths must be 1 or equal");
  return n;
}

static py::ssize_t nout5(py::ssize_t a, py::ssize_t b, py::ssize_t c, py::ssize_t d, py::ssize_t e) {
  if (a < 1 || b < 1 || c < 1 || d < 1 || e < 1) throw std::runtime_error("inputs must be non-empty");
  py::ssize_t n = std::max(std::max(a, b), std::max(std::max(c, d), e));
  if ((a != 1 && a != n) || (b != 1 && b != n) || (c != 1 && c != n) || (d != 1 && d != n) || (e != 1 && e != n)) throw std::runtime_error("lengths must be 1 or equal");
  return n;
}

static void check_h(double x) {
  if (!std::isfinite(x) || x < 0) throw std::runtime_error("h must be finite and non-negative");
}

static void check_c(double x) {
  if (!std::isfinite(x) || x <= 0 || x >= 1) throw std::runtime_error("c must be finite and inside (0,1)");
}

static void check_t(double x) {
  if (!std::isfinite(x) || x <= 0) throw std::runtime_error("t must be finite and positive");
}

static void check_pos(double x, const char* nm) {
  if (!std::isfinite(x) || x <= 0) throw std::runtime_error(std::string(nm) + " must be finite and positive");
}

struct ctx {
  std::vector<volfi::otm_context> q;
  explicit ctx(py::object h0) {
    dvec h(h0);
    if (h.n < 1) throw std::runtime_error("h must be non-empty");
    q.reserve(h.n);
    for (py::ssize_t i = 0; i < h.n; ++i) {
      double x = h(i);
      check_h(x);
      q.emplace_back(x);
    }
  }
  py::ssize_t size() const { return static_cast<py::ssize_t>(q.size()); }
};

static py::array_t<double> w(std::shared_ptr<ctx> p, py::object c0) {
  dvec c(c0);
  py::ssize_t m = p->size(), n = nout(m, c.n);
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ci = c(i);
    check_c(ci);
    y(i) = volfi::implied_variance_otm(p->q[m == 1 ? 0 : i], ci);
  }
  return out;
}

static py::array_t<double> iv(std::shared_ptr<ctx> p, py::object c0, py::object t0) {
  dvec c(c0), t(t0);
  py::ssize_t m = p->size(), n = nout3(m, c.n, t.n);
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ci = c(i), ti = t(i);
    check_c(ci); check_t(ti);
    y(i) = volfi::implied_volatility_otm(p->q[m == 1 ? 0 : i], ci, ti);
  }
  return out;
}

static py::array_t<double> w_otm(py::object h0, py::object c0) {
  dvec h(h0), c(c0);
  py::ssize_t n = nout(h.n, c.n);
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double hi = h(i), ci = c(i);
    check_h(hi); check_c(ci);
    y(i) = volfi::implied_variance_otm(hi, ci);
  }
  return out;
}

static py::array_t<double> iv_otm(py::object h0, py::object c0, py::object t0) {
  dvec h(h0), c(c0), t(t0);
  py::ssize_t n = nout3(h.n, c.n, t.n);
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double hi = h(i), ci = c(i), ti = t(i);
    check_h(hi); check_c(ci); check_t(ti);
    y(i) = volfi::implied_volatility_otm(hi, ci, ti);
  }
  return out;
}

static py::array_t<double> w_call_norm(py::object k0, py::object c0) {
  dvec k(k0), c(c0);
  py::ssize_t n = nout(k.n, c.n);
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ki = k(i), ci = c(i);
    if (!std::isfinite(ki)) throw std::runtime_error("k must be finite");
    check_c(ci);
    y(i) = volfi::implied_variance_call_normalised(ki, ci);
  }
  return out;
}

static py::array_t<double> iv_call_norm(py::object k0, py::object c0, py::object t0) {
  dvec k(k0), c(c0), t(t0);
  py::ssize_t n = nout3(k.n, c.n, t.n);
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ki = k(i), ci = c(i), ti = t(i);
    if (!std::isfinite(ki)) throw std::runtime_error("k must be finite");
    check_c(ci); check_t(ti);
    y(i) = volfi::implied_volatility_call_normalised(ki, ci, ti);
  }
  return out;
}

static py::array_t<double> iv_call(py::object f0, py::object k0, py::object d0, py::object t0, py::object p0) {
  dvec f(f0), k(k0), d(d0), t(t0), p(p0);
  py::ssize_t n = nout5(f.n, k.n, d.n, t.n, p.n);
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double fi = f(i), ki = k(i), di = d(i), ti = t(i), pi = p(i);
    check_pos(fi, "f"); check_pos(ki, "k"); check_pos(di, "d"); check_t(ti); check_pos(pi, "price");
    y(i) = volfi::implied_volatility_call(fi, ki, di, ti, pi);
  }
  return out;
}

PYBIND11_MODULE(_volfi, m) {
  py::class_<ctx, std::shared_ptr<ctx>>(m, "Ctx")
    .def(py::init<py::object>())
    .def("size", &ctx::size)
    .def("w", [](std::shared_ptr<ctx> p, py::object c) { return w(p, c); })
    .def("iv", [](std::shared_ptr<ctx> p, py::object c, py::object t) { return iv(p, c, t); });
  m.def("ctx", [](py::object h) { return std::make_shared<ctx>(h); });
  m.def("w", &w);
  m.def("iv", &iv);
  m.def("w_otm", &w_otm);
  m.def("iv_otm", &iv_otm);
  m.def("w_call_norm", &w_call_norm);
  m.def("iv_call_norm", &iv_call_norm);
  m.def("iv_call", &iv_call);
  m.def("version", []() { return "0.1.7"; });
}
