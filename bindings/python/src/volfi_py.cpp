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

struct ctx {
  std::vector<volfi::otm_context> q;
  explicit ctx(py::array_t<double, py::array::c_style | py::array::forcecast> h) {
    auto r = h.unchecked<1>();
    if (r.shape(0) < 1) throw std::runtime_error("h must be non-empty");
    q.reserve(r.shape(0));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
      double x = r(i);
      if (!std::isfinite(x) || x < 0) throw std::runtime_error("h must be finite and non-negative");
      q.emplace_back(x);
    }
  }
  py::ssize_t size() const { return static_cast<py::ssize_t>(q.size()); }
};

static py::array_t<double> arr(py::object x) {
  return py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(x);
}

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

static double at(const py::detail::unchecked_reference<double, 1>& x, py::ssize_t i) {
  return x.shape(0) == 1 ? x(0) : x(i);
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

static py::array_t<double> w(std::shared_ptr<ctx> p, py::object c0) {
  auto c = arr(c0).reshape({arr(c0).size()});
  auto cr = c.unchecked<1>();
  py::ssize_t m = p->size(), n = nout(m, cr.shape(0));
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ci = at(cr, i);
    check_c(ci);
    y(i) = volfi::implied_variance_otm(p->q[m == 1 ? 0 : i], ci);
  }
  return out;
}

static py::array_t<double> iv(std::shared_ptr<ctx> p, py::object c0, py::object t0) {
  auto c = arr(c0).reshape({arr(c0).size()});
  auto t = arr(t0).reshape({arr(t0).size()});
  auto cr = c.unchecked<1>(), tr = t.unchecked<1>();
  py::ssize_t m = p->size(), n = nout3(m, cr.shape(0), tr.shape(0));
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ci = at(cr, i), ti = at(tr, i);
    check_c(ci); check_t(ti);
    y(i) = volfi::implied_volatility_otm(p->q[m == 1 ? 0 : i], ci, ti);
  }
  return out;
}

static py::array_t<double> w_otm(py::object h0, py::object c0) {
  auto h = arr(h0).reshape({arr(h0).size()});
  auto c = arr(c0).reshape({arr(c0).size()});
  auto hr = h.unchecked<1>(), cr = c.unchecked<1>();
  py::ssize_t n = nout(hr.shape(0), cr.shape(0));
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double hi = at(hr, i), ci = at(cr, i);
    check_h(hi); check_c(ci);
    y(i) = volfi::implied_variance_otm(hi, ci);
  }
  return out;
}

static py::array_t<double> iv_otm(py::object h0, py::object c0, py::object t0) {
  auto h = arr(h0).reshape({arr(h0).size()});
  auto c = arr(c0).reshape({arr(c0).size()});
  auto t = arr(t0).reshape({arr(t0).size()});
  auto hr = h.unchecked<1>(), cr = c.unchecked<1>(), tr = t.unchecked<1>();
  py::ssize_t n = nout3(hr.shape(0), cr.shape(0), tr.shape(0));
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double hi = at(hr, i), ci = at(cr, i), ti = at(tr, i);
    check_h(hi); check_c(ci); check_t(ti);
    y(i) = volfi::implied_volatility_otm(hi, ci, ti);
  }
  return out;
}

static py::array_t<double> w_call_norm(py::object k0, py::object c0) {
  auto k = arr(k0).reshape({arr(k0).size()});
  auto c = arr(c0).reshape({arr(c0).size()});
  auto kr = k.unchecked<1>(), cr = c.unchecked<1>();
  py::ssize_t n = nout(kr.shape(0), cr.shape(0));
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ki = at(kr, i), ci = at(cr, i);
    if (!std::isfinite(ki)) throw std::runtime_error("k must be finite");
    check_c(ci);
    y(i) = volfi::implied_variance_call_normalised(ki, ci);
  }
  return out;
}

static py::array_t<double> iv_call_norm(py::object k0, py::object c0, py::object t0) {
  auto k = arr(k0).reshape({arr(k0).size()});
  auto c = arr(c0).reshape({arr(c0).size()});
  auto t = arr(t0).reshape({arr(t0).size()});
  auto kr = k.unchecked<1>(), cr = c.unchecked<1>(), tr = t.unchecked<1>();
  py::ssize_t n = nout3(kr.shape(0), cr.shape(0), tr.shape(0));
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double ki = at(kr, i), ci = at(cr, i), ti = at(tr, i);
    if (!std::isfinite(ki)) throw std::runtime_error("k must be finite");
    check_c(ci); check_t(ti);
    y(i) = volfi::implied_volatility_call_normalised(ki, ci, ti);
  }
  return out;
}

static py::array_t<double> iv_call(py::object f0, py::object k0, py::object d0, py::object t0, py::object p0) {
  auto f = arr(f0).reshape({arr(f0).size()});
  auto k = arr(k0).reshape({arr(k0).size()});
  auto d = arr(d0).reshape({arr(d0).size()});
  auto t = arr(t0).reshape({arr(t0).size()});
  auto p = arr(p0).reshape({arr(p0).size()});
  auto fr = f.unchecked<1>(), kr = k.unchecked<1>(), dr = d.unchecked<1>(), tr = t.unchecked<1>(), pr = p.unchecked<1>();
  py::ssize_t n = nout5(fr.shape(0), kr.shape(0), dr.shape(0), tr.shape(0), pr.shape(0));
  py::array_t<double> out(n);
  auto y = out.mutable_unchecked<1>();
  for (py::ssize_t i = 0; i < n; ++i) {
    double fi = at(fr, i), ki = at(kr, i), di = at(dr, i), ti = at(tr, i), pi = at(pr, i);
    check_pos(fi, "f"); check_pos(ki, "k"); check_pos(di, "d"); check_t(ti); check_pos(pi, "price");
    y(i) = volfi::implied_volatility_call(fi, ki, di, ti, pi);
  }
  return out;
}

PYBIND11_MODULE(_volfi, m) {
  py::class_<ctx, std::shared_ptr<ctx>>(m, "Ctx")
    .def(py::init<py::array_t<double, py::array::c_style | py::array::forcecast>>())
    .def("size", &ctx::size)
    .def("w", &w)
    .def("iv", &iv);
  m.def("ctx", [](py::object h0) { return std::make_shared<ctx>(arr(h0).reshape({arr(h0).size()})); });
  m.def("w", &w);
  m.def("iv", &iv);
  m.def("w_otm", &w_otm);
  m.def("iv_otm", &iv_otm);
  m.def("w_call_norm", &w_call_norm);
  m.def("iv_call_norm", &iv_call_norm);
  m.def("iv_call", &iv_call);
  m.def("version", []() { return "0.1.7"; });
}
