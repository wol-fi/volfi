#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <volfi/volfi.hpp>
using namespace Rcpp;

struct vctx {
  std::vector<volfi::otm_context> q;
};

static XPtr<vctx> get_ctx(SEXP x) {
  XPtr<vctx> p(x);
  if (!p) stop("invalid ctx");
  return p;
}

static R_xlen_t nout(R_xlen_t a, R_xlen_t b) {
  if (a < 1 || b < 1) stop("inputs must be non-empty");
  R_xlen_t n = std::max(a, b);
  if ((a != 1 && a != n) || (b != 1 && b != n)) stop("lengths must be 1 or equal");
  return n;
}

static R_xlen_t nout3(R_xlen_t a, R_xlen_t b, R_xlen_t c) {
  if (a < 1 || b < 1 || c < 1) stop("inputs must be non-empty");
  R_xlen_t n = std::max(a, std::max(b, c));
  if ((a != 1 && a != n) || (b != 1 && b != n) || (c != 1 && c != n)) stop("lengths must be 1 or equal");
  return n;
}

static R_xlen_t nout5(R_xlen_t a, R_xlen_t b, R_xlen_t c, R_xlen_t d, R_xlen_t e) {
  if (a < 1 || b < 1 || c < 1 || d < 1 || e < 1) stop("inputs must be non-empty");
  R_xlen_t n = std::max(std::max(a, b), std::max(std::max(c, d), e));
  if ((a != 1 && a != n) || (b != 1 && b != n) || (c != 1 && c != n) || (d != 1 && d != n) || (e != 1 && e != n)) stop("lengths must be 1 or equal");
  return n;
}

static void check_h(double h) {
  if (!R_finite(h) || h < 0) stop("h must be finite and non-negative");
}

static void check_c(double c) {
  if (!R_finite(c) || c <= 0 || c >= 1) stop("c must be finite and inside (0,1)");
}

static void check_t(double t) {
  if (!R_finite(t) || t <= 0) stop("t must be finite and positive");
}

static void check_pos(double x, const char *nm) {
  if (!R_finite(x) || x <= 0) stop("%s must be finite and positive", nm);
}

extern "C" SEXP _volfiR_volfi_ctx(SEXP hSEXP) {
  BEGIN_RCPP
  NumericVector h(hSEXP);
  if (h.size() < 1) stop("h must be non-empty");
  XPtr<vctx> p(new vctx, true);
  p->q.reserve(h.size());
  for (R_xlen_t i = 0; i < h.size(); ++i) {
    double hi = h[i];
    check_h(hi);
    p->q.emplace_back(hi);
  }
  p.attr("class") = CharacterVector::create("volfi_ctx", "externalptr");
  return p;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_ctx_size(SEXP ctxSEXP) {
  BEGIN_RCPP
  XPtr<vctx> p = get_ctx(ctxSEXP);
  return wrap(static_cast<double>(p->q.size()));
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_w(SEXP ctxSEXP, SEXP cSEXP) {
  BEGIN_RCPP
  XPtr<vctx> p = get_ctx(ctxSEXP);
  NumericVector c(cSEXP);
  R_xlen_t n = nout(p->q.size(), c.size()), m = p->q.size();
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    double ci = c[c.size() == 1 ? 0 : i];
    check_c(ci);
    out[i] = volfi::implied_variance_otm(p->q[m == 1 ? 0 : i], ci);
  }
  return out;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_iv(SEXP ctxSEXP, SEXP cSEXP, SEXP tSEXP) {
  BEGIN_RCPP
  XPtr<vctx> p = get_ctx(ctxSEXP);
  NumericVector c(cSEXP), t(tSEXP);
  R_xlen_t n = nout3(p->q.size(), c.size(), t.size()), m = p->q.size();
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    double ci = c[c.size() == 1 ? 0 : i], ti = t[t.size() == 1 ? 0 : i];
    check_c(ci); check_t(ti);
    out[i] = volfi::implied_volatility_otm(p->q[m == 1 ? 0 : i], ci, ti);
  }
  return out;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_w_otm(SEXP hSEXP, SEXP cSEXP) {
  BEGIN_RCPP
  NumericVector h(hSEXP), c(cSEXP);
  R_xlen_t n = nout(h.size(), c.size());
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    double hi = h[h.size() == 1 ? 0 : i], ci = c[c.size() == 1 ? 0 : i];
    check_h(hi); check_c(ci);
    out[i] = volfi::implied_variance_otm(hi, ci);
  }
  return out;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_iv_otm(SEXP hSEXP, SEXP cSEXP, SEXP tSEXP) {
  BEGIN_RCPP
  NumericVector h(hSEXP), c(cSEXP), t(tSEXP);
  R_xlen_t n = nout3(h.size(), c.size(), t.size());
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    double hi = h[h.size() == 1 ? 0 : i], ci = c[c.size() == 1 ? 0 : i], ti = t[t.size() == 1 ? 0 : i];
    check_h(hi); check_c(ci); check_t(ti);
    out[i] = volfi::implied_volatility_otm(hi, ci, ti);
  }
  return out;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_w_call_norm(SEXP kSEXP, SEXP cSEXP) {
  BEGIN_RCPP
  NumericVector k(kSEXP), c(cSEXP);
  R_xlen_t n = nout(k.size(), c.size());
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    double ki = k[k.size() == 1 ? 0 : i], ci = c[c.size() == 1 ? 0 : i];
    if (!R_finite(ki)) stop("k must be finite");
    check_c(ci);
    out[i] = volfi::implied_variance_call_normalised(ki, ci);
  }
  return out;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_iv_call_norm(SEXP kSEXP, SEXP cSEXP, SEXP tSEXP) {
  BEGIN_RCPP
  NumericVector k(kSEXP), c(cSEXP), t(tSEXP);
  R_xlen_t n = nout3(k.size(), c.size(), t.size());
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    double ki = k[k.size() == 1 ? 0 : i], ci = c[c.size() == 1 ? 0 : i], ti = t[t.size() == 1 ? 0 : i];
    if (!R_finite(ki)) stop("k must be finite");
    check_c(ci); check_t(ti);
    out[i] = volfi::implied_volatility_call_normalised(ki, ci, ti);
  }
  return out;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_iv_call(SEXP fSEXP, SEXP kSEXP, SEXP dSEXP, SEXP tSEXP, SEXP priceSEXP) {
  BEGIN_RCPP
  NumericVector f(fSEXP), k(kSEXP), d(dSEXP), t(tSEXP), price(priceSEXP);
  R_xlen_t n = nout5(f.size(), k.size(), d.size(), t.size(), price.size());
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    double fi = f[f.size() == 1 ? 0 : i], ki = k[k.size() == 1 ? 0 : i], di = d[d.size() == 1 ? 0 : i], ti = t[t.size() == 1 ? 0 : i], pi = price[price.size() == 1 ? 0 : i];
    check_pos(fi, "f"); check_pos(ki, "k"); check_pos(di, "d"); check_t(ti); check_pos(pi, "price");
    out[i] = volfi::implied_volatility_call(fi, ki, di, ti, pi);
  }
  return out;
  END_RCPP
}

extern "C" SEXP _volfiR_volfi_version() {
  BEGIN_RCPP
  return wrap("0.1.8");
  END_RCPP
}

static const R_CallMethodDef call_entries[] = {
  {"_volfiR_volfi_ctx", (DL_FUNC) &_volfiR_volfi_ctx, 1},
  {"_volfiR_volfi_ctx_size", (DL_FUNC) &_volfiR_volfi_ctx_size, 1},
  {"_volfiR_volfi_w", (DL_FUNC) &_volfiR_volfi_w, 2},
  {"_volfiR_volfi_iv", (DL_FUNC) &_volfiR_volfi_iv, 3},
  {"_volfiR_volfi_w_otm", (DL_FUNC) &_volfiR_volfi_w_otm, 2},
  {"_volfiR_volfi_iv_otm", (DL_FUNC) &_volfiR_volfi_iv_otm, 3},
  {"_volfiR_volfi_w_call_norm", (DL_FUNC) &_volfiR_volfi_w_call_norm, 2},
  {"_volfiR_volfi_iv_call_norm", (DL_FUNC) &_volfiR_volfi_iv_call_norm, 3},
  {"_volfiR_volfi_iv_call", (DL_FUNC) &_volfiR_volfi_iv_call, 5},
  {"_volfiR_volfi_version", (DL_FUNC) &_volfiR_volfi_version, 0},
  {NULL, NULL, 0}
};

extern "C" void R_init_volfiR(DllInfo *dll) {
  R_registerRoutines(dll, NULL, call_entries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
