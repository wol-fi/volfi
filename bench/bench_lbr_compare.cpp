#include <volfi/volfi.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <lets_be_rational.h>
#include <random>
#include <vector>

constexpr int fixed_nv = 41;
constexpr int fixed_reps = 5000;
constexpr int random_acc_cases = 200000;
constexpr int random_timing_cases = 5000;
constexpr int random_timing_reps = 1000;
constexpr int runs = 9;
constexpr std::uint64_t random_seed = 20260502ULL;
constexpr double sr2pi = 2.50662827463100050241576528481104525;

inline double qnorm0(double p) {
  const double a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02, a3 = -2.759285104469687e+02,
               a4 = 1.383577518672690e+02, a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00;
  const double b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02, b3 = -1.556989798598866e+02,
               b4 = 6.680131188771972e+01, b5 = -1.328068155288572e+01;
  const double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01, c3 = -2.400758277161838e+00,
               c4 = -2.549732539343734e+00, c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00;
  const double d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01, d3 = 2.445134137142996e+00,
               d4 = 3.754408661907416e+00;
  double q, r;
  if (p < 0.02425) {
    q = std::sqrt(-2 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
           ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
  }
  if (p <= 0.97575) {
    q = p - 0.5;
    r = q * q;
    return ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
           (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
  }
  q = std::sqrt(-2 * std::log1p(-p));
  return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
         ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
}

inline double qnorm2(double p) {
  double x = qnorm0(p);
  for (int i = 0; i < 2; ++i) {
    const double e = volfi::phi_cdf(x) - p;
    const double u = e * sr2pi * std::exp(0.5 * x * x);
    x -= u / (1 + 0.5 * x * u);
  }
  return x;
}

inline double now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

struct CaseData {
  std::vector<volfi::otm_context> q;
  std::vector<double> c;
  std::vector<double> wt;
  std::vector<double> vt;
};

void stat(const char* name, std::vector<double> x) {
  std::sort(x.begin(), x.end());
  double s = 0;
  for (double y : x) s += y;
  std::printf("%s_mean_ns_per_eval %.17g\n", name, s / x.size());
  std::printf("%s_median_ns_per_eval %.17g\n", name, x[x.size() / 2]);
  std::printf("%s_min_ns_per_eval %.17g\n", name, x.front());
  std::printf("%s_max_ns_per_eval %.17g\n", name, x.back());
}

template <class F>
double bench(int ncase, int reps, F f) {
  volatile double sink = 0;
  const double t0 = now();
  for (int r = 0; r < reps; ++r) {
    for (int i = 0; i < ncase; ++i) sink += std::sqrt(f(i));
  }
  const double t1 = now();
  if (sink == 123456789) std::printf("%g\n", static_cast<double>(sink));
  return 1e9 * (t1 - t0) / (reps * ncase);
}

template <class F>
void acc(int ncase, const char* name, F f, const std::vector<double>& wt, const std::vector<double>& vt) {
  double maw = 0, mxw = 0, mrw = 0, mav = 0, mxv = 0, mrv = 0;
  for (int i = 0; i < ncase; ++i) {
    const double wh = f(i);
    const double v = std::sqrt(wh);
    const double ew = std::fabs(wh - wt[i]);
    const double ev = std::fabs(v - vt[i]);
    maw += ew;
    mxw = std::max(mxw, ew);
    mrw = std::max(mrw, ew / wt[i]);
    mav += ev;
    mxv = std::max(mxv, ev);
    mrv = std::max(mrv, ev / vt[i]);
  }
  std::printf("%s_mean_abs_variance_error %.17g\n", name, maw / ncase);
  std::printf("%s_max_abs_variance_error %.17g\n", name, mxw);
  std::printf("%s_max_rel_variance_error %.17g\n", name, mrw);
  std::printf("%s_mean_abs_vol_error %.17g\n", name, mav / ncase);
  std::printf("%s_max_abs_vol_error %.17g\n", name, mxv);
  std::printf("%s_max_rel_vol_error %.17g\n", name, mrv);
}

inline double lbr_variance_from_otm(const volfi::otm_context& q, double c) {
  const double beta = c / q.eh2;
  const double s = NormalisedImpliedBlackVolatility(beta, -q.h, 1.0);
  return s * s;
}

template <std::size_t N>
CaseData make_fixed_grid(const std::array<double, N>& delta) {
  CaseData data;
  data.q.reserve(fixed_nv * N);
  data.c.reserve(fixed_nv * N);
  data.wt.reserve(fixed_nv * N);
  data.vt.reserve(fixed_nv * N);
  for (int iv = 0; iv < fixed_nv; ++iv) {
    const double v = (iv == 0) ? 0.01 : 0.05 * iv;
    for (double d : delta) {
      const double k = v * (0.5 * v - qnorm2(d));
      const double h = std::fabs(k);
      const double w = v * v;
      data.q.emplace_back(h);
      data.c.push_back(volfi::black_otm_from_variance(w, data.q.back()));
      data.wt.push_back(w);
      data.vt.push_back(v);
    }
  }
  return data;
}

CaseData make_random_grid(int ncase, std::uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<double> dv(0.01, 2.0);
  std::uniform_real_distribution<double> dd(0.01, 0.9);
  CaseData data;
  data.q.reserve(ncase);
  data.c.reserve(ncase);
  data.wt.reserve(ncase);
  data.vt.reserve(ncase);
  for (int i = 0; i < ncase; ++i) {
    const double v = dv(gen);
    const double delta = dd(gen);
    const double k = v * (0.5 * v - qnorm2(delta));
    const double h = std::fabs(k);
    const double w = v * v;
    data.q.emplace_back(h);
    data.c.push_back(volfi::black_otm_from_variance(w, data.q.back()));
    data.wt.push_back(w);
    data.vt.push_back(v);
  }
  return data;
}

void run_fixed() {
  auto data = make_fixed_grid(std::array<double, 11>{0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90});
  const int ncase = static_cast<int>(data.c.size());
  std::printf("grid fixed_otm\n");
  std::printf("cases %d\n", ncase);
  std::printf("repetitions_per_run %d\n", fixed_reps);
  std::printf("evaluations_per_run %d\n", fixed_reps * ncase);
  std::printf("runs %d\n", runs);
  acc(ncase, "volfi", [&](int i) { return volfi::implied_variance_otm(data.q[i], data.c[i]); }, data.wt, data.vt);
  acc(ncase, "lets_be_rational", [&](int i) { return lbr_variance_from_otm(data.q[i], data.c[i]); }, data.wt, data.vt);
  std::vector<double> tv, tl;
  for (int j = 0; j < runs; ++j) {
    tv.push_back(bench(ncase, fixed_reps, [&](int i) { return volfi::implied_variance_otm(data.q[i], data.c[i]); }));
    tl.push_back(bench(ncase, fixed_reps, [&](int i) { return lbr_variance_from_otm(data.q[i], data.c[i]); }));
  }
  stat("volfi", tv);
  stat("lets_be_rational", tl);
  std::printf("\n");
}

void run_random() {
  auto acc_data = make_random_grid(random_acc_cases, random_seed);
  auto timing_data = make_random_grid(random_timing_cases, random_seed);
  std::printf("grid random_otm\n");
  std::printf("accuracy_cases %d\n", random_acc_cases);
  std::printf("random_seed %llu\n", static_cast<unsigned long long>(random_seed));
  std::printf("timing_cases %d\n", random_timing_cases);
  std::printf("repetitions_per_run %d\n", random_timing_reps);
  std::printf("evaluations_per_run %d\n", random_timing_reps * random_timing_cases);
  std::printf("runs %d\n", runs);
  acc(random_acc_cases, "volfi", [&](int i) { return volfi::implied_variance_otm(acc_data.q[i], acc_data.c[i]); }, acc_data.wt, acc_data.vt);
  acc(random_acc_cases, "lets_be_rational", [&](int i) { return lbr_variance_from_otm(acc_data.q[i], acc_data.c[i]); }, acc_data.wt, acc_data.vt);
  std::vector<double> tv, tl;
  for (int j = 0; j < runs; ++j) {
    tv.push_back(bench(random_timing_cases, random_timing_reps, [&](int i) { return volfi::implied_variance_otm(timing_data.q[i], timing_data.c[i]); }));
    tl.push_back(bench(random_timing_cases, random_timing_reps, [&](int i) { return lbr_variance_from_otm(timing_data.q[i], timing_data.c[i]); }));
  }
  stat("volfi", tv);
  stat("lets_be_rational", tl);
  std::printf("\n");
}

int main() {
  run_fixed();
  run_random();
}
