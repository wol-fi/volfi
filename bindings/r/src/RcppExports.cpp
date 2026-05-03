#include <Rcpp.h>
using namespace Rcpp;

NumericVector iv_otm(NumericVector h, NumericVector c);
NumericVector vol_otm(NumericVector h, NumericVector c, NumericVector t);
NumericVector iv_atm(NumericVector c);
NumericVector black_otm(NumericVector w, NumericVector h);
NumericVector iv_call_norm(NumericVector k, NumericVector c);

RcppExport SEXP _volfiR_iv_otm(SEXP hSEXP, SEXP cSEXP){
BEGIN_RCPP
 return wrap(iv_otm(as<NumericVector>(hSEXP),as<NumericVector>(cSEXP)));
END_RCPP
}

RcppExport SEXP _volfiR_vol_otm(SEXP hSEXP, SEXP cSEXP, SEXP tSEXP){
BEGIN_RCPP
 return wrap(vol_otm(as<NumericVector>(hSEXP),as<NumericVector>(cSEXP),as<NumericVector>(tSEXP)));
END_RCPP
}

RcppExport SEXP _volfiR_iv_atm(SEXP cSEXP){
BEGIN_RCPP
 return wrap(iv_atm(as<NumericVector>(cSEXP)));
END_RCPP
}

RcppExport SEXP _volfiR_black_otm(SEXP wSEXP, SEXP hSEXP){
BEGIN_RCPP
 return wrap(black_otm(as<NumericVector>(wSEXP),as<NumericVector>(hSEXP)));
END_RCPP
}

RcppExport SEXP _volfiR_iv_call_norm(SEXP kSEXP, SEXP cSEXP){
BEGIN_RCPP
 return wrap(iv_call_norm(as<NumericVector>(kSEXP),as<NumericVector>(cSEXP)));
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
 {"_volfiR_iv_otm", (DL_FUNC) &_volfiR_iv_otm, 2},
 {"_volfiR_vol_otm", (DL_FUNC) &_volfiR_vol_otm, 3},
 {"_volfiR_iv_atm", (DL_FUNC) &_volfiR_iv_atm, 1},
 {"_volfiR_black_otm", (DL_FUNC) &_volfiR_black_otm, 2},
 {"_volfiR_iv_call_norm", (DL_FUNC) &_volfiR_iv_call_norm, 2},
 {NULL, NULL, 0}
};

RcppExport void R_init_volfiR(DllInfo *dll){
 R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
 R_useDynamicSymbols(dll, FALSE);
}
