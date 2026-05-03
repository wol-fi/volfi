#include <Rcpp.h>
using namespace Rcpp;

NumericVector iv(NumericVector k, NumericVector c);
NumericVector vol(NumericVector k, NumericVector c, NumericVector t);
NumericVector iv_atm(NumericVector c);
NumericVector black(NumericVector w, NumericVector h);

RcppExport SEXP _volfiR_iv(SEXP kSEXP, SEXP cSEXP){
BEGIN_RCPP
 return wrap(iv(as<NumericVector>(kSEXP),as<NumericVector>(cSEXP)));
END_RCPP
}

RcppExport SEXP _volfiR_vol(SEXP kSEXP, SEXP cSEXP, SEXP tSEXP){
BEGIN_RCPP
 return wrap(vol(as<NumericVector>(kSEXP),as<NumericVector>(cSEXP),as<NumericVector>(tSEXP)));
END_RCPP
}

RcppExport SEXP _volfiR_iv_atm(SEXP cSEXP){
BEGIN_RCPP
 return wrap(iv_atm(as<NumericVector>(cSEXP)));
END_RCPP
}

RcppExport SEXP _volfiR_black(SEXP wSEXP, SEXP hSEXP){
BEGIN_RCPP
 return wrap(black(as<NumericVector>(wSEXP),as<NumericVector>(hSEXP)));
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
 {"_volfiR_iv", (DL_FUNC) &_volfiR_iv, 2},
 {"_volfiR_vol", (DL_FUNC) &_volfiR_vol, 3},
 {"_volfiR_iv_atm", (DL_FUNC) &_volfiR_iv_atm, 1},
 {"_volfiR_black", (DL_FUNC) &_volfiR_black, 2},
 {NULL, NULL, 0}
};

RcppExport void R_init_volfiR(DllInfo *dll){
 R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
 R_useDynamicSymbols(dll, FALSE);
}
