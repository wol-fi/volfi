#include <volfi/volfi.hpp>
#include <cmath>
#include <cstdio>

int main(){
 int fail=0;
 double v=0.4;
 double c=2*volfi::phi_cdf(0.5*v)-1;
 double w=volfi::implied_variance_call_normalised(0.0,c);
 if(!(std::isfinite(w) && std::fabs(std::sqrt(w)-v)<1e-14)){std::printf("valid_atm_failed %.17g\n",w); fail=1;}
 double bad_otm=volfi::implied_variance_otm(0.0,0.1);
 if(!std::isnan(bad_otm)){std::printf("bad_otm_not_nan %.17g\n",bad_otm); fail=1;}
 double k=-0.5, lb=1-std::exp(k);
 double bad_norm=volfi::implied_variance_call_normalised(k,0.5*lb);
 if(!std::isnan(bad_norm)){std::printf("bad_norm_not_nan %.17g\n",bad_norm); fail=1;}
 double bad_full=volfi::implied_volatility_call(100.0,60.0,1.0,1.0,1.0);
 if(!std::isnan(bad_full)){std::printf("bad_full_not_nan %.17g\n",bad_full); fail=1;}
 double good=volfi::implied_variance_otm(0.5,volfi::black_otm_from_variance(0.16,0.5));
 if(!(std::isfinite(good) && good>0.0)){std::printf("good_otm_failed %.17g\n",good); fail=1;}
 std::printf("domain checks %s\n",fail?"failed":"passed");
 return fail;
}
