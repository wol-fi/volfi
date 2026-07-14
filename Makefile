# volfi v0.2.0 is header-only. The verification and benchmark harness lives in reproduce/;
# this top-level Makefile delegates to it.
#
#   make check   build and run the host-independent checks (accuracy + scalar==batch)
#   make verify  the accuracy / bit-identity suite alone
#   make smoke   the quick self-check
#   make clean
#
# The LBR head-to-head benchmark needs Jaeckel's Let's Be Rational sources, which are not
# redistributed here; see reproduce/README.md and reproduce/BENCHMARK_PROTOCOL.md.

.PHONY: test check verify smoke clean

test check:
	$(MAKE) -C reproduce check

verify:
	$(MAKE) -C reproduce vv
	cd reproduce && ./vv

smoke:
	$(MAKE) -C reproduce smoke
	cd reproduce && ./smoke | tail -2

clean:
	$(MAKE) -C reproduce clean
