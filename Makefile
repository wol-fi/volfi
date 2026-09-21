# volfi v0.3.1 is header-only. The verification and benchmark harness lives in reproduce/;
# this top-level Makefile delegates to it.
#
#   make check   build and run the host-independent checks (accuracy + scalar==batch)
#   make verify  the accuracy / bit-identity suite alone
#   make smoke   the quick self-check
#   make book    the book kernel's SIMD-twin bit-identity and truth-set gate
#   make clean
#
# The LBR head-to-head benchmark needs Jaeckel's Let's Be Rational sources, which are not
# redistributed here; see reproduce/README.md and reproduce/BENCHMARK_PROTOCOL.md.

.PHONY: test check verify smoke book clean

test check:
	$(MAKE) -C reproduce check

verify:
	$(MAKE) -C reproduce verify

smoke:
	$(MAKE) -C reproduce smoke

book:
	$(MAKE) -C reproduce book

clean:
	$(MAKE) -C reproduce clean
