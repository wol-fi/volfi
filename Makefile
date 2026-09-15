# volfi v0.3.0 is header-only. The verification and benchmark harness lives in reproduce/;
# this top-level Makefile delegates to it.
#
#   make check   build and run the host-independent checks (accuracy + scalar==batch)
#   make verify  the accuracy / bit-identity suite alone
#   make smoke   the quick self-check
#   make book    the book kernel's SIMD-twin bit-identity and truth-set gate (reproduce/book)
#   make clean
#
# The LBR head-to-head benchmark needs Jaeckel's Let's Be Rational sources, which are not
# redistributed here; see reproduce/README.md and reproduce/BENCHMARK_PROTOCOL.md.

.PHONY: test check verify smoke book clean

test check:
	$(MAKE) -C reproduce check

verify:
	$(MAKE) -C reproduce vv
	cd reproduce && ./vv

smoke:
	$(MAKE) -C reproduce smoke
	cd reproduce && ./smoke | tail -2

book:
	cd reproduce/book && g++ -std=c++17 -O3 -march=native -ffp-contract=off -fno-fast-math -funroll-loops -I../../include/volfi wb_vec_gate.cpp -o wbvg && ./wbvg

clean:
	$(MAKE) -C reproduce clean
	rm -f reproduce/book/wbvg
