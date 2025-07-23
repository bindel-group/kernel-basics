QMD=	../_ldoc/testing.qmd \
	../_ldoc/sample.qmd \
	../_ldoc/kfuns.qmd \
	../_ldoc/kmats.qmd \
	../_ldoc/gpp.qmd \
	../_ldoc/hypers.qmd \
	../_ldoc/acquisition.qmd \
	../_ldoc/bo_step.qmd \
	../_ldoc/test_kfuns.qmd \
	../_ldoc/test_kmats.qmd \
	../_ldoc/test_gpp.qmd \
	../_ldoc/test_acquisition.qmd

../_ldoc/%.qmd: %.jl
	lua ldoc.lua -p quarto -highlight julia $< -o $@

.PHONY: all test clean distclean

all: ${QMD}

test:
	julia run_tests.jl

clean:
	rm -f *~

distclean: clean
	rm -f $(QMD)
