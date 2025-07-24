QMD=	quarto/_ldoc/testing.qmd \
	quarto/_ldoc/ext_la.qmd \
	quarto/_ldoc/sample.qmd \
	quarto/_ldoc/kfuns.qmd \
	quarto/_ldoc/kmats.qmd \
	quarto/_ldoc/gpp.qmd \
	quarto/_ldoc/hypers.qmd \
	quarto/_ldoc/acquisition.qmd \
	quarto/_ldoc/bo_step.qmd \
	quarto/_ldoc/test_ext_la.qmd \
	quarto/_ldoc/test_kfuns.qmd \
	quarto/_ldoc/test_kmats.qmd \
	quarto/_ldoc/test_gpp.qmd \
	quarto/_ldoc/test_hypers.qmd \
	quarto/_ldoc/test_acquisition.qmd

quarto/_ldoc/%.qmd: src/%.jl
	lua src/ldoc.lua -p quarto -highlight julia $< -o $@

.PHONY: all test clean distclean

all: ${QMD}

preview: ${QMD}
	(cd quarto; quarto preview basics.qmd)

render: ${QMD}
	(cd quarto; quarto render)

test:
	(cd src; julia run_tests.jl)

clean:
	rm -f *~

distclean: clean
	rm -f $(QMD)
	rm -rf quarto/_output
