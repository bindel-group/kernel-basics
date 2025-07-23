SRC=	testing.jl \
	sample.jl \
	kfuns.jl \
	kmats.jl \
	gpp.jl \
	hypers.jl \
	acquisition.jl \
	bo_step.jl \
	test_kfuns.jl \
	test_kmats.jl

QMD=$(SRC:.jl=.qmd)

%.qmd: %.jl
	lua ldoc.lua -p quarto -highlight julia $< -o $@

.PHONY: all test clean distclean

all: ${QMD}

test:
	julia test_kfuns.jl
	julia test_kmats.jl

clean:
	rm -f *~

distclean: clean
	rm -f sample.qmd
