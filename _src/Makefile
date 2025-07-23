SRC=	sample.jl \
	fdchecks.jl \
	kfuns.jl \
	kmats.jl \
	gpp.jl \
	hypers.jl \
	acquisition.jl \
	bo_step.jl

QMD=$(SRC:.jl=.qmd)

.PHONY: all clean distclean

all: ${QMD}

%.qmd: %.jl
	lua ldoc.lua -p quarto -highlight julia $< -o $@

clean:
	rm -f *~

distclean: clean
	rm -f sample.qmd
