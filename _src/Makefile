SRC=sample.jl fdchecks.jl
QMD=$(SRC:.jl=.qmd)

all: ${QMD}

%.qmd: %.jl
	lua ldoc.lua -p quarto -highlight julia $< -o $@

clean:
	rm -f sample.qmd
