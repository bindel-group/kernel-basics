SRC=sample.jl fdchecks.jl kfuns.jl kmats.jl gpp.jl
QMD=$(SRC:.jl=.qmd)

all: ${QMD}

%.qmd: %.jl
	lua ldoc.lua -p quarto -highlight julia $< -o $@

clean:
	rm -f sample.qmd
