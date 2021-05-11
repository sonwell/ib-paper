texfiles=algo-figure.tex \
		conclusion.tex \
		convergence-study.tex \
		frame-coupling.tex \
		grid-dependence.tex \
		grid-figure.tex \
		ib-discretization.tex \
		ib-overview.tex \
		interp-parallel.tex \
		introduction.tex \
		results.tex \
		serial-comparison.tex \
		spread-parallel.tex \
		spread-serial.tex \
		strong-scaling.tex \
		weak-scaling.tex \
		ib-paper.bib

all: ib-paper-ijhpca.pdf

ib-paper-ijhpca.pdf: ib-paper-ijhpca.tex ${texfiles}
	latexmk -pdf ib-paper-ijhpca

clean:
	rm *.log
	rm *.aux
	rm *.fdb_latexmk
	rm *.fls
	rm *.out
	rm *.bbl
	rm *.blg
