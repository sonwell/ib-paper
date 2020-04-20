all:
	pandoc -d latex.yaml

latex:
	pandoc -d latex.yaml -t latex -o ib-paper.tex
