all:
	pandoc -d latex.yaml -o ib-writeup.pdf ib-writeup.md

latex:
	pandoc -d latex.yaml -o ib-writeup.tex ib-writeup.md
