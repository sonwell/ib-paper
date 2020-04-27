all: 
	latexmk -pdf ib-paper

clean:
	rm *.log
	rm *.aux
	rm *.fdb_latexmk
	rm *.fls
	rm *.out
