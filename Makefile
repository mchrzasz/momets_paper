all: draft.pdf notes.pdf

draft.pdf: draft.tex tab-btokstarll-1to6.tex tab-btokstarll-7to12.tex tab-btokstarll-13to18.tex
	pdflatex draft.tex
	bibtex draft
	pdflatex draft.tex
	pdflatex draft.tex

notes.pdf: notes.tex
	pdflatex notes.tex
	bibtex notes
	pdflatex notes.tex
	pdflatex notes.tex

tab-btokstarll-%.tex: tab-btokstarll-%.tex.in
	sed \
	    -e 's/\\text{eps}(\([^,]*\),\([^)]*\))/\\eps_{\1}^{(\2)}/g' \
	    < $< \
	    > $@
