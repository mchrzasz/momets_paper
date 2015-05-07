all: draft.pdf

draft.pdf: draft.tex tab-btokstarll-1to6.tex tab-btokstarll-7to12.tex tab-btokstarll-13to18.tex
	pdflatex draft.tex
	bibtex draft
	pdflatex draft.tex
	pdflatex draft.tex

tab-btokstarll-%.tex: tab-btokstarll-%.tex.in
	sed \
	    -e 's/\\text{eps}(\([^,]*\),\([^)]*\))/\\eps_{\1}^{(\2)}/g' \
	    < $< \
	    > $@

PLOTS = \
	fig-topology.pdf \
	figs/Q2_5_6_S5_200.pdf \
	figs/Q2_5_6_S7_200.pdf \
	figs/Q2_1_2_S5.pdf \
	figs/S5_scat.pdf \
	figs/S7_scat.pdf

.PHONY: dist
dist:
	tar zcf paper-$$(date +%Y-%m-%d).tar.gz \
	    draft.tex \
	    draft.bbl \
	    $(PLOTS)
