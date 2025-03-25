# Makefile for compiling LaTeX documentation for the generalized implementation

# Default target
all: generalized_orthogonal_vectors.pdf

# Compile the main document
generalized_orthogonal_vectors.pdf: generalized_orthogonal_vectors.tex
	pdflatex generalized_orthogonal_vectors.tex
	pdflatex generalized_orthogonal_vectors.tex  # Run twice for proper references and TOC

# Clean up auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.bbl *.blg *.nav *.snm *.vrb

# Clean up everything including the PDF
distclean: clean
	rm -f *.pdf

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the PDF documentation (default)"
	@echo "  clean      - Remove auxiliary files"
	@echo "  distclean  - Remove auxiliary files and PDFs"
	@echo "  help       - Display this help message"

.PHONY: all clean distclean help
