#!/bin/sh
# Builds the revised manuscript and the response letter (needs pdflatex + bibtex).
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode main >/dev/null
pdflatex -interaction=nonstopmode main >/dev/null
cd response
pdflatex -interaction=nonstopmode response >/dev/null
pdflatex -interaction=nonstopmode response >/dev/null
cd ..
cp main.pdf RTXGNN_revision1.pdf
cp response/response.pdf response/response_to_reviewers.pdf
echo built
