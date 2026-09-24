#!/bin/sh
# Builds the manuscript with changes marked against submission 0 (needs latexdiff).
set -e
cd "$(dirname "$0")/.."
latexdiff --flatten --math-markup=whole --graphics-markup=none \
  --exclude-textcmd="textbf,paragraph,section,subsection" \
  --config="PICTUREENV=(?:picture|DIFnomarkup|table|figure|tikzpicture|algorithm|longtable|tabular|center)[\w\d*@]*" \
  submission0/elsarticle-template-num.tex revision1/main.tex > revision1/main_diff.tex
cd revision1
python3 - <<'PY'
import re
s = open('main_diff.tex').read()
b = open('main.bbl').read()
i = s.find(r'\begin{thebibliography}'); j = s.find(r'\end{thebibliography}')
if 0 <= i < j:
    s = s[:i] + b.strip() + s[j + len(r'\end{thebibliography}'):]
s = re.sub(r'\\ref\{(app:[^}]*)\}', r'\\mbox{\\ref{\1}}', s)
open('main_diff.tex', 'w').write(s)
PY
for i in 1 2 3; do pdflatex -interaction=nonstopmode main_diff >/dev/null || true; done
cp main_diff.pdf RTXGNN_revision1_marked.pdf
echo built
