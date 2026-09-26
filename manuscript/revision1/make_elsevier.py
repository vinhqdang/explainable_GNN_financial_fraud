"""Builds the flat submission package for the Elsevier system: one .tex file with
every \\input, the number macros and the bibliography inlined, plus the figures
as PNG files, all in one directory without subfolders, zipped.

Usage (after build.sh, which produces main.bbl):
    python make_elsevier.py
"""
import os
import re
import shutil
import subprocess
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "elsevier")
NAME = "RTXGNN_revision1"


def inline(path):
    """Recursively replace \\input{file} by the content of the file."""
    text = open(path).read()

    def repl(m):
        f = m.group(1)
        f = f if f.endswith(".tex") else f + ".tex"
        return inline(os.path.join(HERE, f)).rstrip("\n")

    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    tex = inline(os.path.join(HERE, "main.tex"))
    # bibliography: inline the formatted reference list produced by BibTeX
    bbl = open(os.path.join(HERE, "main.bbl")).read().strip()
    tex = re.sub(r"\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}", lambda m: bbl, tex)
    # figures: PDF -> PNG (300 dpi), referenced without a folder
    for f in sorted(os.listdir(os.path.join(HERE, "figures"))):
        if f.endswith(".pdf"):
            base = f[:-4]
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile",
                            os.path.join(HERE, "figures", f), os.path.join(OUT, base)], check=True)
            tex = tex.replace(f"figures/{f}", f"{base}.png")
    assert "\\input{" not in tex and "figures/" not in tex
    open(os.path.join(OUT, f"{NAME}.tex"), "w").write(tex)
    # verify that the flat package compiles on its own
    for _ in range(2):
        # elsarticle.cls is provided by the Elsevier system; locally it lives next to main.tex
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{NAME}.tex"], cwd=OUT,
                       stdout=subprocess.DEVNULL, check=False,
                       env=dict(os.environ, TEXINPUTS=HERE + "//:"))
    log = open(os.path.join(OUT, f"{NAME}.log")).read()
    print("undefined references:", log.count("undefined"), "| errors:", log.count("\n! "))
    for ext in (".aux", ".log", ".out", ".spl", ".pdf"):
        p = os.path.join(OUT, NAME + ext)
        if ext == ".pdf" and os.path.exists(p):
            shutil.move(p, os.path.join(HERE, f"{NAME}_elsevier_check.pdf"))
        elif os.path.exists(p):
            os.remove(p)
    zpath = os.path.join(HERE, f"{NAME}_latex.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, f), f)
    print("wrote", zpath, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
