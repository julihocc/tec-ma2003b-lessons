# Compiles a MA2003B LaTeX source (notes or presentation) using latexmk.
#
# Sets TEXINPUTS to the repo's shared latex/ style directory so that
# \usepackage{ma2003b-notes} / \usepackage{ma2003b-beamer} resolve regardless
# of the caller's current directory. Output (PDF/log/aux) is written next to
# the .tex source, matching this repo's existing layout.
#
# Uses xelatex: several notes sources contain literal Unicode math symbols
# (e.g. lambda, >=, cdot) that plain pdflatex cannot typeset; xelatex handles
# all 16 sources uniformly.
#
# Usage:
#   ./compile.ps1 lessons/L01_Regression_Analysis/notes/regression_analysis_notes.tex
#   ./compile.ps1 regression_analysis_slides.tex   (if already cd'd into the lesson dir)

param(
    [Parameter(Mandatory = $true)]
    [string]$TexFile
)

$RepoRoot = $PSScriptRoot
$env:TEXINPUTS = "$RepoRoot\latex//;"

latexmk -xelatex -interaction=nonstopmode -halt-on-error -cd $TexFile
exit $LASTEXITCODE
