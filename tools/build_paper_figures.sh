#!/usr/bin/env bash
# 논문 그림(report_assets/paper/*.tex, TikZ) 을 PDF -> SVG(+PNG 미리보기) 로 빌드한다.
#   사용:  tools/build_paper_figures.sh            # 전부
#          tools/build_paper_figures.sh fig1_late_fusion_overview
# 요구: pdflatex (사용자 공간 TinyTeX, ~/.TinyTeX), poppler-utils (pdftocairo, pdftoppm).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/report_assets/paper"
TEXBIN="$(ls -d "$HOME"/.TinyTeX/bin/*/ 2>/dev/null | head -1 || true)"
[ -n "$TEXBIN" ] && export PATH="$TEXBIN:$PATH"
command -v pdflatex >/dev/null || { echo "pdflatex not found (install TinyTeX: https://yihui.org/tinytex/)"; exit 1; }
command -v pdftocairo >/dev/null || { echo "pdftocairo not found (apt: poppler-utils)"; exit 1; }

cd "$DIR"
mkdir -p build
if [ $# -gt 0 ]; then names=("$@"); else names=(fig1_late_fusion_overview fig2_fusion_strategies); fi
for f in "${names[@]}"; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build "$f.tex" > "build/$f.stdout" 2>&1 \
    || { echo "!! pdflatex failed for $f (see build/$f.log)"; grep -n -A3 "^!" "build/$f.log" | head -30; exit 1; }
  cp "build/$f.pdf" "$f.pdf"
  pdftocairo -svg "$f.pdf" "$f.svg"
  pdftoppm -png -r 220 -singlefile "$f.pdf" "$f"
  echo "built $f.{pdf,svg,png}"
done
