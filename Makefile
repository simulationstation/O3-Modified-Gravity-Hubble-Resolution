.PHONY: help reproduce reproduce-out cqg-paper

PY ?= python3

help:
	@echo "Targets:"
	@echo "  make reproduce            Generate a timestamped reviewer-seed report (headlines only)."
	@echo "  make reproduce-out OUT=…  Same as reproduce, but write to explicit OUT dir."
	@echo "  make cqg-paper            Rebuild CQG_PAPER/dark_siren_cqg.pdf (pdflatex x2)."

reproduce:
	@$(PY) scripts/reproduce_reviewer_seed.py

reproduce-out:
	@if [ -z "$(OUT)" ]; then echo "OUT is required, e.g. make reproduce-out OUT=outputs/reviewer_seed_custom"; exit 2; fi
	@$(PY) scripts/reproduce_reviewer_seed.py --out "$(OUT)"

cqg-paper:
	@cd CQG_PAPER && pdflatex -interaction=nonstopmode -halt-on-error dark_siren_cqg.tex >/dev/null
	@cd CQG_PAPER && pdflatex -interaction=nonstopmode -halt-on-error dark_siren_cqg.tex >/dev/null
	@echo "[done] built CQG_PAPER/dark_siren_cqg.pdf"
