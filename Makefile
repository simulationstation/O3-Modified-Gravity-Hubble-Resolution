.PHONY: help reproduce reproduce-out cqg-paper cqg-dark-siren

PY ?= python3

help:
	@echo "Targets:"
	@echo "  make reproduce            Generate a timestamped reviewer-seed report (headlines only)."
	@echo "  make reproduce-out OUT=…  Same as reproduce, but write to explicit OUT dir."
	@echo "  make cqg-paper            Rebuild CQG_PAPER/hubble_tension_cqg.pdf (pdflatex x2)."
	@echo "  make cqg-dark-siren       Rebuild CQG_DARK_SIREN/dark_siren_cqg.pdf (pdflatex x2)."

reproduce:
	@$(PY) scripts/reproduce_reviewer_seed.py

reproduce-out:
	@if [ -z "$(OUT)" ]; then echo "OUT is required, e.g. make reproduce-out OUT=outputs/reviewer_seed_custom"; exit 2; fi
	@$(PY) scripts/reproduce_reviewer_seed.py --out "$(OUT)"

cqg-paper:
	@cd CQG_PAPER && pdflatex -interaction=nonstopmode -halt-on-error hubble_tension_cqg.tex >/dev/null
	@cd CQG_PAPER && pdflatex -interaction=nonstopmode -halt-on-error hubble_tension_cqg.tex >/dev/null
	@echo "[done] built CQG_PAPER/hubble_tension_cqg.pdf"

cqg-dark-siren:
	@cd CQG_DARK_SIREN && pdflatex -interaction=nonstopmode -halt-on-error dark_siren_cqg.tex >/dev/null
	@cd CQG_DARK_SIREN && pdflatex -interaction=nonstopmode -halt-on-error dark_siren_cqg.tex >/dev/null
	@echo "[done] built CQG_DARK_SIREN/dark_siren_cqg.pdf"
