# Contributing

## Dev Setup
```bash
conda env create -f environment.yml
conda activate lightgan-ld
pre-commit install
pytest -q
```

## PR Checklist
- Unit tests pass (CPU).
- `pre-commit run -a` passes.
- Provide config diff & hardware in PR body.
- Update docs for new flags.

Conventional commits encouraged (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
