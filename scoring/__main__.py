"""Allow `python -m scoring.enrich_csvs` and `python -m scoring`."""

from scoring.enrich_csvs import main

if __name__ == "__main__":
    raise SystemExit(main())
