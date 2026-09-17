# Compare rendered IRs against stored snapshots (tests/frontend/snapshots/).
snapshots-check:
	uv run pytest tests/frontend/test_snapshots.py

# (Re)write snapshots for every program in tests/frontend/programs.py.
# Review the generated .ir files by eye before committing (§8 checkpoint).
snapshots-update:
	NATSUNE_UPDATE_SNAPSHOTS=1 uv run pytest tests/frontend/test_snapshots.py
