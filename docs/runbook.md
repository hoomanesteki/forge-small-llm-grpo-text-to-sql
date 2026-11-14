# Runbook

Operational notes for running Forge. Sections are filled in by the stories that need them.

## Accounts

Added in FG-005.

## Kaggle jobs

Added in FG-303.

## Resume after a session limit

Added in FG-305.

## Troubleshooting

- **`ModuleNotFoundError: forge` on macOS.** macOS can flag the editable-install `.pth`
  file as hidden, and Python 3.12 skips hidden `.pth` files. `make setup` clears the flag;
  to fix it by hand run `chflags nohidden .venv/lib/python*/site-packages/*.pth`.
