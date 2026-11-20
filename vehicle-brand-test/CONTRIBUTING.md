# Contributing

## Workflow

1. Pick an issue from the current sprint.
2. Create a branch: `git checkout -b feat/short-description`
3. Commit often and push when meaningful.
4. Open a pull request.
5. Fill the PR template.
6. Request review.
7. Merge only after approval.
8. Delete the branch after merge.

## Branch naming

Use:

- `feat/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `exp/` for experiments
- `refactor/` for refactoring
- `test/` for tests
- `chore/` for maintenance

## Commit messages

Use:

`<type>: <subject>`

Examples:

- `docs: add team section to README`
- `feat: add batch prediction skeleton`
- `test: add package import smoke test`
- `chore: initialize project structure`

## Code style

- Use ruff.
- Add type hints on public functions where practical.
- Keep notebooks out of `src/`.
- Keep code modular and explainable.

## AI policy

- AI-assisted code is allowed if the author can explain it.
- Do not paste AI-written prose verbatim into README, slides, model card, decisions log, or presentation script.
- Every contributor must understand their own PR.

## Do not

- Commit datasets.
- Commit model checkpoints.
- Commit `.env` files or secrets.
- Commit `configs/datasets.local.yaml`.
- Force-push to `main`.
- Merge your own PR without review after branch protection is enabled.
