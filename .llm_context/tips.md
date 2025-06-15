# LLM Collaboration Tips

These conventions help AI assistants propose consistent, high-quality changes.

1. **File Paths** – Always reference files relative to repo root.
2. **Atomic PRs** – Focus on one component or feature per PR.
3. **Tests First** – For new logic, ask to add unit tests in `tests/` _before_ feature code.
4. **Docs** – Update component `README.md` or `docs/` with every significant change.
5. **Prompt Injection Safety** – Do not store secrets (API keys, passwords) in repo. Use environment variables managed via 1Password/GitHub secrets.
6. **Language Choice** – Prefer Python 3.12 for backend unless otherwise justified.
7. **Type Annotations** – Mandatory for all new Python code (use `mypy` in CI).
