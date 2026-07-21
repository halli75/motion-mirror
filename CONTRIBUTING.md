# Contributing to Motion Mirror

Thanks for your interest in improving Motion Mirror. This document covers everything you need to get a change from idea to merged PR.

## Development setup

```bash
git clone https://github.com/halli75/motion-mirror.git
cd motion-mirror
pip install -e ".[dev]"
git config core.hooksPath .githooks
```

The last line enables the repository's commit hooks (see [Commit messages](#commit-messages)).

For GPU work you'll also need the CUDA extras and model weights:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[cuda,gpu-inference]"
motion-mirror download --model dwpose
```

## Running tests

The suite is split by the `gpu` pytest marker:

```bash
pytest -m "not gpu"     # fast suite — no GPU or weights needed; this is what CI runs
pytest -m gpu           # needs CUDA + downloaded weights
```

CI (`.github/workflows/ci.yml`) runs shellcheck on the pod scripts, import checks, and the non-GPU suite on every push and PR. A PR must be green before review.

GPU-touching changes should also pass the relevant `pytest -m gpu` tests locally, or be validated with the reproducible RunPod harness in [`runpod-validation/`](runpod-validation/README.md).

## Making changes

- Keep changes minimal and focused — one concern per PR.
- Match the surrounding code style. Public functions get a one-line docstring.
- Comments explain *why*, not *what*. Don't add comments that restate the code.
- New backends, CLI flags, or config fields need tests in the non-GPU suite (mock what needs a GPU).
- Update `README.md` if you change user-facing behavior.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(cli): add --seed option to run command
fix(vace): release GGUF transformer handle before cache clear
docs: clarify 14B VRAM requirements
```

**No AI-assistant attribution.** Do not include `Co-Authored-By:` trailers or "Generated with ..." lines for AI tools in commit messages. The `.githooks/commit-msg` hook strips them automatically once you've run `git config core.hooksPath .githooks`.

## Pull requests

1. Fork and branch from `main`.
2. Make your change with tests.
3. Ensure `pytest -m "not gpu"` passes and shellcheck is clean if you touched shell scripts.
4. Open a PR describing what changed and why. Link related issues.

## Reporting issues

Use [GitHub Issues](https://github.com/halli75/motion-mirror/issues). For bugs, include: OS, GPU + driver version, Python version, the exact command, and the full traceback.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
