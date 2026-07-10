# Releasing `ds-mcp-server`

This project uses **git tags as the source of truth for versions**. Versioning is
handled by [setuptools-scm](https://github.com/pypa/setuptools_scm) — you never
edit a version number by hand. Publishing to PyPI is fully automated by GitHub
Actions using [PyPI Trusted Publishing (OIDC)](https://docs.pypi.org/trusted-publishers/) —
no API tokens are stored in this repo.

---

## TL;DR — cutting a release

```bash
git checkout main
git pull

# 1. Make sure tests pass locally
pytest

# 2. Tag the commit you want to release (follow SemVer: MAJOR.MINOR.PATCH)
git tag -a v0.2.0 -m "Release 0.2.0: sandbox exec, structured MCP content"

# 3. Push the tag — CI takes over from here
git push origin v0.2.0
```

The `Publish` workflow will:

1. Re-run the full test suite (safety net)
2. Build sdist + wheel with the tag's version
3. Publish to **PyPI** (final tag) or **TestPyPI** (pre-release tag)
4. Create a **GitHub Release** with auto-generated release notes

You can watch progress under the **Actions** tab of the repo.

---

## Version numbers (SemVer)

| Change kind                              | Bump      | Example         |
|------------------------------------------|-----------|-----------------|
| Bug fix, no API change                   | PATCH     | `0.1.1 → 0.1.2` |
| New feature, backward compatible         | MINOR     | `0.1.1 → 0.2.0` |
| Breaking change                          | MAJOR     | `0.1.1 → 1.0.0` |
| Pre-release testing (goes to TestPyPI)   | suffix    | `0.2.0rc1`, `0.2.0a1`, `0.2.0b2` |

---

## Pre-releases (TestPyPI staging)

To validate a build before shipping it to real users:

```bash
git tag -a v0.2.0rc1 -m "Release candidate 1 for 0.2.0"
git push origin v0.2.0rc1
```

CI will publish it to TestPyPI. Try installing it in a fresh venv:

```bash
python -m venv /tmp/rc-test && source /tmp/rc-test/bin/activate
pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ds-mcp-server==0.2.0rc1
ds-mcp-server --help
```

If it looks good, tag the final version (`v0.2.0`) to publish to real PyPI.

**Note:** You cannot publish the same version twice — PyPI rejects re-uploads.
If a release is broken, tag a new patch version (`v0.2.1`).

---

## Changelog

Release notes are auto-generated from merged PRs, grouped by label. Configuration
lives in [`.github/release.yml`](.github/release.yml).

Label your PRs with one of: `feature`, `bug`, `security`, `documentation`,
`test`, `chore`, `dependencies`, `breaking`. Unlabeled PRs land in the
"Other changes" bucket.

---

## One-time setup (already done, but documented for reference)

### PyPI Trusted Publishing

On [pypi.org](https://pypi.org/manage/account/publishing/) → *Publishing* →
*Add a new pending publisher*:

- **PyPI Project Name:** `ds-mcp-server`
- **Owner:** `ahmad-zurih`
- **Repository name:** `ds-mcp-server`
- **Workflow name:** `publish.yml`
- **Environment name:** `pypi`

Repeat on [test.pypi.org](https://test.pypi.org/manage/account/publishing/)
with **Environment name:** `testpypi`.

### GitHub Environments

In the repo → *Settings* → *Environments*, create two environments named
exactly `pypi` and `testpypi`. Optionally require manual approval on `pypi`
so no tag can go live without your click.

### Branch protection (recommended)

*Settings* → *Branches* → *Add rule* for `main`:

- ✅ Require a pull request before merging
- ✅ Require status checks to pass (select the `test` job from `Tests` workflow)
- ✅ Do not allow force pushes

---

## Troubleshooting

**"HTTPError: 400 File already exists"** — you tried to reupload a version.
Bump the number and re-tag.

**"setuptools-scm was unable to detect version"** — the workflow needs
`fetch-depth: 0` on `actions/checkout` (already configured). If you build
locally, make sure you're inside the git repo and have at least one tag.

**Version comes out as `0.0.0+unknown` locally** — you're running from a
source checkout that isn't installed (`pip install -e .`) and has no
`_version.py` yet. Run `pip install -e .` once and it will be regenerated.

**Test fails on CI but passes locally** — the CI job runs on a clean Ubuntu
image with the exact `[dev]` extras. Missing an OS-level dep? Add it to
`publish.yml`.

**I want to yank a bad release** — on PyPI you can't delete a version, but
you can [yank it](https://pypi.org/help/#yanked) (hides it from `pip install`
but keeps it available for anyone pinned to it). Do this on pypi.org under
*Manage project* → *Releases*.
