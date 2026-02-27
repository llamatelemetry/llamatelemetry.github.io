# llamatelemetry.github.io

Documentation website source for `llamatelemetry v0.1.0`, built with Zensical and deployed through GitHub Pages.

## Repository purpose

This repository contains only the docs website project for:

- SDK docs
- API reference
- Notebook curriculum docs
- project architecture/guides

## Local development

## Prerequisite

- Python environment with `zensical` installed.

## Serve locally

```bash
zensical serve -f zensical.toml
```

## Build static site

```bash
zensical build -f zensical.toml --clean
```

Generated output is written to `site/`.

## Structure

- `zensical.toml` - site configuration and navigation
- `docs/` - markdown source pages
- `.github/workflows/docs.yml` - GitHub Pages deployment workflow

## Deployment

Pushes to `main` or `master` trigger the docs workflow that builds the site and deploys to GitHub Pages.
