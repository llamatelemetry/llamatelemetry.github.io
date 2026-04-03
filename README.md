# llamatelemetry.github.io

Documentation website source for `llamatelemetry v0.1.1`, built with Zensical and deployed through GitHub Pages.

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

## Google Search Console scripts

Repo-local helper scripts live in `scripts/`:

- `scripts/gsc-auth.sh` - initialize Google ADC auth with `gcloud`
- `scripts/gsc-list-sites.sh` - list Search Console properties
- `scripts/gsc-submit-sitemap.sh` - submit `https://llamatelemetry.github.io/sitemap.xml`
- `scripts/gsc-list-sitemaps.sh` - list submitted sitemaps
- `scripts/gsc-inspect-url.sh <url>` - inspect a single URL
- `scripts/gsc-inspect-defaults.sh` - inspect the homepage, get-started, and reference
- `scripts/gsc-run-all.sh` - run the full flow end to end

They use either:

- `ACCESS_TOKEN` if already exported, or
- `gcloud auth application-default print-access-token`
