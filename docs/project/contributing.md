# Contributing

This documentation site tracks `llamatelemetry` SDK development and should stay aligned with package behavior.

## Contributing to SDK

Primary repository:

- <https://github.com/llamatelemetry/llamatelemetry>

Start with:

- `CONTRIBUTING.md` in the SDK repository
- issue tracker for bugs/feature requests

## Contributing to docs site (`llamatelemetry.github.io`)

## Local preview

```bash
zensical serve -f zensical.toml
```

## Build

```bash
zensical build -f zensical.toml --clean
```

## Documentation standards used here

- Keep examples executable and version-scoped (`v0.1.0` where needed).
- Prefer API-accurate signatures over speculative parameters.
- Call out optional dependency constraints clearly.

## GitHub Pages deployment

Deployment workflow is defined at:

- `.github/workflows/docs.yml`

It builds via `zensical build` and publishes the `site/` artifact to GitHub Pages.
