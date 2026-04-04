# llamatelemetry docs audit and fix plan

## Highest-priority fixes applied in this patch

1. **Fixed docs edit/view repository target**
   - The site configuration previously pointed edit actions at the SDK repo.
   - Because the published docs source lives in `llamatelemetry.github.io`, edit/view actions should point there.

2. **Rewrote the homepage to reduce drift**
   - The original homepage mixed strong marketing claims with very broad scope.
   - The new version keeps the main value proposition but is more tightly aligned with the reviewed SDK snapshot.

3. **Corrected repo-size statements**
   - Older docs said the package had approximately 40 Python files.
   - The uploaded SDK snapshot reviewed here contains 58 Python files and 7 C++/CUDA files.

4. **Removed an incorrect code example variable**
   - `print("llamatelemetry version:", llamatelemetry.__version__)` used the wrong module name after importing `llamatelemetry as lt`.
   - It is now `lt.__version__`.

5. **Made the target environment more consistent**
   - The docs previously mixed several GPU support statements.
   - The updated pages now present Linux + CUDA + NVIDIA GPU, especially Tesla T4 / Kaggle dual T4, as the clearest validated path.

## Next fixes I recommend after this patch

1. Review all pages for support-scope consistency:
   - avoid mixing “compute capability >= 6.1”, “>= 7.0”, and “>= 7.5” without context
   - separate “may work” from “documented and validated”

2. Separate **core docs** from **advanced integrations**:
   - keep Graphistry, Louie, Unsloth, and W&B as advanced or optional tracks
   - keep homepage and getting-started focused on inference + telemetry + Kaggle

3. Add one page named `docs/project/status.md` that clearly states:
   - stable core surfaces
   - experimental modules
   - target environments
   - known limitations

4. Make installation language more precise:
   - document package extras that actually exist in `pyproject.toml`
   - move unrelated optional tools into a separate “ecosystem integrations” section

5. Add one canonical architecture diagram and reuse it everywhere.

## Build verification

This patched docs repo builds successfully with:

```bash
zensical build -f zensical.toml --clean
```

It generates `site/sitemap.xml` during the local build.
