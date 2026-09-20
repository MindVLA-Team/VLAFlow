# Release Preparation

This is a maintainer checklist for merging the code into `MindVLA-Team/VLAFlow`.
It records a technical/documentation review, not legal clearance. No remote changes have
been made as part of this preparation.

## GitHub Pages: Verified Configuration

A read-only GitHub API query on 2026-09-20 returned:

```json
{
  "build_type": "legacy",
  "source": {"branch": "main", "path": "/"},
  "status": "built",
  "html_url": "https://mindvla-team.github.io/VLAFlow/"
}
```

The existing site is deployed from the **root of `main`**, not a separate Pages branch.
Adding Python code does not inherently remove the site. Deleting or relocating its entry
point or resources without updating the publishing source breaks the existing site.

### Recommended for This Release: Keep the Root Site

The release candidate preserves the following paths from the existing website checkout
(commit `4c8d9f3`). The HTML, `.nojekyll`, and figures are unchanged; the report is replaced
with the author-supplied arXiv v2 PDF:

```text
index.html
.nojekyll
assets/
report/
```

Keep these paths in the final `main` tree. No Pages settings change is required, and relative
image/report links continue to resolve. Retain the original Apache license and NOTICE for
these materials, now stored under `LICENSES/Project-materials-*`.

This avoids a website migration during the code release. The root publishing source may
also expose other repository files as static content; do not place private material in it.

### Alternative: Separate Website Branch

For independent website maintenance, create a `gh-pages` branch from the current website
commit and publish that branch **before** removing site files from `main`:

1. Confirm the latest remote website commit and preserve it on `gh-pages`.
2. Push that branch without rewriting `main`.
3. In **Settings > Pages > Build and deployment**, select **Deploy from a branch**,
   branch **gh-pages**, folder **/(root)**, then save.
4. Wait for deployment and verify the homepage, figures, and PDF.
5. Only then remove website-only files from `main`, if desired. Keep any figures/report
   referenced by the code README, or update those links separately.

The repository name remains `VLAFlow`, so changing the source branch does not require a
new project-site URL. Another supported option is `main:/docs`; this checkout already uses
`docs/` for software documentation, so that option needs deliberate restructuring.
A custom GitHub Actions workflow can publish a dedicated site directory as an alternative.

Official configuration reference:
<https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site>

## License Review: Blocking Decisions

The existing website repository uses **Apache-2.0**. The code candidate's root `LICENSE`
contains the StarVLA MIT text **plus additional commit-history wording**. Multiple source
headers also identify MIT, while NVIDIA-derived files identify Apache-2.0.

- [ ] Confirm the intended license for Li Auto's VLAFlow contributions with the authorized
  copyright holder. Apache-2.0 would align with the existing public project, but it must not
  be applied by silently replacing all upstream notices.
- [ ] Resolve StarVLA's additional wording, especially "keep at least the two latest upstream
  StarVLA commits as separate." Determine the applicable upstream revision and acceptable
  history/attribution arrangement with the upstream maintainers or legal reviewer before
  a snapshot import or squash. The wording is preserved here; no conclusion about its legal
  scope is asserted.
- [ ] Review the code history before importing it. The current checkout contains internal
  development history, not only the release snapshot. Do not merge or push that history
  wholesale without checking for private code, credentials, data, and internal paths.
- [ ] After the above decisions, reconcile `LICENSE`, source notices for your contributions,
  README, `CITATION.cff`, and package metadata. The misleading blanket `license: MIT` in
  `CITATION.cff` has been replaced with a license URL pending that decision.
  The packaging check also emitted deprecation warnings for the existing table-form
  `project.license` and `tool.setuptools.license-files`. Once the license is settled, migrate
  to an accurate SPDX expression and `project.license-files` with a compatible build-backend
  minimum version; do not label the modified upstream text as standard MIT to silence warnings.
- [ ] Confirm remaining provenance, including the Prismatic origin noted by the logger and
  any intermediate upstream copies. Adding identified license texts is not a complete audit.

Already addressed in this candidate:

- Added missing OpenVLA, robosuite, and msgpack-numpy license texts based on the explicit
  source attributions and upstream license files.
- Expanded the component-level third-party notice inventory.
- Preserved the website materials' original Apache license and NOTICE unchanged.
- Added root `NOTICE` to the package's `license-files` list alongside third-party licenses.

## Merge and Verification Checklist

The two local checkouts have different remotes: the website checkout targets public GitHub;
the code checkout targets internal GitLab. A plain `git push` from the code checkout does
**not** publish to the intended GitHub repository.

1. Resolve the licensing/history decisions above before choosing an import strategy.
2. Use a release branch based on the latest public `origin/main` and review the code import
   there. Preserve public history; do not force-push or replace the target `.git` directory.
3. Keep the root site files and `.nojekyll`, or complete the separate-branch migration first.
4. Inspect the final diff and tracked files for secrets, internal paths, weights, datasets,
   temporary output, and unintended deletions. Do not assume `.gitignore` cleans old history.
5. Validate installation/training/evaluation on suitable hardware. A launcher dry-run only
   checks configuration assembly; it does not validate training or reproduce paper results.
6. Review the final README links, license metadata, and distributed license files.
7. Merge the reviewed release into `main` through the normal repository review process.
   Once training code is publicly available, mark only the first README TODO as complete.
8. Verify the Pages deployment succeeds and inspect the public homepage, images, PDF,
   README, and code tree. Keep weight/demo TODOs open until those artifacts are available.

## Local Verification on 2026-09-20

- All 40 local link references in the README, third-party notices, software docs, and site
  HTML resolve to existing paths.
- All eight copied site/resource files match the original checkout byte-for-byte, as do
  the preserved project license and NOTICE.
- `CITATION.cff` validates against the official CFF 1.2.0 schema; `pyproject.toml` parses.
- README Bash snippets pass syntax checking; all 12 training launchers pass `--dry-run`.
- A wheel builds successfully without installing runtime dependencies; all ten license/notice
  files appear in the wheel, match the source bytes, and are listed in its metadata.
- `git diff --check` passes. The original website checkout is unchanged.

The checks above describe the initial preparation before the report replacement.
These checks used a temporary macOS/Python 3.14 tooling environment, not the documented
Linux/Python 3.10 training environment. No GPU training, simulator evaluation, full runtime
installation, remote deployment, or exhaustive historical secret scan was performed.

## Report Update and Public Release Branch

The report was updated on 2026-09-20 to the author-supplied **arXiv:2607.01586v2**
(August 4, 2026), replacing the previous 28-page PDF with a 38-page PDF at the same path.
All 38 pages render, and the first and last pages were visually inspected. The replacement
retains its original PDF bytes and license metadata; see `THIRD_PARTY_NOTICES.md`.

The public release candidate is prepared on local branch `release/vlaflow-code` in the
GitHub checkout, based on freshly fetched `origin/main` (`4c8d9f3`). Only working-tree files
are imported; no internal GitLab commit ancestry is attached. The source checkout's tracked
but ignored `CLAUDE.md` is intentionally excluded because it contains internal development
instructions. No `.git` directory, agent configuration, cache, model weights, or dataset is
part of the import.

This is a local review candidate, not a license-cleared publication. The license and
upstream-history decisions above must be resolved before pushing the code publicly.

Verification in the public checkout after importing the release snapshot:

- 155 imported files match the candidate source; `CLAUDE.md` is excluded.
- Both copies of the new PDF match the supplied file byte-for-byte. SHA-256:
  `532a6c30f2c7b119a03dd6d2be1a309698e70633b871958e2b5931e23595a75a`.
- All seven website entry/resource files other than the report match `origin/main`.
- All 41 local README/documentation/site link references resolve.
- 75 Python files pass Python 3.10 syntax parsing; 26 shell scripts pass `bash -n`.
- All 12 training launcher dry-runs pass in the public checkout; CFF and TOML metadata
  validation passes. No GPU training or simulator evaluation was performed.
- A targeted working-tree scan for common credential formats, private keys, internal
  infrastructure hosts, and private home paths found no matches. This is not an exhaustive
  secret scan or a review of the original internal Git history.
- The README retains the user's checked training-code TODO; weight and demo TODOs remain
  pending. This checkbox is prepared for publication, not evidence that a push has occurred.
