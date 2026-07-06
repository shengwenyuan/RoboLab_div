# Asset Management

RoboLab keeps the source repository and materialized assets separate.

## Source contract

The repository may still contain Git LFS tracked assets under `assets/` for compatibility with upstream RoboLab. Do not replace tracked asset directories with committed symlinks to machine-local storage.

Runtime code resolves assets through `robolab.constants.ASSET_DIR`. By default this points at `<repo>/assets`. Set `ROBOLAB_ASSET_DIR` to use an externally materialized asset tree instead:

```bash
export ROBOLAB_ASSET_DIR=/path/to/RoboLabAssets/current/assets
```

The external tree should preserve the same layout as the repository assets directory, for example `objects/`, `scenes/`, `robots/`, `backgrounds/`, `fixtures/`, and `materials/`.

## Local worktree policy

On machines where assets live on fast shared storage, keep those files outside the repo checkout and point `ROBOLAB_ASSET_DIR` at that location. If symlinks are used for local convenience, treat them as uncommitted machine state.

To hide tracked asset deletions caused by local externalization, mark tracked asset files as skip-worktree in the local index:

```bash
git ls-files -z assets | xargs -0 git update-index --skip-worktree --
```

If local symlink entry points still show up as untracked files, ignore those entries in `.git/info/exclude` rather than in the committed `.gitignore`.

To make Git watch repository assets again:

```bash
git ls-files -z assets | xargs -0 git update-index --no-skip-worktree --
```

This is local Git metadata and is not committed.
