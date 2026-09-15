# Post organization

Published articles live under `_posts/<primary-category>/`.

The physical folder is derived from the first `categories` entry in a post's
front matter using a lowercase underscore slug. Examples:

- `Statistics` -> `_posts/statistics/`
- `Data Science` -> `_posts/data_science/`
- `Machine Learning` -> `_posts/machine_learning/`
- `Time Series` -> `_posts/time_series/`
- `Predictive Maintenance` -> `_posts/predictive_maintenance/`

Folder placement is repository organization, not URL design. Existing URLs are
preserved with explicit `permalink` values where a move would otherwise make the
generated route depend on the directory hierarchy.

Idea lists and PhD planning notes are not published posts. They live under
`_drafts/ideas/` and `_drafts/phd/` instead of `_posts/`.

When adding a new article:

1. choose the primary category deliberately;
2. place the file in the matching category folder;
3. keep the standard `YYYY-MM-DD-title.md` filename;
4. add a new category folder when the category is genuinely useful rather than
   forcing an article into an unrelated subject;
5. preserve an established permalink when moving an existing article.

Maintenance utilities that operate on `_posts` must recurse through subject
folders.
