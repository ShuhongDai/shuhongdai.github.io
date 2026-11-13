# shuhongdai.github.io

Source of <https://shuhongdai.github.io>, built with [Jekyll](https://jekyllrb.com/) and the
[al-folio](https://github.com/alshedivat/al-folio) theme (v1.x).

Pushing to `main` builds the site and publishes it to the `gh-pages` branch, which GitHub Pages serves.

## How the site is put together

Since al-folio v1 the theme is no longer copied into the repository. Layouts, includes, styles and
scripts come from versioned Ruby gems (`al_folio_core` and the other `al_*` gems pinned in
[Gemfile](Gemfile)); this repository holds only the content, the configuration, and a small set of
site-specific overrides.

| What                         | Where                                                         |
| ---------------------------- | ------------------------------------------------------------- |
| Site settings                | [\_config.yml](_config.yml)                                   |
| About page                   | [\_pages/about.md](_pages/about.md)                           |
| Blog posts                   | [\_posts/](_posts/)                                           |
| Publications                 | [\_bibliography/papers.bib](_bibliography/papers.bib)         |
| Videos                       | [\_data/videos.yml](_data/videos.yml)                         |
| News                         | [\_news/](_news/)                                             |
| Projects (currently hidden)  | [\_projects/](_projects/)                                     |
| Social links                 | [\_data/socials.yml](_data/socials.yml)                       |
| Images, PDFs, videos         | [assets/](assets/)                                            |
| Site-specific styles         | [\_sass/\_custom.scss](_sass/_custom.scss)                    |
| Visitor-log Cloudflare Worker | [cloudflare/visitor-log-worker/](cloudflare/visitor-log-worker/) |

Folders inside `_posts/` whose names start with an underscore (`_drafts`, `_hiden_blogs`,
`_old_drafts`, `_Texas`) are skipped by Jekyll: posts there are kept in the repository but not
published. Move a file to the top level of `_posts/` to publish it.

## Preview locally

With Docker Desktop running:

```bash
docker compose pull
docker compose up
```

The site is served at <http://localhost:8080> and rebuilds when files change.

To reproduce the production build that CI runs:

```bash
docker compose run --rm -e JEKYLL_ENV=production jekyll bundle exec jekyll build
npx --yes purgecss -c purgecss.config.js
```

## Site-specific customizations

Everything that differs from stock al-folio is listed here, so it can be reviewed after a theme update.

**Styles** — [\_sass/\_custom.scss](_sass/_custom.scss), loaded last by a local copy of the theme's
[assets/css/main.scss](assets/css/main.scss): the light-mode theme colour, the Roboto Slab site name and
about title, profile layout, publication title links, the project status pill, the videos grid, and the
code font.

**Template overrides** — local copies of theme files with a small change each, marked with
`Site override` comments:

| File                                                    | Change                                                                                   |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| [\_layouts/bib.liquid](_layouts/bib.liquid)             | Title links to the entry's `html` field; first names as initials; no button row          |
| [\_layouts/about.liquid](_layouts/about.liquid)         | `profile.width`, title style, "selected videos" section                                  |
| [\_layouts/archive.liquid](_layouts/archive.liquid)     | Categories are called "columns"                                                          |
| [\_includes/header.liquid](_includes/header.liquid)     | A page can set `nav_title` for a navbar label that differs from its title                |
| [\_includes/footer.liquid](_includes/footer.liquid)     | Loads the visitor-logging snippet                                                        |
| [\_includes/related_posts.liquid](_includes/related_posts.liquid) | Heading folded into the sentence; list limited to `related_blog_posts.max_related` |
| [\_includes/projects.liquid](_includes/projects.liquid), [projects_horizontal.liquid](_includes/projects_horizontal.liquid) | Project `status` pill, `img_no_responsive` |

**Site-owned templates** — [\_includes/selected_videos.liquid](_includes/selected_videos.liquid),
[\_includes/project_hero.liquid](_includes/project_hero.liquid),
[\_includes/scripts/visitor_logging.liquid](_includes/scripts/visitor_logging.liquid) and
[\_pages/videos.md](_pages/videos.md).

The overrides are registered in [.al-folio-overrides.yml](.al-folio-overrides.yml). After editing one,
re-register it:

```bash
docker compose run --rm jekyll bundle exec al-folio upgrade overrides accept --all
```

### Publishing the projects page

Projects are written but hidden. To publish them:

1. In `_config.yml`, set `collections.projects.output: true` and remove `_projects/` from `exclude:`.
2. In `_pages/projects.md`, remove `published: false` and set `nav: true`.

## Updating the theme

The gem versions in [Gemfile](Gemfile) are pinned exactly, so `bundle update` alone changes nothing.
To move to a newer al-folio release:

1. Copy the new version pins from the upstream [Gemfile](https://github.com/alshedivat/al-folio/blob/main/Gemfile)
   and keep the `plugins:` list in `_config.yml` in step with it.
2. Refresh the lock file: `docker compose run --rm jekyll bundle lock`.
3. Check the result:

   ```bash
   docker compose run --rm jekyll bundle exec al-folio upgrade audit
   docker compose run --rm jekyll bundle exec al-folio upgrade overrides audit
   ```

   The first reports anything the new version no longer supports. The second lists each override
   whose upstream file changed; compare with `al-folio upgrade overrides diff PATH`, carry the
   upstream change into the local copy, then `overrides accept` it.

4. Preview locally before pushing.

Upstream's upgrade notes are in [docs/INSTALL.md](https://github.com/alshedivat/al-folio/blob/main/docs/INSTALL.md).

## Deployment

- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) builds on every push to `main` and
  publishes `_site` to `gh-pages`. Pull requests are built but never published.
- GitHub Pages must stay set to **Deploy from a branch → `gh-pages`**. GitHub's built-in Jekyll cannot
  load the theme gems, so serving from `main` does not work.
- [.github/workflows/upgrade-check.yml](.github/workflows/upgrade-check.yml) runs the upgrade audit.

Line endings are normalised to LF by [.gitattributes](.gitattributes), so committing from Windows and
macOS gives the same result.

## Known limitations

- TikZ diagrams (`tikzjax: true`) do not render: the theme's content security policy blocks the web
  worker TikZJax starts. No published post uses TikZ; two unpublished drafts do.
- The search palette lists pages by `title`, not `nav_title`.

## The previous version of the site

Until November 2026 this site ran on al-folio v0.12.1 with the whole theme vendored in the repository.
That version is kept at:

- tag `legacy-v0.12.1-source` and branch `legacy/v0.12.1` — the source
- tag `legacy-v0.12.1-site` — the built site as last deployed

To put the old built site back online without rebuilding anything:

```bash
git push --force origin 'legacy-v0.12.1-site^{commit}:refs/heads/gh-pages'
```

The next push to `main` publishes the current source again.

## License

al-folio is released under the MIT License; its notice is kept in [LICENSE](LICENSE).
