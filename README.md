# [Minimal Mistakes Jekyll theme](https://mmistakes.github.io/minimal-mistakes/)

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/DiogoRibeiro7/DiogoRibeiro7.github.io/master/LICENSE)
[![Hosted with GH Pages](https://img.shields.io/badge/Hosted_with-GitHub_Pages-blue?logo=github&logoColor=white)](https://pages.github.com/)
[![Made with GH Actions](https://img.shields.io/badge/CI-GitHub_Actions-blue?logo=github-actions&logoColor=white)](https://github.com/features/actions)

[![Jekyll](https://img.shields.io/badge/jekyll-%3E%3D%204.3-blue.svg)](https://jekyllrb.com/)

[![Ruby Version](https://img.shields.io/badge/ruby-3.1-blue)](https://www.ruby-lang.org)
[![Ruby gem](https://img.shields.io/gem/v/minimal-mistakes-jekyll.svg)](https://rubygems.org/gems/minimal-mistakes-jekyll)


Minimal Mistakes is a flexible two-column Jekyll theme, perfect for building personal sites, blogs, and portfolios. As the name implies, styling is purposely minimalistic to be enhanced and customized by you :smile:.

## Setup

This repository contains a few helper scripts for processing Markdown posts.
Install the Python dependencies listed in `requirements.txt` with:

```bash
pip install -r requirements.txt
```

To work with the JavaScript that powers the theme you'll also need Node
dependencies. Install them with:

```bash
npm install
```

This project uses **npm** for managing JavaScript dependencies and tracks
exact versions in `package-lock.json`.

Bundled JavaScript is compiled from the source files in `assets/js/`. Run the
following to create `main.min.js` (minified with a banner) or watch for changes:

```bash
npm run build:js   # minify and add banner
npm run watch:js   # optional: automatically rebuild on changes
```

## CSS linting

Lint all SCSS files with [Stylelint](https://stylelint.io/):

```bash
npm run lint:css
```

## Local development

Install Ruby gems specified in the `Gemfile` with:

```bash
bundle install
```

Serve the site locally with:

```bash
bundle exec jekyll serve
```


## Running tests

Install the Python dependencies and execute:

```bash
pytest
```
GitHub Actions already runs these commands automatically during deployments.

# ToDo

Have a consistency in the font and font sizes (ideally you want to use 2 fonts. One for the header/subtitle and one for the text. You can use this kind of website https://fontjoy.com/ which allow you to pair fonts).

Choose a few main colours for your site (I would suggest black/white/grey but not in solid. You can also use this kind of site: https://coolors.co/palettes/popular/2a4849).

Reduce then size of the homepage top image (ideally you want your first articles to be visible on load and not hidden below the fold).

Restyle your links (ideally the link should be back with no underline and you add a css style on hover)

Center pagination

Restyle your article detail page breadcrumbs. You want them to be less visible (I would suggest a light grey colour here)

Right now at the top of the detail page, you have your site breadcrumbs, a title then another title and the font sizes are a bit off and it is hard to understand the role of the second title. I would reorganise this to provide a better understanding to the reader

On the detail page, I would suggest you put the `You may also enjoy` straight at the end of the article. Right now it is after comments and you can lose engagement on your site.

I would suggest you remove your description from the detail page. I think having it on the home page is enough. You can have a smaller introduction if needed with a read more button or link that will take the reader to a full page description of yourself and your skillset. That will allow you to tell more about yourself and why you do what you do

I will create card article with a hover animation (add some shape and background colour and ideally a header image for the card. The graphs you show me last week for example.)
