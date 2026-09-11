source "https://rubygems.org"

# The DataLog theme lives in the vendor/datalog git submodule. Depending on
# its gemspec installs every gem its layouts and plugins need. Jekyll reads
# the theme's directories through layouts_dir, includes_dir, plugins_dir and
# sass_dir in _config.yml rather than through the gem, because Jekyll does
# not load a theme gem's _plugins and DataLog's layouts depend on them.
gem "datalog-theme", path: "vendor/datalog"

gem "jekyll", "~> 4.3"
gem "loofah", "~> 2.25" # used by the theme's notebook and HTML sanitiser plugins
gem "webrick"
gem "rake"

group :development do
  gem "tzinfo-data" # IANA timezone database on Windows
  # Declared through `platforms` rather than a Ruby conditional so the lockfile
  # lists the same dependencies on every platform; bundler in frozen mode on
  # Linux CI otherwise reports wdm as deleted from the Gemfile.
  gem "wdm", ">= 0.1.0", platforms: [:mingw, :x64_mingw, :mswin]
end
