# Local development tasks.
#
#   bundle exec rake serve   # jekyll serve with livereload, restarted when the
#                            # config or the theme changes
#   bundle exec rake stop    # stop whatever is serving on the port
#   bundle exec rake build   # one production build into _site
#
# `jekyll serve` regenerates pages, posts and _data on its own, but it never
# watches _config.yml, and it ignores everything listed under `exclude`, which
# includes vendor/ and therefore the whole DataLog theme. The `serve` task
# wraps it: it watches those paths itself, re-copies the theme's assets and
# restarts Jekyll when they change.

require "rbconfig"
require "socket"

$stdout.sync = true # keep the restart messages in order with Jekyll's own output

HOST = ENV.fetch("JEKYLL_HOST", "127.0.0.1")
PORT = Integer(ENV.fetch("JEKYLL_PORT", "4000"))
THEME_DIR = File.expand_path("vendor/datalog", __dir__)
WATCHED_THEME_DIRS = %w[_layouts _includes _sass _plugins _data assets].map { |d| File.join(THEME_DIR, d) }
WINDOWS = RbConfig::CONFIG["host_os"] =~ /mswin|mingw|cygwin/

def jekyll_command(*args)
  # Run Jekyll in a Ruby process we own, so stopping it does not leave an
  # orphan behind the way killing `bundle exec` can on Windows.
  [RbConfig.ruby, "-rbundler/setup", Gem.bin_path("jekyll", "jekyll"), *args]
end

def port_open?(host, port)
  Socket.tcp(host, port, connect_timeout: 0.5) { true }
rescue StandardError
  false
end

def stop_process(pid)
  if WINDOWS
    system("taskkill", "/T", "/F", "/PID", pid.to_s, out: File::NULL, err: File::NULL)
  else
    Process.kill("TERM", pid)
  end
  Process.wait(pid)
rescue Errno::ECHILD, Errno::ESRCH
  nil
end

def sync_theme_assets
  system("python", File.join(__dir__, "scripts", "sync_theme_assets.py")) ||
    warn("theme asset sync failed; continuing with the copies already in place")
end

desc "Serve the site locally; restarts Jekyll when _config.yml or the theme changes"
task :serve do
  require "listen"

  if port_open?(HOST, PORT)
    abort "something is already listening on #{HOST}:#{PORT}; run `bundle exec rake stop` first"
  end

  extra = ENV.fetch("JEKYLL_ARGS", "--livereload").split
  start = lambda do
    sync_theme_assets
    puts "[rake serve] starting jekyll serve on http://#{HOST}:#{PORT}/"
    Process.spawn(*jekyll_command("serve", "--host", HOST, "--port", PORT.to_s, *extra))
  end

  pid = start.call
  restarting = false
  listener = Listen.to(
    __dir__, *WATCHED_THEME_DIRS,
    only: %r{(?:^|/)(?:_config\.ya?ml|vendor/datalog/(?:_layouts|_includes|_sass|_plugins|_data|assets)/.*)$},
    ignore: [%r{node_modules}, %r{/tmp/}, %r{/dist/}],
    wait_for_delay: 1
  ) do |modified, added, removed|
    changed = (modified + added + removed).map { |p| p.sub("#{__dir__}/", "") }
    next if restarting

    restarting = true
    puts "[rake serve] #{changed.first(3).join(', ')}#{changed.size > 3 ? ", ..." : ""} changed; restarting jekyll"
    stop_process(pid)
    pid = start.call
    restarting = false
  end
  listener.start

  trap("INT") do
    listener.stop
    stop_process(pid)
    exit
  end
  Process.wait(pid)
  listener.stop
end

desc "Stop the process listening on the serve port"
task :stop do
  unless port_open?(HOST, PORT)
    puts "nothing is listening on #{HOST}:#{PORT}"
    next
  end
  if WINDOWS
    lines = `netstat -ano`.lines.grep(/\s#{Regexp.escape(HOST)}:#{PORT}\s.*LISTENING/)
    pids = lines.map { |l| l.split.last.to_i }.uniq
    pids.each { |p| system("taskkill", "/T", "/F", "/PID", p.to_s, out: File::NULL, err: File::NULL) }
    puts "stopped #{pids.join(', ')}"
  else
    sh "lsof -ti tcp:#{PORT} | xargs -r kill"
  end
end

desc "Production build into _site"
task :build do
  sync_theme_assets
  sh({ "JEKYLL_ENV" => "production" }, *jekyll_command("build"))
end

task default: :build
