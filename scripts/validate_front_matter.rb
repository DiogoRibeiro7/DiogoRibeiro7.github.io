#!/usr/bin/env ruby
# frozen_string_literal: true

require "date"
require "yaml"

POSTS = File.expand_path("../_posts", __dir__)

errors = []

Dir.glob(File.join(POSTS, "**", "*.md")).sort.each do |path|
  text = File.read(path, encoding: "UTF-8")
  next unless text.start_with?("---\n")

  closing = text.index("\n---\n", 4)
  unless closing
    errors << "#{path}: missing closing front-matter delimiter"
    next
  end

  yaml = text[4...closing]

  begin
    data = YAML.safe_load(
      yaml,
      permitted_classes: [Date, Time],
      aliases: true,
      filename: path
    )

    unless data.is_a?(Hash)
      errors << "#{path}: front matter must parse to a mapping"
    end
  rescue Psych::SyntaxError => e
    errors << "#{path}: #{e.message}"
  end
end

if errors.any?
  warn "Invalid post front matter:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "Validated front matter for all published posts."
