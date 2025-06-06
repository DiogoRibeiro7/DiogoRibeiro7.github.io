const fs = require("fs");
const pkg = require("./package.json");
const filename = "assets/js/main.min.js";
const script = fs.readFileSync(filename, 'utf8');
const padStart = str => ("0" + str).slice(-2);
const dateObj = new Date();
const date = `${dateObj.getFullYear()}-${padStart(
  dateObj.getMonth() + 1
)}-${padStart(dateObj.getDate())}`;
const banner = `/*!
 * Minimal Mistakes Jekyll Theme ${pkg.version} by ${pkg.author}
 * Copyright 2013-${dateObj.getFullYear()} Michael Rose - mademistakes.com | @mmistakes
 * Licensed under ${pkg.license}
 */
`;

if (!script.startsWith('/*!')) {
  fs.writeFileSync(filename, banner + script);
}
