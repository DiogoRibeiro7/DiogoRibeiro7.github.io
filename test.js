const fs = require('fs');
const assert = require('assert');

// Ensure README exists and is not empty
assert(fs.existsSync('README.md'), 'README.md should exist');
const readmeContent = fs.readFileSync('README.md', 'utf8');
assert(readmeContent.trim().length > 0, 'README.md should not be empty');

console.log('All tests passed');
