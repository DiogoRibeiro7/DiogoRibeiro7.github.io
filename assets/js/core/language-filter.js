/**
 * @fileoverview Navigate-on-change select handling.
 * @module core/language-filter
 */

const NAVIGATING_SELECTS = "[data-language-filter], [data-navigate-on-change]";

/**
 * Resolves an option value to a target that is safe to navigate to.
 *
 * Option values come from site data and front matter, so they are not visitor
 * controlled, but rejecting non-http(s) schemes keeps a stray `javascript:`
 * value from turning a dropdown into a script sink. Relative values carry no
 * scheme and pass through untouched, so existing links keep working.
 *
 * @param {unknown} value Raw option value.
 * @returns {string|null} The value to navigate to, or null when unsafe.
 */
function safeNavigationTarget(value) {
  if (typeof value !== "string") {
    return null;
  }

  // Browsers ignore control characters when parsing a scheme, so strip them
  // first; otherwise "java	script:alert(1)" would read as scheme-less here
  // and still execute once assigned. Done by code point rather than a regex,
  // which would trip no-control-regex.
  const candidate = Array.from(value)
    .filter((character) => {
      const code = character.codePointAt(0);
      return code > 0x1f && code !== 0x7f;
    })
    .join("")
    .trim();
  if (!candidate) {
    return null;
  }

  const scheme = /^[a-z][a-z0-9+.-]*:/i.exec(candidate);
  if (scheme && !/^https?:$/i.test(scheme[0])) {
    return null;
  }

  return candidate;
}

/**
 * Binds selects that navigate when their value changes.
 *
 * Replaces inline `onchange` handlers, which the site's CSP blocks because
 * script-src carries no 'unsafe-inline'.
 *
 * @returns {void}
 */
export function initLanguageFilter() {
  const selects = document.querySelectorAll(NAVIGATING_SELECTS);
  if (selects.length === 0) {
    return;
  }

  selects.forEach((select) => {
    select.addEventListener("change", (event) => {
      const target = safeNavigationTarget(event.target && event.target.value);
      if (target) {
        window.location.href = target;
      }
    });
  });
}
