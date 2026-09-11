/**
 * @fileoverview Skip links for keyboard accessibility.
 * Allows users to skip to main content areas.
 * @module core/skip-links
 */

/**
 * Focuses a target element, temporarily making it focusable if needed.
 * @param {HTMLElement|null} target - The element to focus
 */
function focusTarget(target) {
  if (!target) {
    return;
  }

  const previousTabIndex = target.getAttribute("tabindex");
  if (previousTabIndex === null) {
    target.setAttribute("tabindex", "-1");
    target.dataset.skipLinkTempTabindex = "true";
  }

  target.addEventListener(
    "blur",
    () => {
      if (target.dataset.skipLinkTempTabindex) {
        target.removeAttribute("tabindex");
        delete target.dataset.skipLinkTempTabindex;
      }
    },
    { once: true }
  );

  target.focus();
}

/**
 * Initializes skip link functionality for accessibility.
 * Enhances focus management when skip links are activated.
 * @returns {void}
 */
export function initSkipLinks() {
  const skipLinks = document.querySelectorAll(".skip-link[href^='#']");
  if (!skipLinks.length) {
    return;
  }

  skipLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
      const href = link.getAttribute("href") || "";
      const targetId = href.slice(1);
      if (!targetId) {
        return;
      }
      const target = document.getElementById(targetId);
      if (!target) {
        return;
      }
      event.preventDefault();
      focusTarget(target);
    });
  });
}
