import { initCopyButtons } from "./core/copy-buttons.js";
import { initDarkModeToggle } from "./core/dark-mode.js";
import { initGitHubCards } from "./core/github-cards.js";
import { initLanguageFilter } from "./core/language-filter.js";
import { initNavigation } from "./core/navigation.js";
import { initScrollProgress } from "./core/scroll-progress.js";
import { initSearchHotkeys } from "./core/search-hotkeys.js";
import { initSkipLinks } from "./core/skip-links.js";

const CORE_INITIALIZERS = [
  initSkipLinks,
  initNavigation,
  initDarkModeToggle,
  initLanguageFilter,
  initScrollProgress,
  initSearchHotkeys,
  initGitHubCards,
  initCopyButtons
];

/**
 * Runs each core initializer on its own, so one that throws is logged and the
 * others still run. This module is evaluated while the loader imports the core
 * bundle, and an exception here rejected that import, which kept every feature
 * bundle on the page from loading.
 * @param {Function[]} [initializers] - Initializers to run, in order
 * @returns {void}
 */
export function initializeCore(initializers = CORE_INITIALIZERS) {
  initializers.forEach((initialize) => {
    try {
      initialize();
    } catch (error) {
      console.error(`${initialize.name || "Core initializer"} failed`, error);
    }
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => initializeCore());
} else {
  initializeCore();
}
