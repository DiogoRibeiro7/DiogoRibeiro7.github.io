import { initDarkModeToggle } from "./core/dark-mode.js";
import { initGitHubCards } from "./core/github-cards.js";
import { initLanguageFilter } from "./core/language-filter.js";
import { initNavigation } from "./core/navigation.js";
import { initScrollProgress } from "./core/scroll-progress.js";
import { initSearchHotkeys } from "./core/search-hotkeys.js";
import { initSkipLinks } from "./core/skip-links.js";

export function initializeCore() {
  initSkipLinks();
  initNavigation();
  initDarkModeToggle();
  initLanguageFilter();
  initScrollProgress();
  initSearchHotkeys();
  initGitHubCards();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeCore);
} else {
  initializeCore();
}
