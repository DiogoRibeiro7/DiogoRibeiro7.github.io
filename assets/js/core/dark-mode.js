/**
 * @fileoverview Dark mode toggle functionality.
 * Persists user preference and respects system color scheme.
 * @module core/dark-mode
 */

/** @constant {string} LocalStorage key for color mode preference */
const STORAGE_KEY = "datalog-color-mode";
/** @constant {string} Class applied to body for dark mode */
const DARK_CLASS = "dark-mode";

/**
 * Resolves the initial dark mode preference from storage or system settings.
 * @param {MediaQueryList} prefersDarkScheme - Media query for dark scheme preference
 * @returns {boolean} True if dark mode should be enabled
 */
function resolveInitialPreference(prefersDarkScheme) {
  const storedPreference = localStorage.getItem(STORAGE_KEY);
  if (storedPreference === "dark") {
    return true;
  }
  if (storedPreference === "light") {
    return false;
  }
  return prefersDarkScheme.matches;
}

/**
 * Initializes the dark mode toggle button and system preference listener.
 * Reads initial preference from localStorage or system settings.
 * @returns {void}
 */
export function initDarkModeToggle() {
  const toggleButton = document.querySelector("[data-toggle-dark-mode]");
  const body = document.body;
  const prefersDarkScheme = window.matchMedia("(prefers-color-scheme: dark)");

  if (resolveInitialPreference(prefersDarkScheme)) {
    body.classList.add(DARK_CLASS);
  }

  /** Mirrors the effective theme on <body data-theme> for CSS hooks and tests. */
  const syncThemeAttribute = () => {
    const theme = body.classList.contains(DARK_CLASS) ? "dark" : "light";
    body.dataset.theme = theme;
    document.documentElement.dataset.theme = theme;
  };

  const syncButton = () => {
    syncThemeAttribute();
    if (!toggleButton) {
      return;
    }
    toggleButton.setAttribute("aria-pressed", body.classList.contains(DARK_CLASS));
  };

  syncButton();

  if (toggleButton) {
    toggleButton.addEventListener("click", () => {
      const isDark = body.classList.toggle(DARK_CLASS);
      toggleButton.setAttribute("aria-pressed", String(isDark));
      localStorage.setItem(STORAGE_KEY, isDark ? "dark" : "light");
      syncThemeAttribute();
    });
  }

  prefersDarkScheme.addEventListener("change", (event) => {
    if (localStorage.getItem(STORAGE_KEY)) {
      return;
    }
    if (event.matches) {
      body.classList.add(DARK_CLASS);
    } else {
      body.classList.remove(DARK_CLASS);
    }
    syncButton();
  });
}
