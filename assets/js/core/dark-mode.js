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
 * Reads the saved preference. Storage access throws where the browser blocks
 * it (Safari with all cookies blocked, sandboxed iframes, some privacy
 * extensions), and that exception used to stop the whole core bundle, so it
 * counts as no saved preference.
 * @returns {string|null} "dark", "light" or null
 */
function readStoredPreference() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

/**
 * Saves the preference where storage is available. Where it is blocked the
 * choice still applies to the page but is not remembered.
 * @param {string} value - "dark" or "light"
 * @returns {void}
 */
function storePreference(value) {
  try {
    localStorage.setItem(STORAGE_KEY, value);
  } catch (error) {
    // Storage is blocked: there is nowhere to remember the choice.
  }
}

/**
 * Resolves the initial dark mode preference from storage or system settings.
 * The inline script at the top of <body> in _layouts/default.html applies the
 * same rule before the first paint, so the two have to stay in step.
 * @param {MediaQueryList} prefersDarkScheme - Media query for dark scheme preference
 * @returns {boolean} True if dark mode should be enabled
 */
function resolveInitialPreference(prefersDarkScheme) {
  const storedPreference = readStoredPreference();
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
      storePreference(isDark ? "dark" : "light");
      syncThemeAttribute();
    });
  }

  prefersDarkScheme.addEventListener("change", (event) => {
    if (readStoredPreference()) {
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
