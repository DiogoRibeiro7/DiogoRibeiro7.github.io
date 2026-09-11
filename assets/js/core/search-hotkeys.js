/**
 * @fileoverview Search keyboard shortcuts.
 * Enables "/" and Cmd/Ctrl+K hotkeys to focus search.
 * @module core/search-hotkeys
 */

/**
 * Focuses the global search input or navigates to search page.
 * @param {HTMLElement} siteSearchForm - The search form element
 */
function focusGlobalSearch(siteSearchForm) {
  if (!siteSearchForm) {
    return;
  }
  const siteSearchInput = siteSearchForm.querySelector("input[type='search']");
  const searchDestination = siteSearchForm.getAttribute("action");

  if (siteSearchInput) {
    siteSearchInput.focus();
    siteSearchInput.select();
  } else if (searchDestination) {
    window.location.href = searchDestination;
  }
}

function shouldIgnoreHotkey(activeElement) {
  if (!activeElement) {
    return false;
  }
  if (activeElement.isContentEditable) {
    return true;
  }
  const tag = activeElement.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

/**
 * Initializes search keyboard shortcuts (/ and Cmd/Ctrl+K).
 * @returns {void}
 */
export function initSearchHotkeys() {
  const siteSearchForm = document.querySelector(".site-search");
  if (!siteSearchForm) {
    return;
  }

  document.addEventListener("keydown", (event) => {
    if (shouldIgnoreHotkey(document.activeElement)) {
      return;
    }

    if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      focusGlobalSearch(siteSearchForm);
    }

    if ((event.key === "k" || event.key === "K") && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      focusGlobalSearch(siteSearchForm);
    }
  });
}
