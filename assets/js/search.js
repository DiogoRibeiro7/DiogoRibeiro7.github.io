import { initializeSearchApp } from "./search/app.js";

const DEBOUNCE_DELAY = 300;
const RESULT_ID_PREFIX = "search-result-";
const MIN_LOADING_VISIBLE_MS = 500;

function debounce(callback, delay = DEBOUNCE_DELAY) {
  let timerId;
  const debounced = (...args) => {
    if (timerId) {
      window.clearTimeout(timerId);
    }
    timerId = window.setTimeout(() => {
      timerId = undefined;
      callback.apply(null, args);
    }, delay);
  };

  debounced.cancel = () => {
    if (timerId) {
      window.clearTimeout(timerId);
      timerId = undefined;
    }
  };

  return debounced;
}

function updateLiveRegion(region, message) {
  if (!region) {
    return;
  }
  const text = String(message || "");
  if (region.textContent === text) {
    region.textContent = `${text}\u00A0`;
  } else {
    region.textContent = text;
  }
  if (text.trim()) {
    region.removeAttribute("hidden");
  } else {
    region.setAttribute("hidden", "");
  }
}

function setupSearchEnhancements() {
  const app = document.querySelector("[data-search-app]");
  if (!app) {
    return;
  }

  const form = app.querySelector("[data-search-form]");
  const input = app.querySelector("[data-search-input]");
  const resultsList = app.querySelector("[data-search-results]");
  const emptyState = app.querySelector("[data-search-empty]");
  const loadingIndicator = app.querySelector("[data-search-loading]");
  const resultsRegion = app.querySelector("[data-search-region]");
  const liveCount = app.querySelector('[data-search-live="count"]');
  const liveSelection = app.querySelector('[data-search-live="selection"]');
  const liveStatus = app.querySelector('[data-search-live="status"]');
  const filtersContainer = app.querySelector("[data-search-filters]");
  const tagContainer = app.querySelector("[data-filter-tags]");

  if (!input || !resultsList) {
    return;
  }

  let activeIndex = -1;
  let resultIdCounter = 0;
  let resultItems = [];
  let pendingSearch = false;
  let filterControls = [];
  let loadingStartedAt = 0;
  let loadingHideTimer = null;

  const debouncedSubmit = debounce(() => {
    if (!form) {
      return;
    }
    pendingSearch = true;
    showLoading();
    updateLiveRegion(liveStatus, "Loading results…");
    form.dispatchEvent(new Event("submit", { cancelable: true }));
  });

  const observer = new MutationObserver(() => {
    refreshResultItems();
  });
  observer.observe(resultsList, { childList: true });

  refreshResultItems();
  refreshFilterControls();

  if (tagContainer) {
    const tagObserver = new MutationObserver(() => {
      refreshFilterControls();
    });
    tagObserver.observe(tagContainer, { childList: true });
  }

  input.addEventListener("input", () => {
    clearActiveResult(false);
    debouncedSubmit();
  });

  input.addEventListener("keydown", (event) => {
    if (handleInputNavigation(event)) {
      return;
    }
    if (event.key === "Tab" && filterControls.length > 0) {
      event.preventDefault();
      const nextIndex = event.shiftKey ? filterControls.length - 1 : 0;
      filterControls[nextIndex].focus();
    }
  });

  function handleFilterKeydown(event) {
    if (event.key !== "Tab" || filterControls.length === 0) {
      return;
    }
    event.preventDefault();
    const currentIndex = filterControls.indexOf(event.currentTarget);
    if (currentIndex === -1) {
      return;
    }
    const direction = event.shiftKey ? -1 : 1;
    const nextIndex = (currentIndex + direction + filterControls.length) % filterControls.length;
    filterControls[nextIndex].focus();
  }

  function refreshFilterControls() {
    if (!filtersContainer) {
      return;
    }
    const selects = Array.from(filtersContainer.querySelectorAll("select"));
    const tags = tagContainer ? Array.from(tagContainer.querySelectorAll("[data-filter-tag]")) : [];
    filterControls = [...selects, ...tags];
    filterControls.forEach((control) => {
      control.removeEventListener("keydown", handleFilterKeydown);
      control.addEventListener("keydown", handleFilterKeydown);
    });
  }

  function handleInputNavigation(event) {
    if (!resultItems.length && ["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) {
      return false;
    }
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        moveActive(1);
        return true;
      case "ArrowUp":
        event.preventDefault();
        moveActive(-1);
        return true;
      case "Enter":
        if (activeIndex >= 0) {
          event.preventDefault();
          openActiveResult();
          return true;
        }
        return false;
      case "Escape":
        event.preventDefault();
        clearSearch();
        return true;
      default:
        return false;
    }
  }

  function moveActive(delta) {
    if (!resultItems.length) {
      return;
    }
    const nextIndex = activeIndex === -1 ? (delta > 0 ? 0 : resultItems.length - 1) : (activeIndex + delta + resultItems.length) % resultItems.length;
    setActiveResult(nextIndex);
  }

  function setActiveResult(index) {
    if (index < -1 || index >= resultItems.length) {
      return;
    }
    if (activeIndex === index) {
      return;
    }
    if (activeIndex >= 0 && resultItems[activeIndex]) {
      resultItems[activeIndex].classList.remove("is-active");
    }
    activeIndex = index;
    if (activeIndex >= 0) {
      const item = resultItems[activeIndex];
      if (!item) {
        return;
      }
      item.classList.add("is-active");
      const id = ensureResultId(item);
      input.setAttribute("aria-activedescendant", id);
      const title = getResultTitle(item);
      updateLiveRegion(liveSelection, `Result ${activeIndex + 1} of ${resultItems.length}: ${title}`);
      item.scrollIntoView({ block: "nearest" });
    } else {
      input.removeAttribute("aria-activedescendant");
      updateLiveRegion(liveSelection, "Search results selection cleared.");
    }
  }

  function clearActiveResult(announce = true) {
    if (activeIndex >= 0 && resultItems[activeIndex]) {
      resultItems[activeIndex].classList.remove("is-active");
    }
    activeIndex = -1;
    input.removeAttribute("aria-activedescendant");
    if (announce) {
      updateLiveRegion(liveSelection, "Search results selection cleared.");
    }
  }

  function ensureResultId(item) {
    if (item.id) {
      return item.id;
    }
    resultIdCounter += 1;
    const id = `${RESULT_ID_PREFIX}${resultIdCounter}`;
    item.id = id;
    return id;
  }

  function refreshResultItems() {
    resultItems = Array.from(resultsList.querySelectorAll(".search-result"));
    resultItems.forEach((item) => {
      ensureResultId(item);
      item.setAttribute("role", "option");
      item.setAttribute("tabindex", "-1");
    });
    if (activeIndex >= resultItems.length) {
      clearActiveResult(false);
    }
    if (pendingSearch) {
      pendingSearch = false;
      updateLiveRegion(liveStatus, "Results updated.");
    }
    hideLoading();
    announceCount();
  }

  function announceCount() {
    if (!liveCount) {
      return;
    }
    const query = input.value.trim();
    if (!query) {
      updateLiveRegion(liveCount, "Enter a query to begin.");
      return;
    }
    if (emptyState && !emptyState.hidden && resultItems.length === 0) {
      updateLiveRegion(liveCount, `No results found for “${query}”.`);
      return;
    }
    updateLiveRegion(liveCount, `${resultItems.length} result${resultItems.length === 1 ? "" : "s"} found for “${query}”.`);
  }

  function showLoading() {
    if (resultsRegion) {
      resultsRegion.setAttribute("aria-busy", "true");
    }
    if (loadingIndicator) {
      if (loadingHideTimer) {
        window.clearTimeout(loadingHideTimer);
        loadingHideTimer = null;
      }
      loadingStartedAt = Date.now();
      loadingIndicator.hidden = false;
      loadingIndicator.removeAttribute("hidden");
    }
  }

  function hideLoading() {
    if (resultsRegion) {
      resultsRegion.setAttribute("aria-busy", "false");
    }
    if (loadingIndicator) {
      const finalize = () => {
        loadingIndicator.hidden = true;
        loadingIndicator.setAttribute("hidden", "");
        loadingHideTimer = null;
        loadingStartedAt = 0;
      };
      const elapsed = loadingStartedAt ? Date.now() - loadingStartedAt : MIN_LOADING_VISIBLE_MS;
      if (elapsed < MIN_LOADING_VISIBLE_MS) {
        const remaining = MIN_LOADING_VISIBLE_MS - elapsed;
        if (loadingHideTimer) {
          window.clearTimeout(loadingHideTimer);
        }
        loadingHideTimer = window.setTimeout(finalize, remaining);
      } else {
        finalize();
      }
    }
  }

  function openActiveResult() {
    const item = activeIndex >= 0 ? resultItems[activeIndex] : null;
    if (!item) {
      return;
    }
    const link = item.querySelector("[data-result-link]");
    if (link && link.href) {
      window.location.assign(link.href);
    }
  }

  function clearSearch() {
    debouncedSubmit.cancel();
    input.value = "";
    clearActiveResult();
    updateLiveRegion(liveStatus, "Search cleared.");
    if (form) {
      pendingSearch = true;
      showLoading();
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  }

  function getResultTitle(item) {
    const link = item.querySelector("[data-result-link]");
    if (link && link.textContent) {
      return link.textContent.trim();
    }
    const heading = item.querySelector(".search-result__title");
    return heading && heading.textContent ? heading.textContent.trim() : "Result";
  }
}

const isTestEnvironment =
  typeof process !== "undefined" &&
  process.env &&
  (process.env.NODE_ENV === "test" || process.env.VITEST === "true");

function onReady() {
  initializeSearchApp();
  setupSearchEnhancements();
}

if (!isTestEnvironment) {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onReady);
  } else {
    onReady();
  }
}

export { debounce, updateLiveRegion, setupSearchEnhancements };
