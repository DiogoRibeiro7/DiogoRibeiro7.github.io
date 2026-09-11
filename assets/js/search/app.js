import { createAnalyticsManager } from "./analytics.js";
import { createAutocomplete } from "./autocomplete.js";
import { createSearchEngine } from "./engine.js";
import { renderTagFilters } from "./filters.js";
import { renderResults } from "./render.js";

const AUTOCOMPLETE_LIMIT = 8;

export function initializeSearchApp() {
  const app = document.querySelector("[data-search-app]");
  if (!app) {
    return;
  }

  const form = app.querySelector("[data-search-form]");
  const input = app.querySelector("[data-search-input]");
  const resultsList = app.querySelector("[data-search-results]");
  const resultsMeta = app.querySelector("[data-search-meta]");
  const emptyState = app.querySelector("[data-search-empty]");
  const autocompletePanel = app.querySelector("[data-search-autocomplete]");
  const analyticsPanel = app.querySelector("[data-search-analytics]");
  const analyticsList = analyticsPanel ? analyticsPanel.querySelector(".search-app__analytics-list") : null;
  const analyticsEmpty = analyticsPanel ? analyticsPanel.querySelector(".search-app__analytics-empty") : null;
  const resultTemplate = document.getElementById("search-result-template");
  const filterType = app.querySelector("[data-filter-type]");
  const filterLanguage = app.querySelector("[data-filter-language]");
  const filterDifficulty = app.querySelector("[data-filter-difficulty]");
  const filterTagsContainer = app.querySelector("[data-filter-tags]");
  const SEARCH_INDEX_URL = app.getAttribute("data-search-index") || "/search.json";

  const selectedTags = new Set();
  const searchEngine = createSearchEngine();
  const analyticsManager = createAnalyticsManager({
    panel: analyticsPanel,
    list: analyticsList,
    empty: analyticsEmpty,
    onSelect: (query) => {
      if (input) {
        input.value = query;
        executeSearch();
      }
    }
  });
  const autocomplete = createAutocomplete({
    panel: autocompletePanel,
    limit: AUTOCOMPLETE_LIMIT,
    onSelect: (value) => {
      if (value && input) {
        input.value = value;
        input.focus();
        executeSearch();
      }
    }
  });

  let indexReady = false;

  const params = new URLSearchParams(window.location.search);
  if (input && params.has("q")) {
    input.value = params.get("q") || "";
  }
  if (filterType && params.has("type")) {
    filterType.value = params.get("type");
  }
  if (filterLanguage && params.has("language")) {
    filterLanguage.value = params.get("language");
  }
  if (filterDifficulty && params.has("difficulty")) {
    filterDifficulty.value = params.get("difficulty");
  }
  if (params.has("tags")) {
    params
      .get("tags")
      .split(",")
      .map((tag) => tag.trim().toLowerCase())
      .filter(Boolean)
      .forEach((tag) => selectedTags.add(tag));
  }

  focusShortcut(input, () => {
    autocomplete.close();
    executeSearch();
  });

  fetchIndex(SEARCH_INDEX_URL)
    .then((data) => {
      const documents = data.documents || [];
      searchEngine.setDocuments(documents);
      autocomplete.setSuggestions(searchEngine.suggestions());
      renderTagFilters(filterTagsContainer, documents, selectedTags);
      indexReady = true;
      restoreInitialSearch();
    })
    .catch((error) => {
      console.error("Failed to load search index", error);
      if (resultsMeta) {
        resultsMeta.textContent = "Unable to load the search index. Please try again later.";
      }
    });

  if (form) {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      executeSearch();
    });
  }

  if (input) {
    input.addEventListener("input", () => {
      autocomplete.update(input.value);
    });
    input.addEventListener("keydown", (event) => {
      autocomplete.handleNavigation(event, input);
    });
    input.addEventListener("focus", () => {
      autocomplete.update(input.value);
    });
  }

  if (filterType) {
    filterType.addEventListener("change", executeSearch);
  }
  if (filterLanguage) {
    filterLanguage.addEventListener("change", executeSearch);
  }
  if (filterDifficulty) {
    filterDifficulty.addEventListener("change", executeSearch);
  }

  app.addEventListener("click", (event) => {
    const trigger = event.target;
    if (trigger.matches("[data-filter-tag]")) {
      const tag = trigger.getAttribute("data-filter-tag");
      if (tag) {
        if (selectedTags.has(tag)) {
          selectedTags.delete(tag);
          trigger.setAttribute("aria-pressed", "false");
        } else {
          selectedTags.add(tag);
          trigger.setAttribute("aria-pressed", "true");
        }
        executeSearch();
      }
    }
  });

  document.addEventListener("click", (event) => {
    if (!autocompletePanel || autocompletePanel.hidden) {
      return;
    }
    if (!app.contains(event.target)) {
      autocomplete.close();
    }
  });

  document.addEventListener("keydown", (event) => {
    const activeTag = document.activeElement ? document.activeElement.tagName : "";
    if (["INPUT", "TEXTAREA"].includes(activeTag)) {
      return;
    }
    if (event.key === "/" && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      if (input) {
        input.focus();
        input.select();
      }
    } else if ((event.key === "k" || event.key === "K") && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (input) {
        input.focus();
        input.select();
      }
    }
  });

  analyticsManager.render();

  function restoreInitialSearch() {
    if (input && input.value.trim()) {
      executeSearch();
    } else {
      updateFiltersInUrl();
    }
  }

  function executeSearch() {
    if (!input) {
      return;
    }
    if (!indexReady) {
      if (resultsMeta) {
        resultsMeta.textContent = "Loading technical index…";
      }
      return;
    }

    const rawQuery = input.value.trim();
    const filters = {
      type: filterType ? filterType.value : "",
      language: filterLanguage ? filterLanguage.value : "",
      difficulty: filterDifficulty ? filterDifficulty.value : "",
      tags: selectedTags
    };

    const matches = searchEngine.search(rawQuery, filters);
    renderResults(matches, rawQuery, {
      resultsList,
      resultsMeta,
      emptyState,
      template: resultTemplate
    });
    updateFiltersInUrl(rawQuery);
    analyticsManager.record(rawQuery);
  }

  function updateFiltersInUrl(query) {
    const next = new URL(window.location.href);
    const urlParams = next.searchParams;
    if (query) {
      urlParams.set("q", query);
    } else {
      urlParams.delete("q");
    }
    if (filterType && filterType.value) {
      urlParams.set("type", filterType.value);
    } else {
      urlParams.delete("type");
    }
    if (filterLanguage && filterLanguage.value) {
      urlParams.set("language", filterLanguage.value);
    } else {
      urlParams.delete("language");
    }
    if (filterDifficulty && filterDifficulty.value) {
      urlParams.set("difficulty", filterDifficulty.value);
    } else {
      urlParams.delete("difficulty");
    }
    if (selectedTags.size > 0) {
      urlParams.set("tags", Array.from(selectedTags).join(","));
    } else {
      urlParams.delete("tags");
    }
    const queryString = urlParams.toString();
    const nextUrl = queryString ? `${next.pathname}?${queryString}` : next.pathname;
    window.history.replaceState({}, "", nextUrl);
  }
}

function fetchIndex(url) {
  return fetch(url, { credentials: "same-origin" }).then((response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  });
}

function focusShortcut(target, onClear) {
  if (!target) {
    return;
  }
  target.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      target.value = "";
      if (typeof onClear === "function") {
        onClear();
      }
    }
  });
}
