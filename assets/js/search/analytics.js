const ANALYTICS_STORAGE_KEY = "datalog-search-analytics";

export function createAnalyticsManager({ panel, list, empty, onSelect }) {
  if (!panel || !list || !empty) {
    return {
      record() {},
      render() {}
    };
  }

  const analytics = loadAnalytics();
  let lastQuery = "";

  function loadAnalytics() {
    // Search queries become keys on this object, so it is built with a null
    // prototype and filled by explicit own-key copy. A visitor searching for
    // "__proto__" or "constructor" then records a plain entry instead of
    // reaching Object.prototype.
    const counts = Object.create(null);

    try {
      const stored = window.localStorage.getItem(ANALYTICS_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (parsed && typeof parsed === "object") {
          Object.entries(parsed).forEach(([query, count]) => {
            if (typeof count === "number" && Number.isFinite(count)) {
              counts[query] = count;
            }
          });
        }
      }
    } catch (error) {
      console.warn("Unable to load stored analytics", error);
    }

    return counts;
  }

  function saveAnalytics() {
    try {
      window.localStorage.setItem(ANALYTICS_STORAGE_KEY, JSON.stringify(analytics));
    } catch (error) {
      console.warn("Unable to persist analytics", error);
    }
  }

  function record(query) {
    const value = (query || "").trim();
    if (!value || value === lastQuery) {
      return;
    }
    lastQuery = value;
    analytics[value] = (analytics[value] || 0) + 1;
    saveAnalytics();
    render();
  }

  function render() {
    const entries = Object.entries(analytics)
      .filter(([query]) => query && query.trim().length > 1)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5);

    list.replaceChildren();
    if (entries.length === 0) {
      empty.hidden = false;
      return;
    }

    empty.hidden = true;
    const fragment = document.createDocumentFragment();
    entries.forEach(([query, count]) => {
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "search-analytics__query";
      button.textContent = `${query} (${count})`;
      button.addEventListener("click", () => {
        if (typeof onSelect === "function") {
          onSelect(query);
        }
      });
      item.appendChild(button);
      fragment.appendChild(item);
    });
    list.appendChild(fragment);
  }

  return { record, render };
}
