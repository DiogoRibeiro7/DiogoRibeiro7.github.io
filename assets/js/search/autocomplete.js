import { highlightText } from "./utils.js";

export function createAutocomplete({ panel, limit = 8, onSelect }) {
  if (!panel) {
    return {
      setSuggestions() {},
      update() {},
      close() {},
      handleNavigation() {}
    };
  }

  let suggestions = [];
  let suggestionIndex = -1;

  function setSuggestions(values) {
    suggestions = Array.isArray(values) ? [...values] : [];
  }

  function update(query) {
    suggestionIndex = -1;
    const value = (query || "").trim().toLowerCase();

    if (!value) {
      clear();
      return;
    }

    const matches = suggestions
      .filter((item) => item && item.toLowerCase().includes(value))
      .slice(0, limit);

    if (matches.length === 0) {
      clear();
      return;
    }

    const fragment = document.createDocumentFragment();
    matches.forEach((item, index) => {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "search-autocomplete__option";
      option.setAttribute("role", "option");
      option.dataset.autocompleteOption = "";
      option.dataset.value = item;
      option.setAttribute("aria-selected", index === suggestionIndex ? "true" : "false");
      option.replaceChildren();
      option.insertAdjacentHTML("beforeend", highlightText(item, value));
      fragment.appendChild(option);
    });

    panel.replaceChildren();
    panel.appendChild(fragment);
    panel.hidden = false;
  }

  function handleNavigation(event, input) {
    if (panel.hidden) {
      return;
    }

    const options = Array.from(panel.querySelectorAll("[data-autocomplete-option]"));
    if (options.length === 0) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      suggestionIndex = (suggestionIndex + 1) % options.length;
      updateActive(options);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      suggestionIndex = (suggestionIndex - 1 + options.length) % options.length;
      updateActive(options);
    } else if (event.key === "Enter" && suggestionIndex >= 0) {
      event.preventDefault();
      const option = options[suggestionIndex];
      const value = option ? option.dataset.value : "";
      if (value && input) {
        input.value = value;
      }
      if (typeof onSelect === "function") {
        onSelect(value);
      }
      close();
    }
  }

  function updateActive(options) {
    options.forEach((option, index) => {
      const isActive = index === suggestionIndex;
      option.setAttribute("aria-selected", isActive ? "true" : "false");
      option.classList.toggle("is-active", isActive);
    });
  }

  function clear() {
    panel.replaceChildren();
    panel.hidden = true;
    suggestionIndex = -1;
  }

  function close() {
    clear();
  }

  panel.addEventListener("click", (event) => {
    const target = event.target.closest("[data-autocomplete-option]");
    if (!target) {
      return;
    }
    const value = target.dataset.value || "";
    if (value && typeof onSelect === "function") {
      onSelect(value);
    }
    close();
  });

  return { setSuggestions, update, close, handleNavigation };
}
