import { formatType, highlightText, toTitleCase } from "./utils.js";

export function renderResults(results, query, elements) {
  const { resultsList, resultsMeta, emptyState, template } = elements;

  if (!resultsList || !resultsMeta || !emptyState || !template) {
    return;
  }

  resultsList.replaceChildren();

  if (!query) {
    resultsMeta.textContent = "Enter a query to begin.";
    emptyState.hidden = true;
    return;
  }

  if (results.length === 0) {
    resultsMeta.textContent = `No matches for “${query}”.`;
    emptyState.hidden = false;
    return;
  }

  emptyState.hidden = true;
  resultsMeta.textContent = `${results.length} result${results.length === 1 ? "" : "s"} for “${query}”.`;

  const fragment = document.createDocumentFragment();

  results.forEach((result) => {
    const clone = template.content.cloneNode(true);
    const typeEl = clone.querySelector("[data-result-type]");
    const difficultyEl = clone.querySelector("[data-result-difficulty]");
    const languageEl = clone.querySelector("[data-result-languages]");
    const linkEl = clone.querySelector("[data-result-link]");
    const summaryEl = clone.querySelector("[data-result-summary]");
    const codeEl = clone.querySelector("[data-result-code]");
    const codeCodeEl = codeEl ? codeEl.querySelector("code") : null;
    const mathEl = clone.querySelector("[data-result-math]");
    const mathPreviewEl = clone.querySelector("[data-result-math-preview]");
    const mathCodeEl = clone.querySelector("[data-result-math-code]");
    const mathCopyButton = clone.querySelector("[data-result-math-copy]");
    const excerptEl = clone.querySelector("[data-result-excerpt]");
    const tagsEl = clone.querySelector("[data-result-tags]");
    const dateEl = clone.querySelector("[data-result-date]");

    if (typeEl) {
      typeEl.textContent = formatType(result.type);
    }

    if (difficultyEl) {
      if (result.difficulty) {
        difficultyEl.textContent = `Difficulty: ${toTitleCase(result.difficulty)}`;
        difficultyEl.hidden = false;
      } else {
        difficultyEl.hidden = true;
      }
    }

    if (languageEl) {
      if (result.languages && result.languages.length > 0) {
        languageEl.textContent = `Languages: ${result.languages.map((lang) => toTitleCase(lang)).join(", ")}`;
        languageEl.hidden = false;
      } else {
        languageEl.hidden = true;
      }
    }

    if (linkEl) {
      linkEl.textContent = result.title || result.url;
      linkEl.setAttribute("href", result.url);
    }

    if (summaryEl) {
      summaryEl.replaceChildren();
      summaryEl.insertAdjacentHTML("beforeend", highlightText(result.summary || "", query));
    }

    if (mathEl && mathCodeEl) {
      if (result.mathSnippet) {
        mathEl.hidden = false;
        mathCodeEl.textContent = result.mathSnippet;
        if (mathPreviewEl) {
          if (window.DatalogMath && typeof window.DatalogMath.renderLatex === "function") {
            window.DatalogMath.renderLatex(mathPreviewEl, result.mathSnippet, {
              display: false,
              enhance: false
            });
          } else if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
            mathPreviewEl.innerHTML = `\\(${result.mathSnippet}\\)`;
            window.MathJax.typesetPromise([mathPreviewEl]).catch(() => {
              mathPreviewEl.textContent = result.mathSnippet;
            });
          } else {
            mathPreviewEl.textContent = result.mathSnippet;
          }
        }
        if (mathCopyButton) {
          mathCopyButton.addEventListener("click", () => copyToClipboard(result.mathSnippet, mathEl));
        }
      } else {
        mathEl.hidden = true;
      }
    }

    if (excerptEl) {
      if (result.snippet) {
        excerptEl.replaceChildren();
        excerptEl.insertAdjacentHTML("beforeend", highlightText(result.snippet, query));
        excerptEl.hidden = false;
      } else {
        excerptEl.hidden = true;
      }
    }

    if (codeEl && codeCodeEl) {
      if (result.codeSnippet && result.codeSnippet.code) {
        const language = result.codeSnippet.language || "text";
        codeEl.classList.remove("language-none");
        codeEl.classList.add(`language-${language}`);
        codeCodeEl.className = `language-${language}`;
        const code = result.codeSnippet.code;
        if (window.Prism && window.Prism.languages) {
          const grammar = window.Prism.languages[language] || window.Prism.languages.markup;
          codeCodeEl.innerHTML = window.Prism.highlight(code, grammar, language);
        } else {
          codeCodeEl.textContent = code;
        }
        codeEl.hidden = false;
      } else {
        codeEl.hidden = true;
      }
    }

    if (tagsEl) {
      tagsEl.replaceChildren();
      (result.tags || []).forEach((tag) => {
        const li = document.createElement("li");
        li.textContent = `#${tag}`;
        tagsEl.appendChild(li);
      });
    }

    if (dateEl) {
      if (result.date) {
        const date = new Date(result.date);
        if (!Number.isNaN(date.getTime())) {
          dateEl.textContent = date.toLocaleDateString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric"
          });
          dateEl.hidden = false;
        } else {
          dateEl.hidden = true;
        }
      } else {
        dateEl.hidden = true;
      }
    }

    fragment.appendChild(clone);
  });

  resultsList.appendChild(fragment);
}

function copyToClipboard(value, container) {
  if (!value) {
    return;
  }

  const provideFeedback = () => {
    if (!container) {
      return;
    }
    const button = container.querySelector("[data-result-math-copy]");
    const sr = button ? button.querySelector(".visually-hidden") : null;
    if (button && sr) {
      const original = button.getAttribute("data-sr-original") || sr.textContent;
      if (!button.getAttribute("data-sr-original")) {
        button.setAttribute("data-sr-original", original || "Copy LaTeX");
      }
      sr.textContent = "Copied LaTeX";
      button.classList.add("is-copied");
      window.setTimeout(() => {
        sr.textContent = button.getAttribute("data-sr-original") || "Copy LaTeX";
        button.classList.remove("is-copied");
      }, 2000);
    }
  };

  const fallback = () => {
    const temp = document.createElement("textarea");
    temp.value = value;
    temp.setAttribute("aria-hidden", "true");
    temp.style.position = "fixed";
    temp.style.opacity = "0";
    document.body.appendChild(temp);
    temp.select();
    try {
      document.execCommand("copy");
      provideFeedback();
    } catch (error) {
      console.warn("Unable to copy LaTeX snippet", error);
    }
    document.body.removeChild(temp);
  };

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard
      .writeText(value)
      .then(() => provideFeedback())
      .catch(() => fallback());
  } else {
    fallback();
  }
}
