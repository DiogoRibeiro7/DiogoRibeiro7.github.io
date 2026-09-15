export function normalize(value = "") {
  const source = value == null ? "" : String(value);
  try {
    return source.toLowerCase().normalize("NFKD").replace(/\p{Diacritic}/gu, "");
  } catch (error) {
    return source.toLowerCase();
  }
}

export function normalizeCode(value = "") {
  return normalize(value).replace(/\s+/g, "");
}

// Letters and digits of any script, so a query in Portuguese, Greek or
// Cyrillic keeps its words; the index is normalized the same way.
export function tokenize(query = "") {
  return normalize(query)
    .split(/[^\p{L}\p{N}\\._]+/u)
    .filter(Boolean);
}

export function isMathQuery(query = "") {
  return /\\|\^|_|\{|\}|\$/.test(query);
}

export function isCodeQuery(query = "") {
  return /`|::|->|=>|\(|\)|\{|\}|\[|\]|=|\bdef\b|\bfunction\b|\bclass\b/.test(query);
}

export function fuzzyIncludes(source, query) {
  if (!source || !query) {
    return false;
  }
  let idx = 0;
  for (let i = 0; i < source.length && idx < query.length; i += 1) {
    if (source[i] === query[idx]) {
      idx += 1;
    }
  }
  return idx === query.length;
}

// Parsed in a separate, inert document: markup assigned to an element of the
// page itself starts loading any images it contains.
export function stripHtml(value) {
  if (!value) {
    return "";
  }
  const parsed = new DOMParser().parseFromString(String(value), "text/html");
  return parsed.body.textContent || "";
}

export function buildSnippet(content, index, length) {
  if (!content) {
    return "";
  }
  const cleanContent = stripHtml(content);
  const start = Math.max(0, index - 80);
  const end = Math.min(cleanContent.length, index + length + 80);
  return `${cleanContent.slice(start, end)}${end < cleanContent.length ? "…" : ""}`;
}

export function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const HTML_ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => HTML_ESCAPES[character]);
}

/**
 * Returns HTML for `text` with the words of `query` in <mark>. The text is
 * escaped, so content from the search index shows as text instead of being
 * parsed as markup, and all words are matched in a single pass, so one
 * highlight never lands inside the tags of another.
 * @param {string} text
 * @param {string} query
 * @returns {string}
 */
export function highlightText(text, query) {
  if (!text) {
    return "";
  }
  const source = String(text);
  const tokens = Array.from(new Set(tokenize(query || ""))).sort((a, b) => b.length - a.length);
  if (!tokens.length) {
    return escapeHtml(source);
  }
  const pattern = new RegExp(tokens.map(escapeRegex).join("|"), "giu");
  let html = "";
  let last = 0;
  let match;
  while ((match = pattern.exec(source)) !== null) {
    html += `${escapeHtml(source.slice(last, match.index))}<mark>${escapeHtml(match[0])}</mark>`;
    last = match.index + match[0].length;
  }
  return html + escapeHtml(source.slice(last));
}

export function toTitleCase(value) {
  return (value || "")
    .toString()
    .split(/[-_\s]+/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatType(type) {
  switch (type) {
    case "post":
      return "Tutorial & Blog";
    case "project":
      return "Project";
    case "research":
      return "Research";
    case "dataset":
      return "Dataset";
    default:
      return toTitleCase(type || "Page");
  }
}
