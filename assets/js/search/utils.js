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

export function tokenize(query = "") {
  return normalize(query)
    .split(/[^a-z0-9\\._]+/)
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

export function stripHtml(value) {
  if (!value) {
    return "";
  }
  const temp = document.createElement("div");
  temp.innerHTML = value;
  return temp.textContent || temp.innerText || "";
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

export function highlightText(text, query) {
  if (!text || !query) {
    return text;
  }
  const tokens = Array.from(new Set(tokenize(query))).sort((a, b) => b.length - a.length);
  let highlighted = text;
  tokens.forEach((token) => {
    if (!token) return;
    const regex = new RegExp(`(${escapeRegex(token)})`, "gi");
    highlighted = highlighted.replace(regex, "<mark>$1</mark>");
  });
  return highlighted;
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
