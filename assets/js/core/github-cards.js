/**
 * @fileoverview GitHub repository cards with live stats.
 * Fetches and displays repository statistics with caching.
 * @module core/github-cards
 */

/** @constant {string} Prefix for localStorage cache keys */
const CACHE_PREFIX = "datalog-github-";

/**
 * Generates a cache key for a repository.
 * @param {string} owner - Repository owner
 * @param {string} repo - Repository name
 * @returns {string} Cache key
 */
function cacheKey(owner, repo) {
  return `${CACHE_PREFIX}${owner}-${repo}`;
}

function readCache(owner, repo, cacheTtl) {
  try {
    const raw = localStorage.getItem(cacheKey(owner, repo));
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      return null;
    }
    if (Date.now() - parsed.timestamp > cacheTtl) {
      localStorage.removeItem(cacheKey(owner, repo));
      return null;
    }
    return parsed.data;
  } catch (error) {
    console.warn("GitHub cache read failed", error);
    return null;
  }
}

function writeCache(owner, repo, data) {
  try {
    localStorage.setItem(
      cacheKey(owner, repo),
      JSON.stringify({ timestamp: Date.now(), data })
    );
  } catch (error) {
    console.warn("GitHub cache write failed", error);
  }
}

async function fetchRepository(owner, repo, apiBase, cacheTtl) {
  const cached = readCache(owner, repo, cacheTtl);
  if (cached) {
    return cached;
  }
  try {
    const response = await fetch(`${apiBase}/${owner}/${repo}`);
    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }
    const json = await response.json();
    writeCache(owner, repo, json);
    return json;
  } catch (error) {
    console.warn(`Unable to load GitHub data for ${owner}/${repo}`);
    return null;
  }
}

function applyRepositoryData(card, data) {
  const statElements = card.querySelectorAll("[data-github-stat]");
  statElements.forEach((element) => {
    const stat = element.getAttribute("data-github-stat");
    if (!stat) {
      return;
    }
    const value = data[stat];
    if (typeof value === "number") {
      element.textContent = value.toLocaleString();
    } else if (typeof value === "string" && value.length > 0) {
      element.textContent = value;
    }
  });
}

/**
 * Initializes GitHub repository cards by fetching live stats.
 * Uses localStorage caching to minimize API calls.
 * @returns {Promise<void[]>} Resolves when all cards are populated
 */
export function initGitHubCards() {
  const cards = document.querySelectorAll("[data-github-owner][data-github-repo]");
  if (!cards.length) {
    return Promise.resolve();
  }

  const integrations = window.DatalogIntegrations || {};
  const githubConfig = integrations.github || {};
  if (githubConfig.enabled === false) {
    return Promise.resolve();
  }

  const apiBase = githubConfig.api_base || "https://api.github.com/repos";
  const cacheTtl = Number(githubConfig.cache_ttl || 43200) * 1000;

  const promises = Array.from(cards).map((card) => {
    const owner = card.getAttribute("data-github-owner");
    const repo = card.getAttribute("data-github-repo");
    if (!owner || !repo) {
      return Promise.resolve();
    }

    return fetchRepository(owner, repo, apiBase, cacheTtl).then((data) => {
      if (!data) {
        return;
      }
      applyRepositoryData(card, data);
    });
  });

  return Promise.all(promises);
}
