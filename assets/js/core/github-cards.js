/**
 * @fileoverview GitHub repository cards with live stats.
 * Fetches and displays repository statistics with caching.
 * @module core/github-cards
 */

/** @constant {string} Prefix for localStorage cache keys */
const CACHE_PREFIX = "datalog-github-";
/**
 * @constant {number} How long a failed request is remembered, in milliseconds.
 * Without it every page view retried against the unauthenticated API limit of
 * 60 requests an hour.
 */
const FAILURE_TTL = 10 * 60 * 1000;
/** @constant {symbol} What readCache returns for a repository that failed recently */
const RECENTLY_FAILED = Symbol("recently-failed");

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
    const ttl = parsed.failed ? FAILURE_TTL : cacheTtl;
    if (Date.now() - parsed.timestamp > ttl) {
      localStorage.removeItem(cacheKey(owner, repo));
      return null;
    }
    return parsed.failed ? RECENTLY_FAILED : parsed.data;
  } catch (error) {
    console.warn("GitHub cache read failed", error);
    return null;
  }
}

function writeCache(owner, repo, entry) {
  try {
    localStorage.setItem(
      cacheKey(owner, repo),
      JSON.stringify({ timestamp: Date.now(), ...entry })
    );
  } catch (error) {
    console.warn("GitHub cache write failed", error);
  }
}

async function fetchRepository(owner, repo, apiBase, cacheTtl) {
  const cached = readCache(owner, repo, cacheTtl);
  if (cached === RECENTLY_FAILED) {
    return null;
  }
  if (cached) {
    return cached;
  }
  try {
    const response = await fetch(`${apiBase}/${owner}/${repo}`);
    if (!response.ok) {
      throw new Error(`GitHub API error: ${response.status}`);
    }
    const json = await response.json();
    writeCache(owner, repo, { data: json });
    return json;
  } catch (error) {
    console.warn(`Unable to load GitHub data for ${owner}/${repo}`);
    writeCache(owner, repo, { failed: true });
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
 * Marks a card whose statistics could not be loaded, and replaces its
 * placeholders with the card's `data-github-unavailable` text when it has one,
 * so a reader can tell missing data from data still on its way.
 * @param {HTMLElement} card - The repository card
 * @returns {void}
 */
function markUnavailable(card) {
  card.dataset.githubState = "error";
  const label = card.getAttribute("data-github-unavailable");
  if (!label) {
    return;
  }
  card.querySelectorAll("[data-github-stat]").forEach((element) => {
    element.textContent = label;
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

    card.dataset.githubState = "loading";
    return fetchRepository(owner, repo, apiBase, cacheTtl).then((data) => {
      if (!data) {
        markUnavailable(card);
        return;
      }
      applyRepositoryData(card, data);
      card.dataset.githubState = "ready";
    });
  });

  return Promise.all(promises);
}
