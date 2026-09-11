/**
 * @fileoverview Feature loader for DataLog theme.
 * Dynamically loads JavaScript bundles based on page requirements.
 * @module loader
 */

/**
 * Reads and parses the JS manifest from the DOM.
 * @returns {Object} Parsed manifest object with core and features URLs
 */
function readManifest() {
  const element = document.getElementById("datalog-js-manifest");
  if (!element || !element.textContent) {
    console.warn("Loader manifest missing");
    return {};
  }
  try {
    return JSON.parse(element.textContent);
  } catch (error) {
    console.error("Unable to parse loader manifest", error);
    return {};
  }
}

/**
 * Converts a feature name to a dataset key (e.g., "dark-mode" -> "featureDarkMode").
 * @param {string} feature - The feature name with optional hyphens
 * @returns {string} Camel-cased dataset key
 */
function toDatasetKey(feature) {
  return `feature${feature.replace(/(^|-)([a-z])/g, (_, __, char) => char.toUpperCase())}`;
}

/**
 * Updates the loading state of a feature in the body's dataset.
 * @param {string} feature - The feature name
 * @param {('loading'|'ready'|'error'|'skipped')} state - The new state
 */
function updateFeatureState(feature, state) {
  const body = document.body;
  if (!body) {
    return;
  }
  const key = `${toDatasetKey(feature)}State`;
  body.dataset[key] = state;
}

/**
 * Dynamically imports a JavaScript module.
 * @param {string} url - The URL of the module to import
 * @returns {Promise<Object>} The imported module
 */
async function importModule(url) {
  return import(url);
}

/**
 * Feature configuration defining available features and their detection tests.
 * @type {Array<{name: string, test: function(): boolean}>}
 */
const FEATURE_CONFIG = [
  {
    name: "search",
    test: () => document.querySelector("[data-search-app]") || document.querySelector(".site-search")
  },
  {
    name: "visualizations",
    test: () => document.querySelector("[data-viz-type]")
  },
  {
    name: "math",
    test: () => document.body?.dataset.featureMath === "true"
  },
  {
    name: "academic",
    test: () => document.querySelector("[data-citation-metric], [data-citation-table], [data-citation-chart]")
  },
  {
    name: "notebook",
    test: () => document.body?.dataset.featureNotebook === "true" || document.querySelector(".notebook-output")
  }
];

/**
 * Bootstraps the application by loading core and feature bundles.
 * Features are loaded based on page requirements and data attributes.
 * @param {Object} [manifest] - Optional manifest object (reads from DOM if not provided)
 * @param {string} manifest.core - URL of the core bundle
 * @param {Object.<string, string>} [manifest.features] - Map of feature names to bundle URLs
 * @returns {Promise<void>}
 * @example
 * // Auto-detect from DOM
 * await bootstrapFeatures();
 *
 * // With explicit manifest
 * await bootstrapFeatures({
 *   core: '/assets/js/dist/core.js',
 *   features: { search: '/assets/js/dist/search.js' }
 * });
 */
export async function bootstrapFeatures(manifest) {
  const resolvedManifest = manifest || readManifest();
  const { core, features = {} } = resolvedManifest;
  if (!core) {
    console.warn("Core bundle missing from manifest", resolvedManifest);
    return;
  }

  try {
    updateFeatureState("core", "loading");
    await importModule(core);
    updateFeatureState("core", "ready");
  } catch (error) {
    updateFeatureState("core", "error");
    console.error("Failed to load core bundle", error);
    return;
  }

  const body = document.body;
  if (!body) {
    return;
  }

  const requestedFeatures = FEATURE_CONFIG.filter((feature) => {
    const datasetKey = toDatasetKey(feature.name);
    const datasetValue = body.dataset[datasetKey];
    const enabled = datasetValue === "true" || datasetValue === "auto";
    if (!enabled) {
      return false;
    }
    if (datasetValue === "auto") {
      return typeof feature.test === "function" ? Boolean(feature.test()) : true;
    }
    return true;
  });

  const loads = requestedFeatures.map(async ({ name, test }) => {
    const url = features[name];
    if (!url) {
      return;
    }
    updateFeatureState(name, "loading");
    try {
      if (typeof test === "function" && !test()) {
        updateFeatureState(name, "skipped");
        return;
      }
      await importModule(url);
      updateFeatureState(name, "ready");
    } catch (error) {
      updateFeatureState(name, "error");
      console.error(`Failed to load ${name} bundle`, error);
    }
  });

  await Promise.all(loads);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => bootstrapFeatures());
} else {
  bootstrapFeatures();
}
