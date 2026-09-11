/**
 * @fileoverview Jupyter notebook rendering support.
 * Handles Plotly, Vega visualizations, cell annotations, and scroll spy.
 * @module notebook
 */

/**
 * Dynamically loads a script if not already present.
 * @param {string} src - The script URL to load
 * @returns {Promise<void>} Resolves when script is loaded
 */
export const loadScript = (src) =>
  new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve();
      return;
    }
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = (event) => reject(event);
    document.head.appendChild(script);
  });

/**
 * Initializes Plotly visualizations from notebook output elements.
 * @returns {Promise<void>} Resolves when all Plotly figures are rendered
 */
export const initPlotly = () => {
  const plotlyTargets = document.querySelectorAll('.notebook-output-plotly');
  if (!plotlyTargets.length) {
    return Promise.resolve();
  }

  const ensurePlotly = () =>
    typeof window.Plotly !== 'undefined'
      ? Promise.resolve(window.Plotly)
      : loadScript('https://cdn.plot.ly/plotly-2.27.0.min.js').then(() => window.Plotly);

  return ensurePlotly().then((Plotly) => {
    plotlyTargets.forEach((target) => {
      try {
        const spec = JSON.parse(target.dataset.plotly || '{}');
        const layout = spec.layout || {};
        const data = Array.isArray(spec.data) ? spec.data : Array.isArray(spec) ? spec : [];
        Plotly.newPlot(target, data, layout, { responsive: true });
      } catch (error) {
        target.replaceChildren();
        target.insertAdjacentHTML('afterbegin',
          '<p class="notebook-output__warning">Plotly figure could not be rendered. Open in Binder or Colab to interact.</p>');
      }
    });
  });
};

/**
 * Initializes Vega/Vega-Lite visualizations from notebook output elements.
 * @returns {Promise<void>} Resolves when all Vega specs are rendered
 */
export const initVega = () => {
  const vegaTargets = document.querySelectorAll('.notebook-output-vega');
  if (!vegaTargets.length) {
    return Promise.resolve();
  }

  const ensureVega = () =>
    loadScript('https://cdn.jsdelivr.net/npm/vega@5')
      .then(() => loadScript('https://cdn.jsdelivr.net/npm/vega-lite@5'))
      .then(() => loadScript('https://cdn.jsdelivr.net/npm/vega-embed@6'));

  return ensureVega().then(() => {
    vegaTargets.forEach((target) => {
      try {
        const spec = JSON.parse(target.dataset.vega || '{}');
        window.vegaEmbed(target, spec, { actions: false });
      } catch (error) {
        target.replaceChildren();
        target.insertAdjacentHTML('afterbegin',
          '<p class="notebook-output__warning">Vega specification could not be parsed. Open in Binder for the interactive version.</p>');
      }
    });
  });
};

/**
 * Annotates notebook cells with anchor IDs for navigation.
 * @returns {void}
 */
export const annotateCells = () => {
  const segments = document.querySelectorAll('.notebook-segment');
  segments.forEach((segment, index) => {
    const anchorId = segment.id || `cell-${index + 1}`;
    segment.id = anchorId;
    segment.setAttribute('data-cell-anchor', anchorId);
  });
};

/**
 * Sets up scroll spy to highlight active cell in sidebar navigation.
 * @returns {void}
 */
export const handleScrollSpy = () => {
  const observerOptions = {
    root: null,
    rootMargin: '0px 0px -65%',
    threshold: 0
  };

  const entries = document.querySelectorAll('.notebook-segment');
  const sidebarLinks = document.querySelectorAll('[data-cell-anchor-link]');

  if (!entries.length || !sidebarLinks.length) {
    return;
  }

  const observer = new IntersectionObserver((changes) => {
    changes.forEach((change) => {
      if (change.isIntersecting) {
        sidebarLinks.forEach((link) => {
          link.classList.toggle('is-active', link.hash === `#${change.target.id}`);
        });
      }
    });
  }, observerOptions);

  entries.forEach((entry) => observer.observe(entry));
};

/**
 * Initializes all notebook features including visualizations and navigation.
 * @returns {void}
 */
export const initNotebookFeatures = () => {
  annotateCells();
  initPlotly().catch(() => {
    document.querySelectorAll('.notebook-output-plotly').forEach((target) => {
      target.replaceChildren();
      target.insertAdjacentHTML('afterbegin',
        '<p class="notebook-output__warning">Plotly assets failed to load. Launch Binder/Colab to view the interactive chart.</p>');
    });
  });
  initVega().catch(() => {
    document.querySelectorAll('.notebook-output-vega').forEach((target) => {
      target.replaceChildren();
      target.insertAdjacentHTML('afterbegin',
        '<p class="notebook-output__warning">Vega assets failed to load. Launch Binder/Colab to view the interactive chart.</p>');
    });
  });
  handleScrollSpy();
};

// Auto-initialize when module loads (backward compatibility)
if (typeof window !== 'undefined' && document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initNotebookFeatures);
} else if (typeof window !== 'undefined') {
  initNotebookFeatures();
}
