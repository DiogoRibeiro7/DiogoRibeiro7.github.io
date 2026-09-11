/**
 * @fileoverview Visualization rendering for multiple chart libraries.
 * Supports Plotly, D3, Observable, Vega, Bokeh, Shiny, and IPyWidgets.
 * @module visualizations
 */

(function () {
  const isTestEnvironment =
    typeof process !== 'undefined' &&
    process.env &&
    (process.env.NODE_ENV === 'test' || process.env.VITEST === 'true');
  const VIZ_SELECTOR = '[data-viz-type]';
  const SPEC_SELECTOR = 'script[type="application/json"]';
  const WIDGET_STATE_SELECTOR = 'script[type="application/vnd.jupyter.widget-state+json"]';
  const WIDGET_VIEW_SELECTOR = 'script[type="application/vnd.jupyter.widget-view+json"]';

  const loadedScripts = new Map();
  const loadedStyles = new Set();
  const TABLE_TEMPLATE_ID = 'viz-table-fallback-template';

  const loadScript = (src, options = {}) => {
    if (!src) {
      return Promise.reject(new Error('Missing script source'));
    }
    if (loadedScripts.has(src)) {
      return loadedScripts.get(src);
    }
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.async = options.async !== false;
      if (options.integrity) {
        script.integrity = options.integrity;
        script.crossOrigin = options.crossOrigin || 'anonymous';
      }
      script.onload = () => resolve(script);
      script.onerror = (event) => reject(event);
      document.head.appendChild(script);
    });
    loadedScripts.set(src, promise);
    return promise;
  };

  const loadStyle = (href) => {
    if (!href || loadedStyles.has(href)) {
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = href;
      link.onload = () => {
        loadedStyles.add(href);
        resolve(link);
      };
      link.onerror = (event) => reject(event);
      document.head.appendChild(link);
    });
  };

  const isPlainObject = (value) =>
    Object.prototype.toString.call(value) === '[object Object]';

  const stringifyValue = (value) => {
    if (value === null || typeof value === 'undefined') {
      return '';
    }
    if (value instanceof Date) {
      return value.toISOString();
    }
    if (Array.isArray(value)) {
      return value.map((entry) => stringifyValue(entry)).join(', ');
    }
    if (isPlainObject(value)) {
      return Object.entries(value)
        .map(([key, entry]) => `${key}: ${stringifyValue(entry)}`)
        .join('; ');
    }
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    if (typeof value === 'number') {
      return Number.isFinite(value) ? `${value}` : '';
    }
    return String(value);
  };

  const flattenObject = (input, trail = []) => {
    if (!isPlainObject(input)) {
      return { [trail.join(' › ') || 'Value']: input };
    }
    return Object.entries(input).reduce((accumulator, [key, value]) => {
      const path = trail.concat([key]);
      if (isPlainObject(value)) {
        Object.assign(accumulator, flattenObject(value, path));
      } else if (Array.isArray(value)) {
        accumulator[path.join(' › ')] = value
          .map((entry) => stringifyValue(entry))
          .join(', ');
      } else {
        accumulator[path.join(' › ')] = value;
      }
      return accumulator;
    }, {});
  };

  const normalizeRecords = (payload) => {
    if (!payload) {
      return [];
    }

    if (Array.isArray(payload)) {
      if (!payload.length) {
        return [];
      }

      if (payload.every((entry) => isPlainObject(entry))) {
        return payload.map((entry) => flattenObject(entry));
      }

      if (Array.isArray(payload[0])) {
        const candidateHeader = payload[0];
        const headerLooksLabel = candidateHeader.every(
          (cell) => typeof cell === 'string' || typeof cell === 'number'
        );
        const headers = candidateHeader.map((cell, index) =>
          headerLooksLabel ? stringifyValue(cell) : `Column ${index + 1}`
        );
        const startIndex = headerLooksLabel ? 1 : 0;

        return payload.slice(startIndex).map((row, rowIndex) => {
          const record = {};
          if (Array.isArray(row)) {
            headers.forEach((header, columnIndex) => {
              record[header] = row[columnIndex];
            });
            if (!headers.length) {
              row.forEach((value, columnIndex) => {
                record[`Column ${columnIndex + 1}`] = value;
              });
            }
          } else {
            record[`Row ${rowIndex + 1}`] = row;
          }
          return record;
        });
      }

      return payload.map((value, index) => ({
        Index: index + 1,
        Value: value
      }));
    }

    if (isPlainObject(payload)) {
      if (Array.isArray(payload.data)) {
        return normalizeRecords(payload.data);
      }
      if (Array.isArray(payload.values)) {
        return normalizeRecords(payload.values);
      }
      if (Array.isArray(payload.rows)) {
        return normalizeRecords(payload.rows);
      }
      return [flattenObject(payload)];
    }

    return [
      {
        Value: payload
      }
    ];
  };

  const orderColumns = (columns) => {
    const priority = [
      'Series',
      'Trace',
      'Group',
      'Category',
      'Index',
      'X',
      'Y',
      'Z',
      'Value',
      'Text'
    ];
    const result = [];
    priority.forEach((key) => {
      if (columns.includes(key)) {
        result.push(key);
      }
    });
    columns
      .filter((column) => !priority.includes(column))
      .sort()
      .forEach((column) => result.push(column));
    return result;
  };

  const getVizTitle = (element) => {
    const explicitTitle = element.getAttribute('data-viz-title');
    if (explicitTitle) {
      return explicitTitle;
    }
    const heading = element.querySelector('.viz-title');
    if (heading && heading.textContent) {
      return heading.textContent.trim();
    }
    return element.getAttribute('data-viz-slug') || 'Visualization data';
  };

  const ensureTableTemplate = () => {
    let template = document.getElementById(TABLE_TEMPLATE_ID);
    if (template) {
      return template;
    }
    template = document.createElement('template');
    template.id = TABLE_TEMPLATE_ID;
    const templateContent = document.createElement('div');
    templateContent.insertAdjacentHTML('afterbegin',
      '<div class="viz-table-container" data-viz-table-container>' +
      '<button type="button" class="viz-table-toggle" data-viz-table-toggle aria-expanded="false">Show data table</button>' +
      '<div class="viz-table-wrapper" data-viz-table-wrapper hidden>' +
      '<table class="viz-table" data-viz-table>' +
      '<caption data-viz-table-caption></caption>' +
      '<thead data-viz-table-head></thead>' +
      '<tbody data-viz-table-body></tbody>' +
      '</table></div></div>');
    template.content.appendChild(templateContent.firstElementChild);
    document.body.appendChild(template);
    return template;
  };

  const buildDataTable = (element, options = {}) => {
    const { caption, rows } = options;
    const records = normalizeRecords(rows);
    if (!records.length) {
      return;
    }

    const columns = Array.from(
      records.reduce((set, record) => {
        Object.keys(record).forEach((key) => set.add(key));
        return set;
      }, new Set())
    );

    if (!columns.length) {
      return;
    }

    const orderedColumns = orderColumns(columns);
    const template = ensureTableTemplate();
    const clone = template.content.firstElementChild.cloneNode(true);
    const toggle = clone.querySelector('[data-viz-table-toggle]');
    const wrapper = clone.querySelector('[data-viz-table-wrapper]');
    const table = clone.querySelector('[data-viz-table]');
    const captionNode = clone.querySelector('[data-viz-table-caption]');
    const head = clone.querySelector('[data-viz-table-head]');
    const body = clone.querySelector('[data-viz-table-body]');

    const tableId = `${element.getAttribute('data-viz-slug') || element.id || 'viz'}-table`;
    table.id = tableId;
    toggle.setAttribute('aria-controls', tableId);
    captionNode.textContent = caption || getVizTitle(element);

    const headerRow = document.createElement('tr');
    orderedColumns.forEach((column) => {
      const th = document.createElement('th');
      th.scope = 'col';
      th.textContent = column;
      headerRow.appendChild(th);
    });
    head.appendChild(headerRow);

    records.forEach((record) => {
      const tr = document.createElement('tr');
      orderedColumns.forEach((column, index) => {
        const value = stringifyValue(record[column]);
        if (index === 0) {
          const th = document.createElement('th');
          th.scope = 'row';
          th.textContent = value;
          tr.appendChild(th);
        } else {
          const td = document.createElement('td');
          td.textContent = value;
          tr.appendChild(td);
        }
      });
      body.appendChild(tr);
    });

    toggle.addEventListener('click', () => {
      const expanded = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', String(!expanded));
      toggle.textContent = expanded ? 'Show data table' : 'Hide data table';
      if (expanded) {
        wrapper.setAttribute('hidden', '');
      } else {
        wrapper.removeAttribute('hidden');
      }
    });

    const existing = element.querySelector('[data-viz-table-container]');
    if (existing) {
      existing.remove();
    }

    element.appendChild(clone);

    const canvas = element.querySelector('[data-viz-canvas]') || element;
    canvas.setAttribute('aria-describedby', tableId);
  };

  const buildPlotlyRecords = (traces = []) => {
    const records = [];
    traces.forEach((trace, index) => {
      const name = trace.name || `Series ${index + 1}`;
      const xValues = Array.isArray(trace.x) ? trace.x : [];
      const yValues = Array.isArray(trace.y) ? trace.y : [];
      const textValues = Array.isArray(trace.text) ? trace.text : [];
      const zValues = trace.z;

      if (Array.isArray(zValues) && zValues.length && Array.isArray(zValues[0])) {
        const xLabels = xValues.length ? xValues : zValues[0].map((_, column) => column + 1);
        const yLabels = yValues.length ? yValues : zValues.map((_, row) => row + 1);
        yLabels.forEach((yLabel, rowIndex) => {
          const row = zValues[rowIndex] || [];
          xLabels.forEach((xLabel, columnIndex) => {
            records.push({
              Series: name,
              Y: yLabel,
              X: xLabel,
              Value: Array.isArray(row) ? row[columnIndex] : row,
              Text: textValues[rowIndex] ? stringifyValue(textValues[rowIndex]) : ''
            });
          });
        });
        return;
      }

      const length = Math.max(xValues.length, yValues.length, textValues.length);
      for (let cursor = 0; cursor < length; cursor += 1) {
        records.push({
          Series: name,
          X: xValues[cursor],
          Y: yValues[cursor],
          Text: textValues[cursor]
        });
      }
    });
    return records.length ? records : null;
  };

  const buildD3Records = (captured) => {
    if (!captured) {
      return null;
    }
    if (Array.isArray(captured)) {
      return captured;
    }
    if (isPlainObject(captured) && Array.isArray(captured.data)) {
      return captured.data;
    }
    return captured;
  };

  const buildObservableRecords = (element) => {
    const dataNode = element.querySelector('script[data-observable-data]');
    if (!dataNode) {
      return null;
    }
    try {
      const parsed = JSON.parse(dataNode.textContent || 'null');
      return parsed;
    } catch (error) {
      return null;
    }
  };

  const ensurePlotly = () =>
    (typeof window.Plotly !== 'undefined'
      ? Promise.resolve()
      : loadScript('https://cdn.plot.ly/plotly-2.27.0.min.js')).then(() => window.Plotly);

  const ensureD3 = () =>
    (typeof window.d3 !== 'undefined'
      ? Promise.resolve(window.d3)
      : loadScript('https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js').then(() => window.d3));

  const ensureBokeh = () => {
    const script = 'https://cdn.bokeh.org/bokeh/release/bokeh-3.3.3.min.js';
    const style = 'https://cdn.bokeh.org/bokeh/release/bokeh-3.3.3.min.css';
    return Promise.all([loadScript(script), loadStyle(style)]).then(() => window.Bokeh);
  };

  const ensureRequire = () => {
    if (typeof window.requirejs !== 'undefined') {
      return Promise.resolve(window.requirejs);
    }
    return loadScript('https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js', {
      async: false
    }).then(() => window.requirejs);
  };

  const ensureWidgetManager = () =>
    ensureRequire().then((requirejs) => {
      const existing = window.__datalogWidgetManagerPromise;
      if (existing) {
        return existing;
      }
      const promise = new Promise((resolve, reject) => {
        requirejs.config({
          paths: {
            '@jupyter-widgets/html-manager':
              'https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@1.0.8/dist/embed-amd',
            '@jupyter-widgets/base':
              'https://cdn.jsdelivr.net/npm/@jupyter-widgets/base@6.0.7/dist/index',
            '@jupyter-widgets/controls':
              'https://cdn.jsdelivr.net/npm/@jupyter-widgets/controls@5.0.7/dist/index',
            '@lumino/coreutils': 'https://cdn.jsdelivr.net/npm/@lumino/coreutils@2.1.1/dist/index',
            '@lumino/widgets': 'https://cdn.jsdelivr.net/npm/@lumino/widgets@2.3.1/dist/index'
          }
        });
        requirejs(
          ['@jupyter-widgets/html-manager'],
          (widgets) => resolve(widgets),
          (error) => reject(error)
        );
      });
      window.__datalogWidgetManagerPromise = promise;
      return promise;
    });

  const createToolbarButton = (label, onClick) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'viz-toolbar__button';
    button.textContent = label;
    button.addEventListener('click', (event) => {
      event.preventDefault();
      onClick(event);
    });
    return button;
  };

  const setStatus = (element, status, tone = 'info') => {
    const statusNode = element.querySelector('[data-viz-status]');
    if (statusNode) {
      statusNode.textContent = status;
      statusNode.dataset.tone = tone;
    }
  };

  const annotateVersionMetadata = (element) => {
    const version = element.getAttribute('data-viz-version');
    const updated = element.getAttribute('data-viz-updated');
    const metaTarget = element.querySelector('[data-viz-meta]');
    if (!metaTarget) {
      return;
    }
    const fragments = [];
    if (version) {
      fragments.push(`v${version}`);
    }
    if (updated) {
      fragments.push(`Updated ${updated}`);
    }
    metaTarget.textContent = fragments.join(' · ');
  };

  const renderPlotly = (element) => {
    const canvas = element.querySelector('[data-viz-canvas]') || element;
    const specNode = element.querySelector(SPEC_SELECTOR);
    if (!specNode) {
      setStatus(element, 'Missing Plotly specification', 'error');
      return Promise.resolve();
    }
    let spec = {};
    try {
      spec = JSON.parse(specNode.textContent || '{}');
    } catch (error) {
      setStatus(element, 'Invalid Plotly specification', 'error');
      return Promise.resolve();
    }

    const data = Array.isArray(spec.data) ? spec.data : [];
    const layout = Object.assign(
      {
        autosize: true
      },
      spec.layout || {}
    );

    return ensurePlotly()
      .then((Plotly) => Plotly.newPlot(canvas, data, layout, { responsive: true, displaylogo: false }))
      .then(() => {
        setStatus(element, 'Interactive');
        const exportTarget = element.querySelector('[data-viz-export]');
        if (exportTarget) {
          exportTarget.replaceChildren();
          exportTarget.appendChild(
            createToolbarButton('Download PNG', () => {
              ensurePlotly().then((Plotly) => {
                Plotly.downloadImage(canvas, {
                  format: 'png',
                  filename: element.getAttribute('data-viz-slug') || 'visualization'
                });
              });
            })
          );
        }
        const records = buildPlotlyRecords(data);
        if (records) {
          buildDataTable(element, {
            caption: layout && layout.title ? stringifyValue(layout.title.text || layout.title) : undefined,
            rows: records
          });
        }
      })
      .catch(() => {
        setStatus(element, 'Plotly failed to load', 'error');
      });
  };

  const renderD3 = (element) => {
    const scriptNode = element.querySelector('[data-d3-script]');
    if (!scriptNode) {
      setStatus(element, 'Missing D3 rendering instructions', 'error');
      return Promise.resolve();
    }
    return ensureD3()
      .then((d3) => {
        const code = scriptNode.textContent || '';
        const canvas = element.querySelector('[data-viz-canvas]') || element;
        canvas.replaceChildren();
        const captured = [];
        const captureData = (payload) => {
          if (typeof payload !== 'undefined') {
            captured.push(payload);
          }
        };
        try {
          const instrumented = `${code}\n;try { if (typeof data !== 'undefined') captureData(data); } catch (instrumentationError) {}`;
          const runner = new Function('d3', 'element', 'captureData', instrumented);
          const result = runner(d3, canvas, captureData);
          if (typeof result !== 'undefined') {
            captureData(result);
          }
          const harvested = captured.length ? captured[captured.length - 1] : null;
          const records = buildD3Records(harvested);
          if (records) {
            buildDataTable(element, { rows: records });
          }
          setStatus(element, 'Interactive');
        } catch (error) {
          canvas.replaceChildren();
          setStatus(element, 'D3 rendering error', 'error');
        }
      })
      .catch(() => setStatus(element, 'D3 assets failed to load', 'error'));
  };

  const renderObservable = (element) => {
    const src = element.getAttribute('data-observable-src');
    if (!src) {
      setStatus(element, 'Missing Observable notebook source', 'error');
      return Promise.resolve();
    }
    const iframe = document.createElement('iframe');
    iframe.src = src;
    iframe.loading = 'lazy';
    iframe.title = element.getAttribute('data-viz-title') || 'Observable visualization';
    iframe.setAttribute('allowfullscreen', '');
    iframe.setAttribute('sandbox', 'allow-scripts allow-same-origin allow-popups allow-forms');
    iframe.className = 'viz-embed';
    const canvas = element.querySelector('[data-viz-canvas]') || element;
    canvas.replaceChildren(iframe);
    setStatus(element, 'Embedded');
    const records = buildObservableRecords(element);
    if (records) {
      buildDataTable(element, { rows: records });
    }
    return Promise.resolve();
  };

  const renderBokeh = (element) => {
    const specNode = element.querySelector(SPEC_SELECTOR);
    const scriptNode = element.querySelector('[data-bokeh-script]');
    const canvas = element.querySelector('[data-viz-canvas]') || element;

    if (!specNode && !scriptNode) {
      setStatus(element, 'Missing Bokeh specification', 'error');
      return Promise.resolve();
    }

    return ensureBokeh()
      .then((Bokeh) => {
        if (specNode) {
          let spec = {};
          try {
            spec = JSON.parse(specNode.textContent || '{}');
          } catch (error) {
            setStatus(element, 'Invalid Bokeh specification', 'error');
            return;
          }
          const targetId = canvas.id || `bokeh-${Math.random().toString(36).slice(2)}`;
          canvas.id = targetId;
          try {
            Bokeh.embed.embed_item(spec, targetId);
            setStatus(element, 'Interactive');
          } catch (error) {
            setStatus(element, 'Bokeh rendering error', 'error');
          }
          return;
        }

        if (scriptNode) {
          const code = scriptNode.textContent || '';
          canvas.replaceChildren();
          try {
            const runner = new Function('Bokeh', 'element', code);
            runner(Bokeh, canvas);
            setStatus(element, 'Interactive');
          } catch (error) {
            console.error('Bokeh script error', error);
            setStatus(element, 'Bokeh rendering error', 'error');
          }
        }
      })
      .catch(() => {
        setStatus(element, 'Bokeh assets failed to load', 'error');
      });
  };

  const renderShiny = (element) => {
    const src = element.getAttribute('data-shiny-src');
    if (!src) {
      setStatus(element, 'Missing Shiny application source', 'error');
      return Promise.resolve();
    }
    const iframe = document.createElement('iframe');
    iframe.src = src;
    iframe.loading = 'lazy';
    iframe.title = element.getAttribute('data-viz-title') || 'R Shiny application';
    iframe.className = 'viz-embed';
    iframe.setAttribute('allow', 'fullscreen');
    iframe.setAttribute('sandbox', 'allow-scripts allow-forms allow-same-origin');
    const height = element.getAttribute('data-shiny-height');
    iframe.style.height = height || '640px';
    const canvas = element.querySelector('[data-viz-canvas]') || element;
    canvas.replaceChildren(iframe);
    setStatus(element, 'Embedded');
    return Promise.resolve();
  };

  const renderWidgets = (element) => {
    const stateNode = element.querySelector(WIDGET_STATE_SELECTOR);
    const viewNode = element.querySelector(WIDGET_VIEW_SELECTOR);
    if (!stateNode || !viewNode) {
      setStatus(element, 'Missing widget state payload', 'error');
      return Promise.resolve();
    }

    let state;
    let view;
    try {
      state = JSON.parse(stateNode.textContent || '{}');
      view = JSON.parse(viewNode.textContent || '{}');
    } catch (error) {
      setStatus(element, 'Invalid widget payload', 'error');
      return Promise.resolve();
    }

    const canvas = element.querySelector('[data-viz-canvas]') || element;
    canvas.replaceChildren();

    return ensureWidgetManager()
      .then((widgets) => {
        const manager = new widgets.WidgetManager();
        return manager
          .set_state(state)
          .then(() => manager.get_model(view.model_id))
          .then((model) => manager.create_view(model))
          .then((widgetView) => manager.display_view(undefined, widgetView, { el: canvas }))
          .then(() => setStatus(element, 'Interactive'))
          .catch((error) => {
            console.error('Widget render error', error);
            setStatus(element, 'Widget rendering error', 'error');
          });
      })
      .catch(() => setStatus(element, 'Widget assets failed to load', 'error'));
  };

  const renderVisualization = (element) => {
    const type = element.getAttribute('data-viz-type');
    annotateVersionMetadata(element);
    switch (type) {
      case 'plotly':
        return renderPlotly(element);
      case 'observable':
        return renderObservable(element);
      case 'd3':
        return renderD3(element);
      case 'bokeh':
        return renderBokeh(element);
      case 'shiny':
        return renderShiny(element);
      case 'ipywidgets':
        return renderWidgets(element);
      default:
        setStatus(element, 'Unsupported visualization type', 'error');
        return Promise.resolve();
    }
  };

  const initializeVisualizations = () => {
    if (!('IntersectionObserver' in window)) {
      document.querySelectorAll(VIZ_SELECTOR).forEach((element) => {
        renderVisualization(element);
      });
      return;
    }

    const observer = new IntersectionObserver(
      (entries, entryObserver) => {
        entries
          .filter((entry) => entry.isIntersecting)
          .forEach((entry) => {
            const element = entry.target;
            entryObserver.unobserve(element);
            renderVisualization(element);
          });
      },
      { rootMargin: '200px 0px' }
    );

    document.querySelectorAll(VIZ_SELECTOR).forEach((element) => {
      observer.observe(element);
    });
  };

  const filterGallery = (filters) => {
    const items = document.querySelectorAll('[data-viz-card]');
    items.forEach((item) => {
      const matchesTech = !filters.tech || item.dataset.vizTech === filters.tech;
      const tags = (item.dataset.vizTags || '')
        .split(',')
        .map((tag) => tag.trim())
        .filter(Boolean);
      const matchesTag = !filters.tag || tags.includes(filters.tag);
      item.hidden = !(matchesTech && matchesTag);
    });
  };

  const initializeGallery = () => {
    const gallery = document.querySelector('[data-viz-gallery]');
    if (!gallery) {
      return;
    }

    const techSelect = gallery.querySelector('[data-viz-filter="tech"]');
    const tagSelect = gallery.querySelector('[data-viz-filter="tag"]');

    const update = () => {
      filterGallery({
        tech: techSelect && techSelect.value !== 'all' ? techSelect.value : '',
        tag: tagSelect && tagSelect.value !== 'all' ? tagSelect.value : ''
      });
    };

    if (techSelect) {
      techSelect.addEventListener('change', update);
    }
    if (tagSelect) {
      tagSelect.addEventListener('change', update);
    }

    update();
  };

  const boot = () => {
    initializeVisualizations();
    initializeGallery();
  };

  const internals = {
    loadScript,
    loadStyle,
    normalizeRecords,
    stringifyValue,
    buildDataTable,
    annotateVersionMetadata,
    buildPlotlyRecords,
    buildD3Records,
    buildObservableRecords,
    filterGallery,
    renderVisualization,
    initializeVisualizations,
    initializeGallery
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.__DATALOG_VIZ_INTERNALS__ = internals;
  }

  if (!isTestEnvironment) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', boot);
    } else {
      boot();
    }
  }
})();
