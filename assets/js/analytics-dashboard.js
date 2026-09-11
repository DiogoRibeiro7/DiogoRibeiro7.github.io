/**
 * @fileoverview Analytics dashboard for displaying site metrics.
 * Renders visitor charts, engagement stats, and search term reports.
 * @module analytics-dashboard
 */

/**
 * Retrieves analytics data from global window object.
 * @returns {Object} Analytics data object
 */
function getAnalyticsData() {
  return window.__DATALOG_ANALYTICS__ || {};
}

/**
 * Executes callback when DOM is ready.
 * @param {function} callback - Function to execute
 * @returns {void}
 */
export function onReady(callback) {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", callback, { once: true });
  } else {
    callback();
  }
}

/**
 * Parses rows from an analytics report.
 * @param {Object} report - Report object with rows array
 * @returns {Array} Array of row objects
 */
export function parseRows(report) {
  if (!report || !Array.isArray(report.rows)) {
    return [];
  }
  return report.rows;
}

/**
 * Formats a number for display using locale settings.
 * @param {number} value - Number to format
 * @param {Object} [options] - Intl.NumberFormat options
 * @returns {string} Formatted number string
 */
export function formatNumber(value, options) {
  const formatter = new Intl.NumberFormat(undefined, options || { maximumFractionDigits: 0 });
  return formatter.format(value || 0);
}

/**
 * Displays a placeholder message in a table container.
 * @param {HTMLElement} container - Table body element
 * @param {string} message - Message to display
 * @param {number} [columns] - Number of columns to span
 * @returns {void}
 */
export function ensurePlaceholder(container, message, columns) {
  if (!container) return;
  const colSpan = columns || container.closest("table")?.querySelectorAll("th").length || 1;
  const tr = document.createElement("tr");
  tr.className = "placeholder";
  const td = document.createElement("td");
  td.setAttribute("colspan", colSpan);
  td.textContent = message;
  tr.appendChild(td);
  container.replaceChildren(tr);
}

/**
 * Renders the top posts table with page view and engagement data.
 * @returns {void}
 */
export function renderTopPosts() {
  const tbody = document.getElementById("analytics-top-posts");
  const rows = parseRows(getAnalyticsData().top_posts);
  if (!tbody) return;

  if (!rows.length) {
    ensurePlaceholder(tbody, tbody.dataset.emptyMessage || "No data available", 3);
    return;
  }

  tbody.replaceChildren();
  rows.forEach((row) => {
    const dimensions = row.dimensionValues || [];
    const metrics = row.metricValues || [];
    const path = dimensions[0]?.value || "";
    const views = Number(metrics[0]?.value || 0);
    const engagement = Number(metrics[1]?.value || 0);
    const tr = document.createElement("tr");
    const link = document.createElement("a");
    link.href = path.startsWith("/") ? path : `/${path}`;
    link.textContent = path || "—";
    link.rel = "noopener";
    link.className = "analytics-link";

    const pathCell = document.createElement("td");
    pathCell.appendChild(link);

    const viewsCell = document.createElement("td");
    viewsCell.textContent = formatNumber(views);

    const engagementCell = document.createElement("td");
    engagementCell.textContent = formatNumber(engagement, { maximumFractionDigits: 0 });

    tr.append(pathCell, viewsCell, engagementCell);
    tbody.appendChild(tr);
  });
}

export function renderSearchTerms() {
  const tbody = document.getElementById("analytics-search-terms");
  if (!tbody) return;
  const rows = parseRows(getAnalyticsData().search_terms);

  if (!rows.length) {
    ensurePlaceholder(tbody, tbody.dataset.emptyMessage || "No search queries recorded", 2);
    return;
  }

  tbody.replaceChildren();
  rows.forEach((row) => {
    const dimensions = row.dimensionValues || [];
    const metrics = row.metricValues || [];
    const term = dimensions[0]?.value || "(not provided)";
    const sessions = Number(metrics[0]?.value || 0);
    const tr = document.createElement("tr");
    const termCell = document.createElement("td");
    termCell.textContent = term;
    const countCell = document.createElement("td");
    countCell.textContent = formatNumber(sessions);
    tr.append(termCell, countCell);
    tbody.appendChild(tr);
  });
}

export function formatDateLabel(value) {
  if (!value) return value;
  if (value.includes("-")) return value;
  if (value.length === 8) {
    return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}`;
  }
  return value;
}

export function renderVisitorChart() {
  const canvas = document.getElementById("analytics-visitors-chart");
  if (!canvas || typeof Chart === "undefined") return;

  const rows = parseRows(getAnalyticsData().visitor_trends);
  if (!rows.length) return;

  const labels = rows.map((row) => formatDateLabel(row.dimensionValues?.[0]?.value));
  const totalUsers = rows.map((row) => Number(row.metricValues?.[0]?.value || 0));
  const newUsers = rows.map((row) => Number(row.metricValues?.[1]?.value || 0));
  const sessions = rows.map((row) => Number(row.metricValues?.[2]?.value || 0));
  const pageViews = rows.map((row) => Number(row.metricValues?.[3]?.value || 0));

  new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Total users",
          data: totalUsers,
          borderColor: "#2563eb",
          backgroundColor: "rgba(37, 99, 235, 0.15)",
          tension: 0.35,
          fill: true,
        },
        {
          label: "New users",
          data: newUsers,
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.12)",
          tension: 0.35,
          fill: false,
        },
        {
          label: "Sessions",
          data: sessions,
          borderColor: "#9333ea",
          backgroundColor: "rgba(147, 51, 234, 0.1)",
          tension: 0.35,
          fill: false,
        },
        {
          label: "Page views",
          data: pageViews,
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.1)",
          borderDash: [6, 6],
          tension: 0.35,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { callback: (val) => formatNumber(val) },
        },
      },
      plugins: {
        legend: { position: "top" },
      },
    },
  });
}

export function renderEngagementChart() {
  const canvas = document.getElementById("analytics-engagement-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const rows = parseRows(getAnalyticsData().engagement_by_page).slice(0, 10);
  if (!rows.length) return;

  const labels = rows.map((row) => row.dimensionValues?.[0]?.value || "—");
  const engagedSessions = rows.map((row) => Number(row.metricValues?.[2]?.value || 0));
  const views = rows.map((row) => Number(row.metricValues?.[0]?.value || 0));

  new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Engaged sessions",
          data: engagedSessions,
          backgroundColor: "rgba(59, 130, 246, 0.7)",
        },
        {
          label: "Page views",
          data: views,
          backgroundColor: "rgba(14, 165, 233, 0.5)",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: { beginAtZero: true, ticks: { callback: (val) => formatNumber(val) } },
      },
      plugins: {
        legend: { position: "top" },
      },
    },
  });
}

export function renderKeyEvents() {
  const container = document.getElementById("analytics-key-events");
  if (!container) return;
  const rows = parseRows(getAnalyticsData().key_events);
  if (!rows.length) {
    const li = document.createElement("li");
    li.className = "placeholder";
    li.textContent = "No tracked events in the selected window.";
    container.replaceChildren(li);
    return;
  }

  container.replaceChildren();
  rows.forEach((row) => {
    const name = row.dimensionValues?.[0]?.value || "event";
    const count = Number(row.metricValues?.[0]?.value || 0);
    const li = document.createElement("li");
    const strong = document.createElement("strong");
    strong.textContent = formatNumber(count);
    li.appendChild(strong);
    li.appendChild(document.createTextNode(` ${name.replace(/_/g, " ")}`));
    container.appendChild(li);
  });
}

export function renderScholar() {
  const metrics = getAnalyticsData().scholar || {};
  const totalField = document.querySelector('[data-field="scholar-total"]');
  const hIndexField = document.querySelector('[data-field="scholar-h-index"]');
  const i10Field = document.querySelector('[data-field="scholar-i10-index"]');
  if (totalField) totalField.textContent = metrics.total ? formatNumber(Number(metrics.total)) : "—";
  if (hIndexField) hIndexField.textContent = metrics.h_index ? formatNumber(Number(metrics.h_index)) : "—";
  if (i10Field) i10Field.textContent = metrics.i10_index ? formatNumber(Number(metrics.i10_index)) : "—";

  const canvas = document.getElementById("analytics-scholar-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const yearly = metrics.yearly_totals || {};
  const entries = Object.entries(yearly)
    .map(([year, value]) => [Number(year), Number(value)])
    .sort((a, b) => a[0] - b[0]);
  if (!entries.length) return;

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: entries.map((item) => item[0]),
      datasets: [
        {
          label: "Citations",
          data: entries.map((item) => item[1]),
          backgroundColor: "rgba(99, 102, 241, 0.8)",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, ticks: { callback: (val) => formatNumber(val) } },
      },
    },
  });
}

export function renderMonthlyReports() {
  const tbody = document.getElementById("analytics-monthly-reports");
  if (!tbody) return;
  const reports = Array.isArray(getAnalyticsData().monthly_reports) ? getAnalyticsData().monthly_reports : [];
  if (!reports.length) {
    ensurePlaceholder(tbody, tbody.dataset.emptyMessage || "No monthly reports available", 5);
    return;
  }

  tbody.replaceChildren();
  reports.forEach((report) => {
    const tr = document.createElement("tr");
    const month = document.createElement("td");
    month.textContent = report.month || "—";
    const total = document.createElement("td");
    total.textContent = formatNumber(report.total_users || 0);
    const news = document.createElement("td");
    news.textContent = formatNumber(report.new_users || 0);
    const sessions = document.createElement("td");
    sessions.textContent = formatNumber(report.sessions || 0);
    const views = document.createElement("td");
    views.textContent = formatNumber(report.page_views || 0);
    tr.append(month, total, news, sessions, views);
    tbody.appendChild(tr);
  });
}

export function render() {
  const status = getAnalyticsData().status;
  if (status !== "ok") {
    ["analytics-top-posts", "analytics-search-terms", "analytics-monthly-reports"].forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        ensurePlaceholder(el, getAnalyticsData().message || "Analytics data unavailable.");
      }
    });
    const events = document.getElementById("analytics-key-events");
    if (events) {
      const li = document.createElement("li");
      li.className = "placeholder";
      li.textContent = getAnalyticsData().message || "Analytics data unavailable.";
      events.replaceChildren(li);
    }
    renderScholar();
    return;
  }

  renderTopPosts();
  renderSearchTerms();
  renderVisitorChart();
  renderEngagementChart();
  renderKeyEvents();
  renderScholar();
  renderMonthlyReports();
}

// Auto-initialize when module loads (backward compatibility)
// Skip auto-init in test environment
if (typeof window !== 'undefined' && !window.__VITEST__) {
  onReady(render);
}
