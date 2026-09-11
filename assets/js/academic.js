/**
 * @fileoverview Academic features for citation display and research tools.
 * Handles citation metrics, publication lists, and academic integrations.
 * @module academic
 */

/**
 * Populates citation metric elements with data from global academic data.
 * Supports total citations, h-index, i10-index, and yearly breakdowns.
 * @returns {void}
 */
export function populateCitationMetrics() {
  const academicData = window.DatalogAcademic || {};
  const publicationsData = window.DatalogPublications || {};
  const metrics = publicationsData.metrics || (academicData.citations ? academicData.citations.metrics : {}) || {};
  const yearlyTotals = publicationsData.yearly_totals || (academicData.citations ? academicData.citations.yearly_totals : {}) || {};

  document.querySelectorAll('[data-citation-metric]').forEach((element) => {
    const key = element.getAttribute('data-citation-metric');
    if (!key) {
      return;
    }
    let value;
    switch (key) {
      case 'total':
        value = metrics.total;
        break;
      case 'h_index':
      case 'h-index':
        value = metrics.h_index;
        break;
      case 'i10_index':
      case 'i10-index':
        value = metrics.i10_index;
        break;
      case 'since_total':
        value = metrics.since_2019 ? metrics.since_2019.total : undefined;
        break;
      case 'since_h_index':
        value = metrics.since_2019 ? metrics.since_2019.h_index : undefined;
        break;
      case 'since_i10':
        value = metrics.since_2019 ? metrics.since_2019.i10_index : undefined;
        break;
      default:
        if (key.startsWith('year_')) {
          const yearKey = key.replace('year_', '');
          value = yearlyTotals[yearKey];
        }
        break;
    }
    if (typeof value === 'number') {
      element.textContent = value.toLocaleString();
    } else if (value) {
      element.textContent = value;
    }
  });
}

/**
 * Populates citation counts for individual publications.
 * @returns {void}
 */
export function populatePublicationCitations() {
  const academicData = window.DatalogAcademic || {};
  const publicationsData = window.DatalogPublications || {};
  const perPublication = publicationsData.citation_totals || (academicData.citations ? academicData.citations.per_publication : {}) || {};

  document.querySelectorAll('[data-publication-citations]').forEach((element) => {
    const publicationId = element.getAttribute('data-publication-citations');
    if (!publicationId) {
      return;
    }
    const lower = publicationId.toLowerCase();
    let value;
    if (perPublication[publicationId]) {
      value = perPublication[publicationId].total || perPublication[publicationId];
    } else if (perPublication[lower]) {
      value = perPublication[lower].total || perPublication[lower];
    }
    if (typeof value === 'number') {
      element.textContent = value.toLocaleString();
    }
  });
}

/**
 * Populates the citation timeline chart with yearly data.
 * @returns {void}
 */
export function populateCitationTimeline() {
  const academicData = window.DatalogAcademic || {};
  const publicationsData = window.DatalogPublications || {};
  const yearlyTotals = publicationsData.yearly_totals || (academicData.citations ? academicData.citations.yearly_totals : {}) || {};

  const timelineContainer = document.querySelector('[data-citation-timeline]');
  if (timelineContainer && yearlyTotals) {
    const listItems = timelineContainer.querySelectorAll('li[data-year]');
    listItems.forEach((item) => {
      const year = item.getAttribute('data-year');
      if (!year) {
        return;
      }
      const display = item.querySelector('[data-year-total]');
      const total = yearlyTotals[year];
      if (display && typeof total !== 'undefined') {
        display.textContent = Number(total).toLocaleString();
        item.style.setProperty('--timeline-value', Math.min(Number(total) / 5, 100));
      }
    });
  }
}

/**
 * Attaches copy-to-clipboard handlers to elements matching the selector.
 * @param {string} selector - CSS selector for trigger elements
 * @param {function(HTMLElement): string|string} getText - Function to get text or static text
 * @returns {void}
 */
export function attachCopyHandler(selector, getText) {
  document.querySelectorAll(selector).forEach((trigger) => {
    trigger.addEventListener('click', () => {
      const text = typeof getText === 'function' ? getText(trigger) : getText;
      if (!text) {
        return;
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard
          .writeText(text)
          .then(() => {
            trigger.dataset.copied = 'true';
            trigger.textContent = 'Copied!';
            setTimeout(() => {
              trigger.dataset.copied = 'false';
              trigger.textContent = trigger.dataset.originalText || 'Copy';
            }, 2000);
          })
          .catch(() => {
            trigger.dataset.copied = 'error';
            trigger.textContent = 'Error';
          });
      }
    });
    trigger.dataset.originalText = trigger.textContent;
  });
}

/**
 * Initializes copy handlers for citation and bibliography buttons.
 * @returns {void}
 */
export function initCopyHandlers() {
  attachCopyHandler('[data-copy-citation]', (trigger) => {
    const targetId = trigger.getAttribute('data-target');
    if (!targetId) {
      return '';
    }
    const target = document.getElementById(targetId);
    return target ? target.value : '';
  });

  attachCopyHandler('[data-copy-bibliography]', (trigger) => {
    const targetId = trigger.getAttribute('data-target');
    if (!targetId) {
      return '';
    }
    const target = document.getElementById(targetId);
    return target ? target.value : '';
  });
}

/**
 * Initializes the submission status filter for paper tracking.
 * @returns {void}
 */
export function initSubmissionFilter() {
  const submissionFilter = document.querySelector('[data-submission-filter]');
  if (submissionFilter) {
    const cards = document.querySelectorAll('.submission-card');
    const filterCards = () => {
      const value = submissionFilter.value;
      cards.forEach((card) => {
        if (!value) {
          card.hidden = false;
          return;
        }
        const status = card.getAttribute('data-submission-status');
        card.hidden = status !== value;
      });
    };
    submissionFilter.addEventListener('change', filterCards);
    filterCards();
  }
}

/**
 * Initializes the academic calendar event type filter.
 * @returns {void}
 */
export function initCalendarFilter() {
  const calendarFilter = document.querySelector('[data-calendar-filter]');
  const calendarList = document.querySelector('[data-academic-calendar]');
  if (calendarFilter && calendarList) {
    const events = calendarList.querySelectorAll('li[data-event-type]');
    const filterEvents = () => {
      const value = calendarFilter.value;
      events.forEach((event) => {
        if (!value) {
          event.hidden = false;
          return;
        }
        const type = event.getAttribute('data-event-type');
        event.hidden = type !== value;
      });
    };
    calendarFilter.addEventListener('change', filterEvents);
    filterEvents();
  }
}

/**
 * Initializes open science badges with tooltip descriptions.
 * @returns {void}
 */
export function initOpenScienceBadges() {
  const badges = document.querySelectorAll('.open-science-badge');
  badges.forEach((badge) => {
    const description = badge.querySelector('.open-science-badge__description');
    const criteria = badge.querySelector('.open-science-badge__criteria');
    const tooltipParts = [];
    if (description) {
      tooltipParts.push(description.textContent.trim());
    }
    if (criteria) {
      tooltipParts.push(criteria.textContent.trim());
    }
    if (tooltipParts.length) {
      badge.setAttribute('title', tooltipParts.join(' • '));
    }
  });
}

/**
 * Initializes all academic features on the page.
 * @returns {void}
 */
export function initAcademicFeatures() {
  populateCitationMetrics();
  populatePublicationCitations();
  populateCitationTimeline();
  initCopyHandlers();
  initSubmissionFilter();
  initCalendarFilter();
  initOpenScienceBadges();
}

// Auto-initialize when module loads (backward compatibility)
if (typeof window !== 'undefined' && document.readyState !== 'loading') {
  initAcademicFeatures();
} else if (typeof window !== 'undefined') {
  document.addEventListener('DOMContentLoaded', initAcademicFeatures);
}
