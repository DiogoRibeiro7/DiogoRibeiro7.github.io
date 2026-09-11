/**
 * @fileoverview Scroll progress indicator.
 * Shows reading progress as user scrolls through content.
 * @module core/scroll-progress
 */

/**
 * Calculates and updates the scroll progress indicator.
 * @param {HTMLElement} progressBar - The progress bar element
 * @param {HTMLElement|null} progressValue - Optional element to display percentage
 */
function calculateProgress(progressBar, progressValue) {
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
  const docHeight = document.documentElement.scrollHeight - window.innerHeight;
  const progress = docHeight > 0 ? Math.min((scrollTop / docHeight) * 100, 100) : 0;

  if (progressBar.tagName === "PROGRESS") {
    progressBar.setAttribute("value", String(progress));
  } else {
    progressBar.style.setProperty("--progress", `${progress}%`);
  }

  if (progressValue) {
    progressValue.textContent = Math.round(progress);
  }
}

/**
 * Initializes the scroll progress indicator.
 * Listens to scroll and resize events to update progress.
 * @returns {void}
 */
export function initScrollProgress() {
  const progressBar = document.querySelector("[data-scroll-progress]");
  if (!progressBar) {
    return;
  }

  const progressValue = document.querySelector("[data-scroll-progress-value]");
  const updateProgress = () => calculateProgress(progressBar, progressValue);

  updateProgress();
  window.addEventListener("scroll", updateProgress, { passive: true });
  window.addEventListener("resize", updateProgress);
}
