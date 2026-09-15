/**
 * @fileoverview Copy buttons for citation and bibliography exports.
 * The citation tools render on every post, dataset and project, so their
 * buttons are handled here in the core bundle, with one delegated listener,
 * rather than by the academic bundle, which those pages never load.
 * @module core/copy-buttons
 */

/** @constant {string} Buttons that copy the text of the element named by data-target */
const TRIGGER_SELECTOR = "[data-copy-citation], [data-copy-bibliography]";
/** @constant {number} How long the confirmation stays on the button, in ms */
const FEEDBACK_MS = 2000;

let listening = false;

function textToCopy(trigger) {
  const targetId = trigger.getAttribute("data-target");
  const target = targetId ? document.getElementById(targetId) : null;
  if (!target) {
    return "";
  }
  return "value" in target ? target.value : target.textContent || "";
}

async function copy(trigger) {
  const text = textToCopy(trigger);
  if (!text || !navigator.clipboard || !navigator.clipboard.writeText) {
    return;
  }
  if (trigger.dataset.originalText === undefined) {
    trigger.dataset.originalText = trigger.textContent;
  }
  try {
    await navigator.clipboard.writeText(text);
    trigger.dataset.copied = "true";
    trigger.textContent = "Copied!";
    setTimeout(() => {
      trigger.dataset.copied = "false";
      trigger.textContent = trigger.dataset.originalText;
    }, FEEDBACK_MS);
  } catch (error) {
    trigger.dataset.copied = "error";
    trigger.textContent = "Error";
  }
}

/**
 * Handles clicks on copy buttons anywhere in the document, including buttons
 * added later. Calling it again does not add a second listener.
 * @returns {void}
 */
export function initCopyButtons() {
  if (listening) {
    return;
  }
  listening = true;
  document.addEventListener("click", (event) => {
    const trigger = event.target instanceof Element ? event.target.closest(TRIGGER_SELECTOR) : null;
    if (trigger) {
      copy(trigger);
    }
  });
}
