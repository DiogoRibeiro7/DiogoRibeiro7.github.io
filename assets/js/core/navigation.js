/**
 * @fileoverview Navigation module for responsive site navigation.
 * Handles mobile menu toggling, smooth scrolling, and keyboard navigation.
 * @module core/navigation
 */

/** @constant {string} Media query for desktop viewport */
const DESKTOP_MEDIA = "(min-width: 48rem)";
/** @constant {string} Selector for the main navigation element */
const NAV_SELECTOR = "#site-nav";
/** @constant {string} Selector for the navigation toggle button */
const TOGGLE_SELECTOR = ".nav-toggle";
/** @constant {string} Class applied to body when nav is open */
const NAV_OPEN_CLASS = "nav-open";

/**
 * Normalizes a URL path by removing trailing slashes and index.html.
 * @param {string} path - The path to normalize
 * @returns {string} Normalized path
 */
function normalizePath(path) {
  return path.replace(/index\.html$/, "").replace(/\/$/, "");
}

function getAnchorTarget(link) {
  const href = link.getAttribute("href");
  if (!href) {
    return null;
  }
  try {
    const destination = new URL(href, window.location.origin);
    if (!destination.hash) {
      return null;
    }
    const current = new URL(window.location.href);
    if (normalizePath(destination.pathname) !== normalizePath(current.pathname)) {
      return null;
    }
    return destination.hash.replace("#", "");
  } catch (error) {
    return null;
  }
}

function smoothScrollToId(id) {
  if (!id) {
    return false;
  }
  const target = document.getElementById(id);
  if (!target) {
    return false;
  }
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  return true;
}

/**
 * Initializes the navigation system including mobile toggle,
 * smooth scrolling for anchor links, and keyboard accessibility.
 * @returns {void}
 */
export function initNavigation() {
  const siteNav = document.querySelector(NAV_SELECTOR);
  if (!siteNav) {
    return;
  }

  const navToggle = document.querySelector(TOGGLE_SELECTOR);
  const body = document.body;
  const navMediaQuery = window.matchMedia(DESKTOP_MEDIA);

  const syncNavForViewport = (isDesktop) => {
    if (isDesktop) {
      siteNav.dataset.open = "true";
      body.classList.remove(NAV_OPEN_CLASS);
      if (navToggle) {
        navToggle.setAttribute("aria-expanded", "false");
        navToggle.classList.remove("is-active");
      }
      return;
    }

    const isOpen = siteNav.dataset.open === "true";
    if (!isOpen) {
      siteNav.dataset.open = "false";
    }
    if (navToggle) {
      navToggle.setAttribute("aria-expanded", String(isOpen));
      navToggle.classList.toggle("is-active", isOpen);
    }
  };

  const closeNav = ({ focusToggle = false } = {}) => {
    if (navMediaQuery.matches) {
      return;
    }
    siteNav.dataset.open = "false";
    body.classList.remove(NAV_OPEN_CLASS);
    if (navToggle) {
      navToggle.setAttribute("aria-expanded", "false");
      navToggle.classList.remove("is-active");
      if (focusToggle) {
        navToggle.focus();
      }
    }
  };

  const openNav = () => {
    if (navMediaQuery.matches) {
      return;
    }
    siteNav.dataset.open = "true";
    body.classList.add(NAV_OPEN_CLASS);
    if (navToggle) {
      navToggle.setAttribute("aria-expanded", "true");
      navToggle.classList.add("is-active");
    }
  };

  const toggleNav = () => {
    if (navMediaQuery.matches) {
      return;
    }
    if (siteNav.dataset.open === "true") {
      closeNav();
    } else {
      openNav();
    }
  };

  if (!navMediaQuery.matches) {
    siteNav.dataset.open = "false";
  }

  syncNavForViewport(navMediaQuery.matches);
  navMediaQuery.addEventListener("change", (event) => {
    if (event.matches) {
      syncNavForViewport(true);
    } else {
      closeNav();
      siteNav.dataset.open = "false";
      syncNavForViewport(false);
    }
  });

  const navLinks = siteNav.querySelectorAll(".nav-link");
  navLinks.forEach((link) => {
    const anchorId = getAnchorTarget(link);
    link.addEventListener("click", (event) => {
      if (!navMediaQuery.matches) {
        closeNav();
      }
      if (anchorId && smoothScrollToId(anchorId)) {
        event.preventDefault();
      }
    });
  });

  if (navToggle) {
    navToggle.addEventListener("click", () => {
      toggleNav();
    });
  }

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeNav({ focusToggle: true });
    }
  });

  document.addEventListener("click", (event) => {
    if (navMediaQuery.matches || siteNav.dataset.open !== "true") {
      return;
    }
    const target = event.target;
    if (!target || typeof target.closest !== "function") {
      return;
    }
    if (target.closest(NAV_SELECTOR) || (navToggle && target.closest(TOGGLE_SELECTOR))) {
      return;
    }
    closeNav();
  });
}
