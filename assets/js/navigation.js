// Collapses header on scroll and toggles mobile menu.

const header = document.getElementById('site-header');
const toggleBtn = document.getElementById('nav-toggle');
const nav = document.getElementById('main-nav');
let lastScrollY = window.scrollY;

window.addEventListener('scroll', () => {
  const currentY = window.scrollY;
  if (currentY > lastScrollY && currentY > 50) {
    header.classList.replace('header--expanded', 'header--collapsed');
  } else {
    header.classList.replace('header--collapsed', 'header--expanded');
  }
  lastScrollY = currentY;
});

toggleBtn.addEventListener('click', () => {
  const isOpen = nav.classList.toggle('is-open');
  toggleBtn.classList.toggle('is-active');
  toggleBtn.setAttribute('aria-expanded', isOpen);
});
