/**
 * router.js
 * ---------
 * Hash-based client-side router with anime.js page transitions.
 *
 * Why a router at all: the console holds uploaded imagery and a rendered
 * result. A full page load would throw both away every time someone checked
 * the model status or the guide, so navigation is done in-place and the
 * transition is what tells you the page changed.
 *
 * Motion budget: one directional wipe per navigation, plus a short stagger on
 * the incoming panels. Nothing loops, nothing animates on hover. Under
 * prefers-reduced-motion the transitions collapse to instant swaps — the
 * router still works, it just stops moving.
 */

import { animate, stagger } from './anime.esm.min.js';

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/** Page order, used to decide whether a transition moves forward or back. */
const ORDER = ['/', '/console', '/model', '/guide'];

export class Router {
  /**
   * @param {object} options
   * @param {string} options.pageSelector  selector matching every page section
   * @param {string} options.linkSelector  selector matching in-app links
   * @param {(path: string) => void} [options.onEnter] called after a page shows
   */
  constructor({ pageSelector, linkSelector, onEnter }) {
    this.pages = Array.from(document.querySelectorAll(pageSelector));
    this.linkSelector = linkSelector;
    this.onEnter = onEnter || (() => {});
    this.current = null;
    this.busy = false;
  }

  start() {
    document.addEventListener('click', (event) => {
      const link = event.target.closest(this.linkSelector);
      if (!link) return;
      const href = link.getAttribute('href') || '';
      if (!href.startsWith('#/')) return;
      event.preventDefault();
      this.go(href.slice(1));
    });

    window.addEventListener('hashchange', () => this.go(this.pathFromHash(), false));
    this.go(this.pathFromHash(), false);
  }

  pathFromHash() {
    const raw = (window.location.hash || '#/').slice(1);
    const path = raw.split('?')[0] || '/';
    return this.pages.some((page) => page.dataset.page === path) ? path : '/';
  }

  /**
   * Navigate to a path.
   * @param {string} path
   * @param {boolean} [pushHash=true] update location.hash
   */
  async go(path, pushHash = true) {
    if (this.busy || path === this.current) {
      if (pushHash && window.location.hash !== `#${path}`) {
        window.location.hash = path;
      }
      return;
    }

    const next = this.pages.find((page) => page.dataset.page === path);
    if (!next) return;

    this.busy = true;
    const previous = this.pages.find((page) => page.dataset.page === this.current);

    // Direction is derived from page order so navigating "back" reads as back.
    const from = ORDER.indexOf(this.current);
    const to = ORDER.indexOf(path);
    const forward = from === -1 || to >= from;
    const shift = forward ? 26 : -26;

    if (previous && !REDUCED) {
      await animate(previous, {
        opacity: [1, 0],
        translateY: [0, -10],
        duration: 180,
        ease: 'inQuad',
      }).then();
    }

    this.pages.forEach((page) => page.classList.remove('is-active'));
    next.classList.add('is-active');
    if (previous) previous.style.opacity = '';
    window.scrollTo({ top: 0, behavior: REDUCED ? 'auto' : 'smooth' });

    this.syncNav(path);
    this.current = path;
    if (pushHash && window.location.hash !== `#${path}`) {
      window.location.hash = path;
    }

    if (REDUCED) {
      next.style.opacity = '';
      this.onEnter(path);
      this.busy = false;
      return;
    }

    await animate(next, {
      opacity: [0, 1],
      translateX: [shift, 0],
      duration: 300,
      ease: 'out(3)',
    }).then();
    next.style.transform = '';

    this.revealChildren(next);
    this.onEnter(path);
    this.busy = false;
  }

  /** Stagger the incoming page's panels so the layout assembles rather than pops. */
  revealChildren(page) {
    const items = page.querySelectorAll('[data-anim="stagger"], [data-anim="hero"]');
    if (!items.length) return;
    animate(items, {
      opacity: [0, 1],
      translateY: [14, 0],
      duration: 420,
      delay: stagger(55),
      ease: 'out(3)',
    });
  }

  syncNav(path) {
    document.querySelectorAll(`${this.linkSelector}`).forEach((link) => {
      const href = link.getAttribute('href') || '';
      if (href.slice(1) === path) {
        link.setAttribute('aria-current', 'page');
      } else {
        link.removeAttribute('aria-current');
      }
    });
  }
}

/**
 * The landing page's orbital scan: satellite crosses the limb, beams sweep
 * down, ground nodes light up in sequence. Runs once, on first load only.
 */
export function playOrbitalScan() {
  if (REDUCED) return;

  const limb = document.getElementById('scan-limb');
  const swath = document.getElementById('scan-swath');
  const terrain = document.getElementById('scan-terrain');
  const sat = document.getElementById('scan-sat');
  const beams = document.querySelectorAll('#scan-beams line');
  const nodes = document.querySelectorAll('#scan-nodes circle');
  if (!limb || !sat) return;

  // Draw the arcs by animating their dash offset — the classic SVG line-draw,
  // done manually so we do not depend on anime's svg helper resolving here.
  [limb, swath, terrain].forEach((path) => {
    if (!path) return;
    const length = path.getTotalLength();
    path.style.strokeDasharray = `${length}`;
    path.style.strokeDashoffset = `${length}`;
    animate(path, {
      strokeDashoffset: [length, 0],
      duration: 1100,
      ease: 'out(2)',
    });
  });

  animate(sat, {
    translateX: [-120, 0],
    opacity: [0, 1],
    duration: 900,
    delay: 260,
    ease: 'out(3)',
  });

  animate(beams, {
    opacity: [0, 0.35],
    duration: 460,
    delay: stagger(110, { start: 700 }),
    ease: 'out(2)',
  });

  animate(nodes, {
    scale: [0, 1],
    opacity: [0, 1],
    duration: 520,
    delay: stagger(110, { start: 780 }),
    ease: 'out(4)',
  });
}
