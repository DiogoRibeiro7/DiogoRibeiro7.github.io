/**
 * @fileoverview MathJax enhancements for LaTeX rendering.
 * Provides equation numbering, copy-to-clipboard, live editor, and accessibility features.
 * @module math
 */

(function () {
  /**
   * MathToolkit object containing all math enhancement functionality.
   * @namespace MathToolkit
   */
  const MathToolkit = {
    init(MathJax) {
      this.MathJax = MathJax;
      this.displayCounter = 0;
      this.anchorCounter = 0;
      this.equationMap = new Map();
      this.editor = null;
      this.editorOpen = false;
      this.messageTimeout = null;
      this.initialized = true;
      this.pendingTypeset = [];

      this.setupEditor();

      MathJax.startup.promise
        .then(() => {
          this.decorateExisting(MathJax.startup.document.math || []);
          this.updateEquationReferences();
          if (this.pendingTypeset.length > 0) {
            const pending = [...this.pendingTypeset];
            this.pendingTypeset = [];
            pending.forEach((item) => {
              this.renderLatex(item.target, item.latex, item.options || {});
            });
          }
        })
        .catch((error) => {
          console.error('Failed to initialize MathJax enhancements', error);
        });

      document.addEventListener('click', (event) => this.handleClick(event));
      document.addEventListener('keydown', (event) => this.handleKeydown(event));
    },

    onPageReady() {
      this.updateEquationReferences();
    },

    decorateExisting(mathItems) {
      if (!mathItems || mathItems.length === 0) {
        return;
      }

      this.displayCounter = 0;
      this.anchorCounter = 0;
      this.equationMap.clear();

      mathItems.forEach((item) => {
        try {
          this.decorateMathItem(item);
        } catch (error) {
          console.warn('Unable to decorate math item', error);
        }
      });
    },

    decorateMathItem(item) {
      if (!item || !item.typesetRoot) {
        return;
      }

      const container = item.typesetRoot;
      const latex = item.math || '';
      const wrapper = container.closest('[data-math-alt]') || container;

      const altText = this.resolveAltText(wrapper, latex);

      container.dataset.mathLatex = latex;
      container.setAttribute('tabindex', '0');
      container.setAttribute('role', 'math');

      this.syncAltAttributes(wrapper, container, altText);

      if (item.display) {
        this.decorateDisplay(container, latex);
      } else {
        this.decorateInline(container, latex);
      }
    },

    decorateInline(container, latex) {
      if (!container) {
        return;
      }
      if (!container.classList.contains('math-expression__inline')) {
        container.classList.add('math-expression__inline');
      }
      container.dataset.mathLatexClean = this.cleanLatex(latex);
    },

    decorateDisplay(container, latex) {
      if (!container) {
        return;
      }

      const existingWrapper = container.parentElement && container.parentElement.classList.contains('math-expression')
        ? container.parentElement
        : null;
      let wrapper = existingWrapper;

      if (!wrapper) {
        wrapper = document.createElement('figure');
        wrapper.className = 'math-expression math-expression--display';
        container.parentNode.insertBefore(wrapper, container);
        wrapper.appendChild(container);
      }

      const info = this.extractEquationInfo(latex);
      this.anchorCounter += 1;
      const fallbackAnchor = `eq:auto-${this.anchorCounter}`;
      let equationNumber = info.tag || '';

      if (!info.skipNumber) {
        this.displayCounter += 1;
        if (!equationNumber) {
          equationNumber = String(this.displayCounter);
        }
      }

      const formattedNumber = equationNumber ? `(${equationNumber})` : '';
      const anchorId = this.buildAnchorId(info.label, fallbackAnchor);

      wrapper.dataset.mathLatex = latex;
      wrapper.dataset.mathLatexClean = this.cleanLatex(latex);
      if (container.dataset.mathAlt && !wrapper.dataset.mathAlt) {
        wrapper.dataset.mathAlt = container.dataset.mathAlt;
      }
      if (formattedNumber) {
        wrapper.dataset.equationNumber = formattedNumber;
      } else {
        wrapper.removeAttribute('data-equation-number');
      }
      wrapper.dataset.equationLabel = info.label || '';

      if (anchorId) {
        wrapper.id = anchorId;
        const displayText = formattedNumber || info.label || anchorId;
        if (formattedNumber || info.label) {
          this.setEquationReference(anchorId, displayText);
          if (info.label && info.label !== anchorId) {
            this.setEquationReference(info.label, displayText, anchorId);
          }
          if (info.label && info.label.startsWith('eq:')) {
            const plainLabel = info.label.replace(/^eq:/, '');
            this.setEquationReference(plainLabel, displayText, anchorId);
          }
        }
      }

      this.attachNumberBadge(wrapper, formattedNumber);
      this.attachToolbar(wrapper);
    },

    extractEquationInfo(latex) {
      const source = (latex || '').trim();
      const tagMatch = source.match(/\\tag\*?\{([^}]*)\}/);
      const labelMatch = source.match(/\\label\{([^}]*)\}/);
      const skipNumber = /\\notag|\\nonumber/.test(source);

      return {
        tag: tagMatch ? tagMatch[1].trim() : '',
        label: labelMatch ? labelMatch[1].trim() : '',
        skipNumber
      };
    },

    buildAnchorId(label, fallback) {
      if (label) {
        return label.startsWith('eq:') ? label : `eq:${label}`;
      }
      if (fallback) {
        return fallback;
      }
      return '';
    },

    setEquationReference(key, displayText, anchor) {
      if (!key) {
        return;
      }
      const target = anchor || key;
      const numberText = displayText || key;
      this.equationMap.set(key, {
        anchor: target,
        label: key,
        number: numberText
      });
      if (!key.startsWith('eq:')) {
        this.equationMap.set(`eq:${key}`, {
          anchor: target,
          label: key,
          number: numberText
        });
      }
    },

    attachNumberBadge(wrapper, formattedNumber) {
      let badge = wrapper.querySelector('.math-expression__number');
      let sr = wrapper.querySelector('.math-expression__sr');

      if (!formattedNumber) {
        if (badge) {
          badge.remove();
        }
        if (sr) {
          sr.remove();
        }
        return;
      }

      if (!badge) {
        badge = document.createElement('span');
        badge.className = 'math-expression__number';
        badge.setAttribute('aria-hidden', 'true');
        wrapper.appendChild(badge);
      }
      badge.textContent = formattedNumber;

      if (!sr) {
        sr = document.createElement('span');
        sr.className = 'visually-hidden math-expression__sr';
        wrapper.appendChild(sr);
      }
      sr.textContent = `Equation ${formattedNumber.replace(/[()]/g, '')}`;
    },

    attachToolbar(wrapper) {
      let toolbar = wrapper.querySelector('.math-expression__toolbar');

      if (!toolbar) {
        toolbar = document.createElement('div');
        toolbar.className = 'math-expression__toolbar';

        const copyButton = document.createElement('button');
        copyButton.type = 'button';
        copyButton.className = 'math-expression__button';
        copyButton.setAttribute('data-math-copy', 'true');
        copyButton.setAttribute('aria-label', 'Copy LaTeX code');
        const copyIcon = document.createElement('span');
        copyIcon.setAttribute('aria-hidden', 'true');
        copyIcon.textContent = '⧉';
        const copySr = document.createElement('span');
        copySr.className = 'visually-hidden';
        copySr.textContent = 'Copy equation';
        copyButton.appendChild(copyIcon);
        copyButton.appendChild(copySr);

        const editButton = document.createElement('button');
        editButton.type = 'button';
        editButton.className = 'math-expression__button';
        editButton.setAttribute('data-math-edit', 'true');
        editButton.setAttribute('aria-label', 'Open equation editor');
        const editIcon = document.createElement('span');
        editIcon.setAttribute('aria-hidden', 'true');
        editIcon.textContent = '✎';
        const editSr = document.createElement('span');
        editSr.className = 'visually-hidden';
        editSr.textContent = 'Edit equation';
        editButton.appendChild(editIcon);
        editButton.appendChild(editSr);

        const message = document.createElement('span');
        message.className = 'math-expression__message';
        message.setAttribute('aria-live', 'polite');
        message.hidden = true;

        toolbar.appendChild(copyButton);
        toolbar.appendChild(editButton);
        toolbar.appendChild(message);
        wrapper.appendChild(toolbar);
      }
    },

    handleClick(event) {
      const copyTrigger = event.target.closest('[data-math-copy]');
      if (copyTrigger) {
        event.preventDefault();
        const wrapper = copyTrigger.closest('.math-expression');
        const latex = wrapper ? wrapper.dataset.mathLatexClean || wrapper.dataset.mathLatex : null;
        this.copyLatex(latex, wrapper);
        return;
      }

      const editTrigger = event.target.closest('[data-math-edit]');
      if (editTrigger) {
        event.preventDefault();
        const wrapper = editTrigger.closest('.math-expression');
        const latex = wrapper ? wrapper.dataset.mathLatex || '' : '';
        this.openEditor(latex);
        return;
      }

      const closeTrigger = event.target.closest('[data-math-editor-close]');
      if (closeTrigger) {
        event.preventDefault();
        this.closeEditor();
        return;
      }

      const copyEditor = event.target.closest('[data-math-editor-copy]');
      if (copyEditor) {
        event.preventDefault();
        if (this.editor && this.editor.textarea) {
          this.copyLatex(this.editor.textarea.value, null, this.editor.status);
        }
        return;
      }

      const symbolTrigger = event.target.closest('[data-symbol-insert]');
      if (symbolTrigger) {
        event.preventDefault();
        const value = symbolTrigger.getAttribute('data-symbol-insert');
        this.insertSymbol(value || '');
      }
    },

    handleKeydown(event) {
      const active = document.activeElement;
      const typingContext =
        active &&
        (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable);

      if ((event.ctrlKey || event.metaKey) && event.altKey && (event.key === 'e' || event.key === 'E')) {
        event.preventDefault();
        const seedLatex = typingContext && active ? active.value || '' : '';
        this.openEditor(seedLatex);
        return;
      }

      if (event.key === 'Escape' && this.editorOpen) {
        event.preventDefault();
        this.closeEditor();
      }
    },

    copyLatex(latex, wrapper, statusTarget) {
      if (!latex) {
        return;
      }

      const showStatus = (message) => {
        if (statusTarget) {
          statusTarget.hidden = false;
          statusTarget.textContent = message;
          clearTimeout(this.messageTimeout);
          this.messageTimeout = window.setTimeout(() => {
            statusTarget.hidden = true;
            statusTarget.textContent = '';
          }, 2000);
        } else if (wrapper) {
          const toolbar = wrapper.querySelector('.math-expression__message');
          if (toolbar) {
            toolbar.hidden = false;
            toolbar.textContent = message;
            clearTimeout(this.messageTimeout);
            this.messageTimeout = window.setTimeout(() => {
              toolbar.hidden = true;
              toolbar.textContent = '';
            }, 2000);
          }
        }
      };

      const fallbackCopy = () => {
        const temp = document.createElement('textarea');
        temp.value = latex;
        temp.setAttribute('aria-hidden', 'true');
        temp.style.position = 'fixed';
        temp.style.opacity = '0';
        document.body.appendChild(temp);
        temp.select();
        try {
          document.execCommand('copy');
          showStatus('Copied equation');
        } catch (error) {
          console.warn('Unable to copy equation', error);
          showStatus('Unable to copy');
        }
        document.body.removeChild(temp);
      };

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard
          .writeText(latex)
          .then(() => showStatus('Copied equation'))
          .catch(() => fallbackCopy());
      } else {
        fallbackCopy();
      }
    },

    setupEditor() {
      if (this.editor) {
        return;
      }

      const overlay = document.createElement('div');
      overlay.id = 'math-tooling-panel';
      overlay.hidden = true;

      const dialog = document.createElement('div');
      dialog.className = 'math-tooling';
      dialog.setAttribute('role', 'dialog');
      dialog.setAttribute('aria-modal', 'true');
      dialog.setAttribute('aria-labelledby', 'math-tooling-title');

      const title = document.createElement('h2');
      title.id = 'math-tooling-title';
      title.className = 'math-tooling__title';
      title.textContent = 'Equation editor';

      const helper = document.createElement('p');
      helper.className = 'math-tooling__description';
      // innerHTML required for MathJax rendering
      helper.innerHTML =
        'Craft LaTeX expressions with live preview, quick statistical symbols, and chemistry support.';

      const textarea = document.createElement('textarea');
      textarea.className = 'math-tooling__input';
      textarea.setAttribute('data-math-editor-input', 'true');
      textarea.setAttribute('spellcheck', 'false');
      textarea.rows = 10;

      const actions = document.createElement('div');
      actions.className = 'math-tooling__actions';

      const preview = document.createElement('div');
      preview.className = 'math-tooling__preview';
      preview.setAttribute('data-math-editor-preview', 'true');
      preview.setAttribute('aria-live', 'polite');

      const status = document.createElement('span');
      status.className = 'math-tooling__status';
      status.hidden = true;

      const library = document.createElement('div');
      library.className = 'math-tooling__library';

      const libraryTitle = document.createElement('h3');
      libraryTitle.className = 'math-tooling__library-title';
      libraryTitle.textContent = 'Quick insert';

      const symbolList = document.createElement('div');
      symbolList.className = 'math-tooling__symbols';

      const symbols = [
        { label: 'Mean μ', value: '\\mu' },
        { label: 'Variance σ²', value: '\\sigma^2' },
        { label: 'Std. dev σ', value: '\\sigma' },
        { label: 'Probability', value: '\\mathbb{P}(A)' },
        { label: 'Expectation', value: '\\mathbb{E}[X]' },
        { label: 'Covariance', value: '\\mathrm{Cov}(X, Y)' },
        { label: 'Vector', value: '\\mathbf{x}' },
        { label: 'Norm', value: '\\lVert x \\rVert_2' },
        { label: 'Inner product', value: '\\langle x, y \\rangle' },
        { label: 'Derivative', value: '\\frac{\\mathrm{d}}{\\mathrm{d}t}' },
        { label: 'Integral', value: '\\int_0^1 f(x) \\, \\mathrm{d}x' },
        { label: 'Bayes', value: '\\mathbb{P}(A \\mid B)' },
        { label: 'Chemical H₂O', value: '\\ce{H2O}' },
        { label: 'Chemical CO₂', value: '\\ce{CO2}' }
      ];

      symbols.forEach((symbol) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'math-tooling__symbol';
        button.setAttribute('data-symbol-insert', symbol.value);
        button.textContent = symbol.label;
        symbolList.appendChild(button);
      });

      const closeButton = document.createElement('button');
      closeButton.type = 'button';
      closeButton.className = 'math-tooling__close';
      closeButton.setAttribute('data-math-editor-close', 'true');
      const closeIcon = document.createElement('span');
      closeIcon.setAttribute('aria-hidden', 'true');
      closeIcon.textContent = '✕';
      const closeSr = document.createElement('span');
      closeSr.className = 'visually-hidden';
      closeSr.textContent = 'Close equation editor';
      closeButton.appendChild(closeIcon);
      closeButton.appendChild(closeSr);

      const copyButton = document.createElement('button');
      copyButton.type = 'button';
      copyButton.className = 'math-tooling__copy';
      copyButton.setAttribute('data-math-editor-copy', 'true');
      copyButton.textContent = 'Copy LaTeX';

      actions.appendChild(copyButton);
      actions.appendChild(closeButton);
      actions.appendChild(status);

      library.appendChild(libraryTitle);
      library.appendChild(symbolList);

      dialog.appendChild(title);
      dialog.appendChild(helper);
      dialog.appendChild(textarea);
      dialog.appendChild(preview);
      dialog.appendChild(actions);
      dialog.appendChild(library);

      overlay.appendChild(dialog);
      document.body.appendChild(overlay);

      textarea.addEventListener('input', () => this.updateEditorPreview());

      this.editor = {
        overlay,
        dialog,
        textarea,
        preview,
        status
      };
    },

    openEditor(latex) {
      if (!this.editor) {
        return;
      }

      const value = (latex || '').trim();
      this.editor.overlay.hidden = false;
      document.body.classList.add('math-tooling-open');
      this.editor.textarea.value = value;
      this.editor.textarea.focus();
      this.editor.textarea.setSelectionRange(value.length, value.length);
      this.editorOpen = true;
      this.updateEditorPreview();
    },

    closeEditor() {
      if (!this.editor) {
        return;
      }
      this.editor.overlay.hidden = true;
      document.body.classList.remove('math-tooling-open');
      this.editorOpen = false;
    },

    updateEditorPreview() {
      if (!this.editor || !this.MathJax) {
        return;
      }
      const latex = this.editor.textarea.value.trim();
      if (!latex) {
        this.editor.preview.replaceChildren();
        const placeholder = document.createElement('p');
        placeholder.className = 'math-tooling__placeholder';
        placeholder.textContent = 'Preview will appear here.';
        this.editor.preview.appendChild(placeholder);
        return;
      }
      this.renderLatex(this.editor.preview, latex, { display: true, enhance: false });
    },

    insertSymbol(value) {
      if (!this.editor || !this.editor.textarea) {
        return;
      }
      const textarea = this.editor.textarea;
      const start = textarea.selectionStart || 0;
      const end = textarea.selectionEnd || 0;
      const current = textarea.value;
      const next = `${current.slice(0, start)}${value}${current.slice(end)}`;
      textarea.value = next;
      const caret = start + value.length;
      textarea.focus();
      textarea.setSelectionRange(caret, caret);
      this.updateEditorPreview();
    },

    updateEquationReferences() {
      if (!this.equationMap || this.equationMap.size === 0) {
        return;
      }

      const references = document.querySelectorAll('[data-equation-ref]');
      references.forEach((ref) => {
        const key = ref.getAttribute('data-equation-ref');
        if (!key) {
          return;
        }
        const lookup = this.lookupEquation(key.replace(/^#/, ''));
        if (!lookup) {
          return;
        }
        ref.setAttribute('href', `#${lookup.anchor}`);
        const text = ref.textContent.trim();
        if (!text || text === key || text === lookup.anchor || text === `(${key})`) {
          ref.textContent = lookup.number || lookup.anchor;
        }
        ref.classList.add('math-reference');
        ref.setAttribute('aria-label', `Jump to equation ${lookup.number || lookup.anchor}`);
      });

      const anchorRefs = document.querySelectorAll('a[href^="#eq:"]');
      anchorRefs.forEach((link) => {
        const href = link.getAttribute('href');
        if (!href) {
          return;
        }
        const key = href.replace(/^#/, '');
        const lookup = this.lookupEquation(key);
        if (!lookup) {
          return;
        }
        link.setAttribute('href', `#${lookup.anchor}`);
        const text = link.textContent.trim();
        if (!text || text === href || text === key) {
          link.textContent = lookup.number || lookup.anchor;
        }
        link.dataset.equationRef = lookup.anchor;
        link.classList.add('math-reference');
      });
    },

    lookupEquation(key) {
      if (!key) {
        return null;
      }
      if (this.equationMap.has(key)) {
        return this.equationMap.get(key);
      }
      const normalized = key.startsWith('eq:') ? key : `eq:${key}`;
      return this.equationMap.get(normalized) || null;
    },

    renderLatex(target, latex, options = {}) {
      if (!target) {
        return Promise.resolve();
      }
      const display = options.display !== false;
      if (!this.MathJax) {
        target.textContent = latex;
        if (!this.pendingTypeset) {
          this.pendingTypeset = [];
        }
        this.pendingTypeset.push({ target, latex, options });
        return Promise.resolve();
      }
      const wrapperLatex = display ? `\\[${latex}\\]` : `\\(${latex}\\)`;
      // innerHTML required for MathJax rendering
      target.innerHTML = wrapperLatex;
      return this.MathJax.typesetPromise([target]).catch((error) => {
        console.warn('MathJax failed to typeset preview', error);
      });
    },

    cleanLatex(latex) {
      return (latex || '')
        .replace(/%\s*alt:.*$/gm, '')
        .replace(/\\label\{[^}]*\}/g, '')
        .replace(/\\tag\*?\{[^}]*\}/g, '')
        .trim();
    },

    resolveAltText(wrapper, latex) {
      const sourceNode = wrapper || null;
      const datasetAlt = this.findDatasetAlt(sourceNode);
      if (datasetAlt) {
        return datasetAlt;
      }

      const commentAlt = this.extractAltComment(latex);
      if (commentAlt) {
        return commentAlt;
      }

      const sourceLatex = sourceNode && sourceNode.getAttribute ? sourceNode.getAttribute('data-math-source') : null;
      const targetLatex = sourceLatex || latex;
      return this.generateAltFromLatex(targetLatex);
    },

    findDatasetAlt(node) {
      let current = node;
      while (current) {
        if (current.dataset && current.dataset.mathAlt) {
          const alt = current.dataset.mathAlt.trim();
          if (alt) {
            return alt;
          }
        }
        current = current.parentElement;
      }
      return '';
    },

    extractAltComment(latex) {
      if (!latex) {
        return '';
      }
      const lines = String(latex).split(/\r?\n/);
      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        if (!line) {
          continue;
        }
        const match = line.match(/%\s*alt:\s*(.+)$/i);
        if (match) {
          return match[1].trim();
        }
      }
      return '';
    },

    generateAltFromLatex(latex) {
      if (!latex) {
        return '';
      }
      let text = String(latex).replace(/%.*$/gm, '');

      text = text.replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, (_match, numerator, denominator) => {
        const top = this.sanitizeSegment(numerator);
        const bottom = this.sanitizeSegment(denominator);
        return `${top} over ${bottom}`.trim();
      });

      text = text.replace(/\\int(?:_\{([^}]*)\}|_([^\s^{}]+))?(?:\^\{([^}]*)\}|\^([^\s_{}]+))?/g, (_match, lowerBraced, lowerSimple, upperBraced, upperSimple) => {
        const lower = this.sanitizeSegment(lowerBraced || lowerSimple);
        const upper = this.sanitizeSegment(upperBraced || upperSimple);
        let phrase = 'integral';
        if (lower) {
          phrase += ` from ${lower}`;
        }
        if (upper) {
          phrase += ` to ${upper}`;
        }
        return phrase;
      });

      text = text.replace(/\\sum(?:_\{([^}]*)\}|_([^\s^{}]+))?(?:\^\{([^}]*)\}|\^([^\s_{}]+))?/g, (_match, lowerBraced, lowerSimple, upperBraced, upperSimple) => {
        const lower = this.sanitizeSegment(lowerBraced || lowerSimple);
        const upper = this.sanitizeSegment(upperBraced || upperSimple);
        let phrase = 'summation';
        if (lower) {
          phrase += ` from ${lower}`;
        }
        if (upper) {
          phrase += ` to ${upper}`;
        }
        return phrase;
      });

      text = text.replace(/\\sqrt\s*\{([^{}]+)\}/g, (_match, radicand) => `square root of ${this.sanitizeSegment(radicand)}`);
      text = text.replace(/\\mathrm\s*\{([^{}]+)\}/g, (_match, content) => this.sanitizeSegment(content));
      text = text.replace(/\\operatorname\*?\s*\{([^{}]+)\}/g, (_match, content) => this.sanitizeSegment(content));
      text = text.replace(/\\[a-zA-Z]+/g, ' ');
      text = text.replace(/[{}]/g, ' ');
      text = text.replace(/\s+/g, ' ').trim();

      if (!text) {
        return 'Mathematical expression';
      }
      return text;
    },

    sanitizeSegment(segment) {
      if (!segment) {
        return '';
      }
      return String(segment)
        .replace(/\\[a-zA-Z]+/g, ' ')
        .replace(/[{}]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    },

    syncAltAttributes(wrapper, container, altText) {
      if (!altText) {
        return;
      }
      if (wrapper && wrapper.setAttribute) {
        wrapper.setAttribute('aria-label', altText);
        if (wrapper.dataset) {
          wrapper.dataset.mathAlt = altText;
        }
      }
      if (container && container.setAttribute) {
        container.setAttribute('aria-label', altText);
        if (container.dataset) {
          container.dataset.mathAlt = altText;
        }
      }
    }
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.__DATALOG_MATH_INTERNALS__ = MathToolkit;
  }

  window.DatalogMath = {
    init(MathJax) {
      MathToolkit.init(MathJax);
    },
    onPageReady() {
      if (MathToolkit.initialized) {
        MathToolkit.onPageReady();
      }
    },
    renderLatex(target, latex, options) {
      return MathToolkit.renderLatex(target, latex, options);
    }
  };
})();
