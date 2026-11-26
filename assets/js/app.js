/* hplugins: static catalog renderer and hash router. No dependencies. */
(function () {
  "use strict";

  var INDEX_URL = "data/search-index.json";
  var CATALOG_URL = "data/catalog.json";
  var view = document.getElementById("view");
  var dossierView = document.getElementById("dossier-view");
  var typeNav = document.getElementById("type-nav");
  var total = document.getElementById("entry-total");
  var themeButton = document.getElementById("theme-button");
  var themeLabel = document.getElementById("theme-label");

  var state = {
    index: [],
    filter: "all",
    query: "",
    sort: "index",
    previewSlug: null,
    catalogPromise: null
  };

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function pad(number) {
    return String(number).padStart(2, "0");
  }

  function safeUrl(url) {
    if (/^https:\/\//.test(url) || /^[A-Za-z0-9_./-]+$/.test(url)) return esc(url);
    return "#/";
  }

  function filtered() {
    var query = state.query.trim().toLowerCase();
    var items = state.index.filter(function (item) {
      if (state.filter !== "all" && item.type !== state.filter) return false;
      return !query || item.search.indexOf(query) !== -1;
    });
    if (state.sort === "name") {
      items = items.slice().sort(function (a, b) { return a.name.localeCompare(b.name); });
    } else if (state.sort === "author") {
      items = items.slice().sort(function (a, b) { return a.author.localeCompare(b.author); });
    }
    return items;
  }

  function getCatalog() {
    if (!state.catalogPromise) {
      state.catalogPromise = fetch(CATALOG_URL).then(function (response) {
        if (!response.ok) throw new Error("catalog request failed: " + response.status);
        return response.json();
      });
    }
    return state.catalogPromise;
  }

  function parseRoute() {
    var hash = window.location.hash || "#/";
    var match = hash.match(/^#\/p\/([a-z0-9-]+)$/);
    return match ? { kind: "detail", slug: match[1] } : { kind: "list" };
  }

  function stamp(status) {
    var filled = status === "available" ? " is-filled" : "";
    return '<span class="stamp"><span class="stamp-mark' + filled + '" aria-hidden="true"></span>' +
      esc(status) + "</span>";
  }

  function renderTypeNav() {
    var counts = {};
    state.index.forEach(function (item) { counts[item.type] = (counts[item.type] || 0) + 1; });
    var keys = ["all"].concat(Object.keys(counts).sort());
    typeNav.innerHTML = keys.map(function (key) {
      var count = key === "all" ? state.index.length : counts[key];
      return '<button class="type-filter" type="button" data-type="' + esc(key) + '" aria-pressed="' +
        (state.filter === key ? "true" : "false") + '"><span>' + esc(key === "all" ? "all parts" : key) +
        '</span><span class="type-count">' + pad(count) + "</span></button>";
    }).join("");
    total.textContent = pad(state.index.length);
  }

  function setFilter(type) {
    state.filter = type;
    renderTypeNav();
    if (parseRoute().kind !== "list") window.location.hash = "#/";
    else updateList();
  }

  function updateList() {
    var list = document.getElementById("entry-list");
    var counter = document.getElementById("result-count");
    if (!list || !counter) return;

    var items = filtered();
    counter.textContent = pad(items.length) + " shown / " + pad(state.index.length) + " total";
    if (!items.length) {
      list.innerHTML = '<li class="notice"><h2>No parts match</h2><p>Try a different search term or clear the type filter.</p>' +
        '<button class="text-button" type="button" id="clear-filters">Clear filters</button></li>';
      document.getElementById("clear-filters").addEventListener("click", function () {
        state.query = ""; state.filter = "all";
        var search = document.getElementById("search-input");
        if (search) search.value = "";
        renderTypeNav(); updateList();
      });
      return;
    }

    list.innerHTML = items.map(function (item, i) {
      return '<li><a class="entry-row" style="--i:' + Math.min(i, 12) + '" href="#/p/' + esc(item.slug) +
        '" data-slug="' + esc(item.slug) + '">' +
        '<span class="entry-index">' + pad(i + 1) + '</span>' +
        '<span class="entry-body"><span class="entry-name">' + esc(item.name) + '</span>' +
        '<span class="entry-summary">' + esc(item.summary) + '</span></span>' +
        '<span class="entry-meta"><span>/' + esc(item.type) + '</span>' + stamp(item.status) + "</span></a></li>";
    }).join("");

    if (state.previewSlug && !items.some(function (item) { return item.slug === state.previewSlug; })) {
      state.previewSlug = null;
    }
    if (state.previewSlug) previewDossier(state.previewSlug);
    else renderEmptyDossier();
  }

  function renderEmptyDossier() {
    dossierView.innerHTML =
      '<p class="dossier-kicker">Dossier</p>' +
      '<h2 class="dossier-name">Nothing selected yet.</h2>' +
      '<p class="dossier-summary">Hover a row to preview it here. Open a part to read its guide and copy its setup prompt.</p>';
  }

  function previewDossier(slug) {
    var item = state.index.find(function (entry) { return entry.slug === slug; });
    if (!item) return;
    state.previewSlug = slug;
    var head =
      '<p class="dossier-kicker">Selected part</p>' +
      '<h2 class="dossier-name">' + esc(item.name) + "</h2>" +
      '<p class="dossier-meta">/' + esc(item.type) + " - " + esc(item.author) + " - " + esc(item.status) + "</p>" +
      '<p class="dossier-summary">' + esc(item.summary) + "</p>";
    dossierView.innerHTML = head + '<p class="dossier-kicker">Loading setup prompt...</p>';

    getCatalog().then(function (catalog) {
      if (state.previewSlug !== slug || parseRoute().kind !== "list") return;
      var entry = catalog.find(function (record) { return record.slug === slug; });
      if (!entry) return;
      dossierView.innerHTML = head +
        '<ul class="dossier-links">' + entry.where_to_find.map(function (link) {
          return '<li><a href="' + safeUrl(link.url) + '" rel="noopener noreferrer">' +
            esc(link.label) + "</a></li>";
        }).join("") + "</ul>" +
        '<a class="open-button" href="#/p/' + esc(slug) + '">Read the full guide</a>' +
        manifest(entry);
      bindCopy(entry);
    }).catch(function () {
      if (state.previewSlug === slug && parseRoute().kind === "list") {
        dossierView.innerHTML = head + '<p class="manifest-note">The setup prompt is unavailable right now. ' +
          '<a href="#/p/' + esc(slug) + '">Open the dossier</a> to retry.</p>';
      }
    });
  }

  function renderCatalog() {
    document.title = "hplugins - community Heretic plugins";
    view.innerHTML =
      '<header class="catalog-head"><h1 class="catalog-title">Do more with Heretic.</h1>' +
      '<p class="catalog-intro">Community-built scorers, modifiers, and configs. Every entry has a source, a guide, ' +
      'and a prompt your coding model can follow.</p></header>' +
      '<div class="toolbar">' +
      '<div class="field"><label class="field-label" for="search-input">Search the index</label>' +
      '<input id="search-input" type="search" autocomplete="off" placeholder="krea2, gibberish, classifier..."></div>' +
      '<div class="field"><label class="field-label" for="sort-select">Order</label>' +
      '<select id="sort-select"><option value="index">Index</option><option value="name">Name</option>' +
      '<option value="author">Author</option></select></div>' +
      '<button class="text-button" type="button" id="reset-button">Reset</button></div>' +
      '<div class="results-line"><span id="result-count"></span><span>select a part to see its setup prompt</span></div>' +
      '<ol class="entry-list" id="entry-list"></ol>';

    var search = document.getElementById("search-input");
    search.value = state.query;
    search.addEventListener("input", function () { state.query = search.value; updateList(); });

    document.getElementById("sort-select").addEventListener("change", function (event) {
      state.sort = event.target.value; updateList();
    });
    document.getElementById("reset-button").addEventListener("click", function () {
      state.query = ""; state.filter = "all"; state.sort = "index";
      search.value = "";
      document.getElementById("sort-select").value = "index";
      renderTypeNav(); updateList();
    });

    var list = document.getElementById("entry-list");
    list.addEventListener("mouseover", function (event) {
      var row = event.target.closest(".entry-row");
      if (row && row.dataset.slug !== state.previewSlug) previewDossier(row.dataset.slug);
    });
    list.addEventListener("focusin", function (event) {
      var row = event.target.closest(".entry-row");
      if (row) previewDossier(row.dataset.slug);
    });

    state.previewSlug = null;
    updateList();
  }

  function wrapText(text, width) {
    var output = [];
    text.split("\n").forEach(function (paragraph) {
      var line = "";
      paragraph.split(/\s+/).forEach(function (word) {
        if (!word) return;
        if (line && line.length + word.length + 1 > width) {
          output.push(line);
          line = word;
        } else {
          line = line ? line + " " + word : word;
        }
      });
      output.push(line);
    });
    return output;
  }

  function manifest(entry) {
    var lines = wrapText(entry.setup_prompt, 42);
    return '<div class="manifest-head"><p class="dossier-kicker">Setup prompt</p>' +
      '<span class="dossier-meta">' + pad(lines.length) + ' lines</span></div>' +
      '<div class="manifest-lines">' + lines.map(function (line, i) {
        return '<div class="manifest-line"><span class="line-no">' + pad(i + 1) +
          '</span><span class="line-text">' + esc(line || " ") + "</span></div>";
      }).join("") + "</div>" +
      '<button class="copy-button" type="button" id="copy-prompt"><span>Copy setup prompt</span>' +
      '<span aria-hidden="true">[ ]</span></button>' +
      '<p class="manifest-note">Paste this into a coding model in the Heretic checkout you use. ' +
      'It names the files, config keys, and a check that proves the setup loaded.</p>';
  }

  function bindCopy(entry) {
    var button = document.getElementById("copy-prompt");
    if (!button) return;
    button.addEventListener("click", function () {
      var text = entry.setup_prompt;
      function done(label, stateName) {
        button.firstElementChild.textContent = label;
        button.setAttribute("data-state", stateName);
        window.setTimeout(function () {
          if (!document.body.contains(button)) return;
          button.firstElementChild.textContent = "Copy setup prompt";
          button.setAttribute("data-state", "idle");
        }, 1800);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () { done("Copied", "done"); }, function () { fallback(); });
      } else {
        fallback();
      }
      function fallback() {
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "");
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        var ok = false;
        try { ok = document.execCommand("copy"); } catch (error) { ok = false; }
        document.body.removeChild(area);
        if (ok) done("Copied", "done");
        else {
          var block = document.querySelector(".manifest-lines");
          if (block && window.getSelection) {
            var range = document.createRange();
            range.selectNodeContents(block);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
          }
          done("Select and copy", "idle");
        }
      }
    });
  }

  function renderDetail(slug, catalog) {
    var entry = catalog.find(function (item) { return item.slug === slug; });
    if (!entry) {
      view.innerHTML = '<div class="notice"><h2>Entry not found</h2><p>No catalog entry has the slug ' +
        esc(slug) + '.</p><p><a href="#/">Back to the index</a></p></div>';
      dossierView.innerHTML = "";
      return;
    }

    document.title = entry.name + " - hplugins";
    var order = filtered().map(function (item) { return item.slug; });
    var position = order.indexOf(slug);
    var previous = position > 0 ? order[position - 1] : null;
    var next = position !== -1 && position < order.length - 1 ? order[position + 1] : null;

    view.innerHTML =
      '<a class="crumb" href="#/">&larr; back to index</a>' +
      '<h1 class="article-title">' + esc(entry.name) + "</h1>" +
      '<p class="dossier-meta">/' + esc(entry.type) + " - by " + esc(entry.author) + " - " + esc(entry.status) + "</p>" +
      '<p class="article-lead">' + esc(entry.summary) + "</p>" +
      '<section class="article-section"><h2 class="section-title">What it does</h2>' +
      '<p class="prose">' + esc(entry.description) + "</p></section>" +
      '<section class="article-section"><h2 class="section-title">Where to find it</h2>' +
      '<ul class="link-list">' + entry.where_to_find.map(function (link) {
        return '<li><a href="' + safeUrl(link.url) + '" rel="noopener noreferrer">' + esc(link.label) + "</a></li>";
      }).join("") + "</ul></section>" +
      '<section class="article-section"><h2 class="section-title">How to use it</h2>' +
      '<ol class="steps">' + entry.how_to_use.map(function (step) {
        return "<li>" + esc(step) + "</li>";
      }).join("") + "</ol></section>" +
      (entry.config_snippet ? '<section class="article-section"><h2 class="section-title">Config fragment</h2>' +
        '<pre class="code-block">' + esc(entry.config_snippet) + "</pre></section>" : "") +
      '<nav class="pair-nav">' +
      (previous ? '<a href="#/p/' + esc(previous) + '">Previous part</a>' : "<span></span>") +
      (next ? '<a href="#/p/' + esc(next) + '">Next part</a>' : "<span></span>") + "</nav>";

    dossierView.innerHTML = manifest(entry);
    bindCopy(entry);
  }

  function showError(message) {
    view.innerHTML = '<div class="notice"><h2>The index could not be read</h2><p>' + esc(message) +
      '</p><button class="text-button" type="button" id="retry-button">Retry</button></div>';
    dossierView.innerHTML = '<p class="dossier-kicker">Dossier</p><p class="dossier-summary">' +
      "The selected part is unavailable until the index loads.</p>";
    document.getElementById("retry-button").addEventListener("click", function () { boot(); });
  }

  function showSkeleton() {
    view.innerHTML = '<div class="skeleton-line"></div><div class="skeleton-line"></div>' +
      '<div class="skeleton-line"></div>';
    dossierView.innerHTML = '<p class="dossier-kicker">Dossier</p>';
  }

  function route() {
    var current = parseRoute();
    if (current.kind === "list") {
      renderCatalog();
      return;
    }
    showSkeleton();
    getCatalog().then(function (catalog) { renderDetail(current.slug, catalog); })
      .catch(function (error) { showError(error.message); });
  }

  function boot() {
    showSkeleton();
    fetch(INDEX_URL).then(function (response) {
      if (!response.ok) throw new Error("index request failed: " + response.status);
      return response.json();
    }).then(function (items) {
      if (!Array.isArray(items) || !items.length) throw new Error("the generated index is empty");
      state.index = items;
      renderTypeNav();
      route();
    }).catch(function (error) { showError(error.message); });
  }

  typeNav.addEventListener("click", function (event) {
    var button = event.target.closest(".type-filter");
    if (button) setFilter(button.dataset.type);
  });

  function applyThemeLabel() {
    var dark = document.documentElement.dataset.theme === "dark";
    themeLabel.textContent = dark ? "Light mode" : "Dark mode";
  }

  themeButton.addEventListener("click", function () {
    var dark = document.documentElement.dataset.theme === "dark";
    var next = dark ? "light" : "dark";
    if (next === "dark") document.documentElement.dataset.theme = "dark";
    else delete document.documentElement.dataset.theme;
    try { localStorage.setItem("hplugins-theme", next); } catch (error) {}
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", next === "dark" ? "#0D1215" : "#EDEFF1");
    applyThemeLabel();
  });

  applyThemeLabel();
  window.addEventListener("hashchange", route);
  boot();
})();
