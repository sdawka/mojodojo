// Provenance tooltip: any element with data-s (source id) and/or data-d (detail) shows its
// source file, date and status on hover, keyboard focus or tap. No network, no libraries.
(function () {
  var S = {};
  try { S = JSON.parse(document.getElementById("srcdata").textContent); } catch (e) {}
  var tip = document.getElementById("tip");
  var pinned = null;

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function target(el) {
    while (el && el !== document) {
      if (el.nodeType === 1 && (el.hasAttribute("data-s") || el.hasAttribute("data-d"))) return el;
      el = el.parentNode;
    }
    return null;
  }
  function show(el) {
    var id = el.getAttribute("data-s"), d = el.getAttribute("data-d"), src = id && S[id];
    var h = "";
    if (d) h += esc(d);
    if (src) {
      h += '<span class="tp">[' + esc(id) + "] " + esc(src.p) + "</span>" +
        (src.n ? '<span class="tp">' + esc(src.n) + "</span>" : "") +
        '<span class="tst k-' + esc(src.k) + '">' + esc(src.st) + "</span> " + esc(src.d);
    }
    if (!h) return;
    tip.innerHTML = h;
    tip.hidden = false;
    var r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
    var x = Math.min(Math.max(8, r.left + r.width / 2 - t.width / 2), window.innerWidth - t.width - 8);
    var y = r.top - t.height - 8;
    if (y < 8) y = r.bottom + 8;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }
  function hide() { tip.hidden = true; pinned = null; }

  document.addEventListener("mouseover", function (e) {
    var el = target(e.target);
    if (el) show(el); else if (!pinned) tip.hidden = true;
  });
  document.addEventListener("focusin", function (e) { var el = target(e.target); if (el) show(el); });
  document.addEventListener("focusout", function () { if (!pinned) tip.hidden = true; });
  document.addEventListener("click", function (e) {
    var el = target(e.target);
    if (el && el.tagName !== "A") { pinned = el; show(el); } else hide();
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") hide(); });
  window.addEventListener("scroll", function () { if (!tip.hidden) hide(); }, { passive: true });
})();
