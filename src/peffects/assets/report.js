(function () {
  var tip = document.getElementById("tip");

  function place(e) {
    var r = tip.getBoundingClientRect();
    var x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  }

  function fill(lines) {
    tip.replaceChildren();
    lines.forEach(function (text, i) {
      var row = document.createElement("div");
      row.textContent = text;
      if (i === 0) row.className = "head";
      tip.appendChild(row);
    });
    tip.hidden = false;
  }

  function mark(svg, x, y) {
    var line = svg.querySelector(".crosshair");
    if (line && x != null) {
      line.setAttribute("x1", x);
      line.setAttribute("x2", x);
      line.setAttribute("visibility", "visible");
    }
    var dot = svg.querySelector(".focus");
    if (dot && y != null) {
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", y);
      dot.setAttribute("visibility", "visible");
    }
  }

  function clear(svg) {
    tip.hidden = true;
    if (svg) svg.querySelectorAll(".crosshair,.focus").forEach(function (n) {
      n.setAttribute("visibility", "hidden");
    });
  }

  document.querySelectorAll("[data-tip]").forEach(function (el) {
    el.addEventListener("pointerenter", function (e) {
      fill(JSON.parse(el.getAttribute("data-tip")));
      place(e);
      if (el.dataset.x) mark(el.ownerSVGElement, el.dataset.x, el.dataset.y);
    });
    el.addEventListener("pointermove", place);
    el.addEventListener("pointerleave", function () { clear(el.ownerSVGElement); });
  });

  // Points are [x, y, ...tooltip lines], nearest by x.
  document.querySelectorAll("[data-points]").forEach(function (el) {
    var svg = el.ownerSVGElement, points = JSON.parse(el.getAttribute("data-points"));
    el.addEventListener("pointermove", function (e) {
      var pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      var x = pt.matrixTransform(svg.getScreenCTM().inverse()).x;
      var lo = 0, hi = points.length - 1;
      while (hi - lo > 1) {
        var mid = (lo + hi) >> 1;
        if (points[mid][0] < x) lo = mid; else hi = mid;
      }
      var p = Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? points[lo] : points[hi];
      fill(p.slice(2));
      place(e);
      mark(svg, p[0], p[1]);
    });
    el.addEventListener("pointerleave", function () { clear(svg); });
  });

  var root = document.documentElement, button = document.getElementById("theme");
  function isDark() {
    var set = root.getAttribute("data-theme");
    return set ? set === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function label() { button.textContent = isDark() ? "Light" : "Dark"; }
  try {
    var saved = localStorage.getItem("theme");
    if (saved) root.setAttribute("data-theme", saved);
  } catch (e) {}
  label();
  button.addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    label();
  });
})();
