// Display-only: no uploads, resampling, generation settings or backend events.
(() => {
  const constrain = (pose, view, image) => ({
    ...pose,
    x:
      image.w * pose.scale <= view.w
        ? (view.w - image.w * pose.scale) / 2
        : Math.max(view.w - image.w * pose.scale, Math.min(0, pose.x)),
    y:
      image.h * pose.scale <= view.h
        ? (view.h - image.h * pose.scale) / 2
        : Math.max(view.h - image.h * pose.scale, Math.min(0, pose.y)),
  });
  const preset = (mode, view, image) => {
    const ratios = [view.w / image.w, view.h / image.h];
    const scale = mode === "actual" ? 1 : (mode === "fill" ? Math.max : Math.min)(...ratios);
    return { scale, x: (view.w - image.w * scale) / 2, y: (view.h - image.h * scale) / 2 };
  };
  const zoomAt = (pose, factor, point, view, image) => {
    const fit = preset("fit", view, image).scale;
    const scale = Math.max(Math.min(fit, 0.05), Math.min(Math.max(16, fit), pose.scale * factor));
    return constrain(
      {
        scale,
        x: point.x - ((point.x - pose.x) * scale) / pose.scale,
        y: point.y - ((point.y - pose.y) * scale) / pose.scale,
      },
      view,
      image,
    );
  };
  const reframed = (pose, previous, view, image, mode) => {
    if (mode === "fit") return preset(mode, view, image);
    const scale = mode === "fill" ? preset(mode, view, image).scale : pose.scale;
    return constrain(
      {
        scale,
        x: view.w / 2 - ((previous.w / 2 - pose.x) * scale) / pose.scale,
        y: view.h / 2 - ((previous.h / 2 - pose.y) * scale) / pose.scale,
      },
      view,
      image,
    );
  };
  // Double-click / double-tap, as in Photos: the whole image zooms to actual
  // pixels at the pointer (or 2x when the whole image already is 1:1); any
  // other zoom returns to the whole image.
  const toggled = (pose, point, view, image) => {
    const fit = preset("fit", view, image);
    if (Math.abs(pose.scale - fit.scale) > 1e-6) return { pose: fit, mode: "fit" };
    const factor = Math.abs(fit.scale - 1) < 1e-6 ? 2 : 1 / pose.scale;
    return {
      pose: zoomAt(pose, factor, point, view, image),
      mode: factor === 2 ? "custom" : "actual",
    };
  };
  // Two-finger pinch: scale by the change in finger distance around the
  // midpoint, and follow the midpoint as it moves.
  const pinched = (start, from, to, view, image) => {
    const factor = to.distance / Math.max(1, from.distance);
    const zoomed = zoomAt(start, factor, from.mid, view, image);
    return constrain(
      { ...zoomed, x: zoomed.x + to.mid.x - from.mid.x, y: zoomed.y + to.mid.y - from.mid.y },
      view,
      image,
    );
  };
  // Arrow keys move a tenth of the view, like ScrollView's line steps.
  const nudged = (pose, dx, dy, view, image) =>
    constrain(
      { ...pose, x: pose.x - dx * view.w * 0.1, y: pose.y - dy * view.h * 0.1 },
      view,
      image,
    );
  const resized = (width, height, available, screenHeight) => ({
    width: Math.min(available, Math.max(Math.min(220, available), width)),
    height: Math.min(Math.max(400, screenHeight * 2), Math.max(200, height)),
  });
  // The tests exercise the same geometry used below, without a DOM substitute.
  if (typeof document === "undefined" && typeof module !== "undefined" && module.exports) {
    module.exports = { constrain, preset, zoomAt, resized, reframed, toggled, pinched, nudged };
    return;
  }
  if (window.rfImageViewer || !document.getElementById("generation-workspace")) return;
  let workspace,
    root,
    viewport,
    image,
    listeners,
    source = "",
    mode = "fit";
  let pose = { scale: 1, x: 0, y: 0 },
    view = { w: 0, h: 0 },
    dimensions;
  let pan,
    pinch,
    resize,
    animation,
    pending = false;
  const pointers = new Map();
  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)");
  // Buttons and keys glide to the new view; wheel, drag and pinch follow the hand.
  function animate() {
    if (!viewport || reducedMotion?.matches) return;
    viewport.classList.add("rf-animate");
    clearTimeout(animation);
    animation = setTimeout(() => viewport?.classList.remove("rf-animate"), 260);
  }
  const workspaceObserver = new MutationObserver(() => schedule(bind));
  const imageObserver = new MutationObserver(() => schedule(bind));
  const sizeObserver = new ResizeObserver(() => schedule(reflow));
  let nextUpdate = null;
  function schedule(update) {
    // Structural changes take precedence over a resize in the same frame.
    nextUpdate = nextUpdate === bind || update === bind ? bind : update;
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const run = nextUpdate;
      nextUpdate = null;
      run();
    });
  }
  function paint() {
    if (!image?.naturalWidth || !dimensions || !view.w || !view.h) return;
    pose = constrain(pose, view, dimensions);
    image.style.width = `${dimensions.w}px`;
    image.style.height = `${dimensions.h}px`;
    image.style.transform = `translate3d(${pose.x}px, ${pose.y}px, 0) scale(${pose.scale})`;
    viewport.classList.toggle(
      "rf-can-pan",
      dimensions.w * pose.scale > view.w + 1 || dimensions.h * pose.scale > view.h + 1,
    );
    for (const key of ["fit", "fill", "actual"]) {
      const button = document.getElementById(`viewer-${key}`);
      const pressed = String(mode === key);
      if (button && button.getAttribute("aria-pressed") !== pressed)
        button.setAttribute("aria-pressed", pressed);
    }
    const output = document.querySelector("#viewer-zoom-value output");
    const percent = pose.scale * 100;
    const text = `${percent < 10 ? Math.round(percent * 10) / 10 : Math.round(percent)}%`;
    if (output && output.textContent !== text) output.textContent = text;
  }
  function reflow() {
    placeHandle();
    if (!viewport || !image?.naturalWidth || !viewport.clientWidth || !viewport.clientHeight)
      return;
    const previous = view;
    view = { w: viewport.clientWidth, h: viewport.clientHeight };
    const key = image.currentSrc || image.src;
    const changed = key !== source;
    dimensions = { w: image.naturalWidth, h: image.naturalHeight };
    if (changed) {
      source = key;
      if (mode === "custom") mode = "fit";
    }
    if (changed || !previous.w) {
      pose = preset(mode, view, dimensions);
    } else {
      pose = reframed(pose, previous, view, dimensions, mode);
    }
    paint();
  }
  function placeHandle() {
    const controls = document.getElementById("image-view-controls");
    if (!handle || !root || !controls) return;
    const rect = root.getBoundingClientRect(),
      row = controls.getBoundingClientRect();
    handle.hidden = root.classList.contains("fullscreen") || !rect.width;
    if (handle.hidden) return;
    handle.style.setProperty(
      "--rf-handle-x",
      `${rect.right - row.left - handle.offsetWidth - 8}px`,
    );
    handle.style.setProperty(
      "--rf-handle-y",
      `${rect.bottom - row.top - handle.offsetHeight - 8}px`,
    );
  }
  function setMode(value, smooth = false) {
    bind();
    mode = ["fit", "fill", "actual"].includes(value) ? value : "fit";
    if (dimensions && view.w) {
      if (smooth) animate();
      pose = preset(mode, view, dimensions);
      paint();
    }
  }
  function toggle(point) {
    bind();
    if (!dimensions || !view.w) return;
    const next = toggled(pose, point || { x: view.w / 2, y: view.h / 2 }, view, dimensions);
    animate();
    pose = next.pose;
    mode = next.mode;
    paint();
  }
  function zoom(factor, point, smooth = false) {
    if (!viewport?.isConnected || !image?.isConnected) bind();
    if (!image?.naturalWidth || !dimensions || !view.w || !Number.isFinite(factor) || factor <= 0)
      return;
    if (smooth) animate();
    pose = zoomAt(pose, factor, point || { x: view.w / 2, y: view.h / 2 }, view, dimensions);
    mode = "custom";
    schedule(paint);
  }
  function setSize(width, height) {
    if (!workspace) return;
    const size = resized(width, height, workspace.clientWidth, window.innerHeight);
    workspace.style.setProperty("--rf-viewer-width", `${size.width}px`);
    workspace.style.setProperty("--rf-viewer-height", `${size.height}px`);
  }
  function bind() {
    const currentWorkspace = document.getElementById("generation-workspace");
    if (currentWorkspace !== workspace) {
      workspaceObserver.disconnect();
      workspace = currentWorkspace;
      if (workspace) workspaceObserver.observe(workspace, { childList: true });
    }
    const currentRoot = document.getElementById("main_view");
    const currentViewport = currentRoot?.querySelector(".image-container > .upload-container");
    const currentImage = currentViewport?.querySelector(".image-frame img");
    if (currentRoot !== root) {
      imageObserver.disconnect();
      root = currentRoot;
      if (root)
        imageObserver.observe(root, {
          childList: true,
          subtree: true,
          attributes: true,
          attributeFilter: ["src"],
        });
    }
    if (currentViewport === viewport && currentImage === image) {
      reflow();
      return;
    }
    listeners?.abort();
    sizeObserver.disconnect();
    if (root) sizeObserver.observe(root);
    // The resize grip is placed from the floating command bar's box.
    const bar = document.getElementById("image-view-controls");
    if (bar) sizeObserver.observe(bar);
    viewport = currentViewport;
    image = currentImage;
    pan = null;
    viewport?.classList.remove("rf-can-pan", "rf-panning");
    if (!viewport || !image) {
      reflow();
      return;
    }
    root.dataset.rfViewer = "true";
    listeners = new AbortController();
    const signal = listeners.signal;
    image.draggable = false;
    image.addEventListener("load", reflow, { signal });
    sizeObserver.observe(viewport);
    viewport.addEventListener(
      "wheel",
      (event) => {
        if (event.ctrlKey || !image.naturalWidth || event.deltaY === 0) return;
        event.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const pixels =
          event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? view.h : 1);
        zoom(Math.exp(-Math.max(-500, Math.min(500, pixels)) * 0.002), {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top,
        });
      },
      { passive: false, signal },
    );
    const local = (event) => {
      const rect = viewport.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };
    const fingers = () => {
      const [a, b] = [...pointers.values()];
      return {
        mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
        distance: Math.hypot(a.x - b.x, a.y - b.y),
      };
    };
    viewport.addEventListener(
      "pointerdown",
      (event) => {
        if (!image.naturalWidth || (event.pointerType === "mouse" && event.button !== 0)) return;
        pointers.set(event.pointerId, local(event));
        // Panning cancels the default focus change; the keyboard still belongs here.
        if (document.activeElement !== viewport) viewport.focus({ preventScroll: true });
        if (pointers.size === 2) {
          pan = null;
          pinch = { from: fingers(), pose: { ...pose } };
        } else if (pointers.size === 1 && viewport.classList.contains("rf-can-pan")) {
          pan = { id: event.pointerId, x: event.clientX, y: event.clientY, pose: { ...pose } };
        } else {
          return;
        }
        event.preventDefault();
        viewport.setPointerCapture(event.pointerId);
        viewport.classList.add("rf-panning");
      },
      { signal },
    );
    viewport.addEventListener(
      "pointermove",
      (event) => {
        if (pointers.has(event.pointerId)) pointers.set(event.pointerId, local(event));
        if (pinch && pointers.size >= 2) {
          pose = pinched(pinch.pose, pinch.from, fingers(), view, dimensions);
          mode = "custom";
          schedule(paint);
          return;
        }
        if (!pan || pan.id !== event.pointerId) return;
        pose = {
          ...pan.pose,
          x: pan.pose.x + event.clientX - pan.x,
          y: pan.pose.y + event.clientY - pan.y,
        };
        schedule(paint);
      },
      { signal },
    );
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
      viewport.addEventListener(
        name,
        (event) => {
          pointers.delete(event.pointerId);
          if (pointers.size < 2) pinch = null;
          if (pointers.size === 0) {
            pan = null;
            viewport.classList.remove("rf-panning");
          }
        },
        { signal },
      );
    }
    viewport.addEventListener(
      "dblclick",
      (event) => {
        if (!image.naturalWidth) return;
        event.preventDefault();
        toggle(local(event));
      },
      { signal },
    );
    // Keyboard, as WinUI's ScrollView: the focused image zooms with +/- (also
    // with Ctrl), returns to the whole image with 0 or Home, shows actual pixels
    // with 1 and moves with the arrow keys.
    viewport.tabIndex = 0;
    viewport.addEventListener(
      "keydown",
      (event) => {
        if (event.altKey || event.metaKey || !image.naturalWidth) return;
        const arrows = {
          ArrowLeft: [-1, 0],
          ArrowRight: [1, 0],
          ArrowUp: [0, -1],
          ArrowDown: [0, 1],
        };
        if (event.key === "+" || event.key === "=") zoom(1.25, null, true);
        else if (event.key === "-" || event.key === "_") zoom(0.8, null, true);
        else if (event.key === "0" || event.key === "Home") setMode("fit", true);
        else if (event.key === "1") setMode("actual", true);
        else if (arrows[event.key] && viewport.classList.contains("rf-can-pan")) {
          const [dx, dy] = arrows[event.key];
          animate();
          pose = nudged(pose, dx, dy, view, dimensions);
          mode = "custom";
          schedule(paint);
        } else return;
        event.preventDefault();
      },
      { signal },
    );
    reflow();
  }
  const handle = document.getElementById("viewer-resize");
  if (handle) {
    handle.addEventListener("pointerenter", () => {
      handle.title = handle.textContent.trim();
    });
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || !root) return;
      event.preventDefault();
      const rect = root.getBoundingClientRect();
      resize = {
        id: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        width: rect.width,
        height: rect.height,
      };
      handle.setPointerCapture(event.pointerId);
    });
    handle.addEventListener("pointermove", (event) => {
      if (resize?.id === event.pointerId)
        setSize(resize.width + event.clientX - resize.x, resize.height + event.clientY - resize.y);
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
      handle.addEventListener(name, () => {
        resize = null;
      });
    handle.addEventListener("dblclick", () => {
      workspace?.style.removeProperty("--rf-viewer-width");
      workspace?.style.removeProperty("--rf-viewer-height");
    });
    handle.addEventListener("keydown", (event) => {
      const delta = {
        ArrowLeft: [-20, 0],
        ArrowRight: [20, 0],
        ArrowUp: [0, -20],
        ArrowDown: [0, 20],
      }[event.key];
      if (!delta || !root) return;
      event.preventDefault();
      const rect = root.getBoundingClientRect();
      setSize(rect.width + delta[0], rect.height + delta[1]);
    });
  }
  // Tooltips name the keyboard shortcut; icon buttons keep their text as the
  // accessible name.
  const shortcuts = {
    "viewer-zoom-out": "-",
    "viewer-zoom-in": "+",
    "viewer-fit": "0",
    "viewer-fill": "",
    "viewer-actual": "1",
  };
  for (const [id, key] of Object.entries(shortcuts)) {
    const button = document.getElementById(id);
    button?.addEventListener("pointerenter", () => {
      const label = button.textContent.trim();
      button.setAttribute("aria-label", label);
      button.title = key ? `${label} (${key})` : label;
    });
  }
  const readout = document.querySelector("#viewer-zoom-value output");
  if (readout) {
    // The zoom readout is also a button: whole image <-> actual pixels.
    readout.setAttribute("role", "button");
    readout.tabIndex = 0;
    readout.addEventListener("click", () => toggle());
    readout.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggle();
    });
  }
  window.rfImageViewer = {
    mode: (value) => setMode(value, true),
    zoom: (factor, point) => zoom(factor, point, true),
  };
  bind();
})();
