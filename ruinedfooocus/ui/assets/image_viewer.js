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
  const resized = (width, height, available, screenHeight) => ({
    width: Math.min(available, Math.max(Math.min(220, available), width)),
    height: Math.min(Math.max(400, screenHeight * 2), Math.max(200, height)),
  });
  // The tests exercise the same geometry used below, without a DOM substitute.
  if (typeof document === "undefined" && typeof module !== "undefined" && module.exports) {
    module.exports = { constrain, preset, zoomAt, resized, reframed };
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
    resize,
    pending = false;
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
  function setMode(value) {
    bind();
    mode = ["fit", "fill", "actual"].includes(value) ? value : "fit";
    if (dimensions && view.w) {
      pose = preset(mode, view, dimensions);
      paint();
    }
  }
  function zoom(factor, point) {
    if (!viewport?.isConnected || !image?.isConnected) bind();
    if (!image?.naturalWidth || !dimensions || !view.w || !Number.isFinite(factor) || factor <= 0)
      return;
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
    viewport.addEventListener(
      "pointerdown",
      (event) => {
        if (event.button !== 0 || !image.naturalWidth || !viewport.classList.contains("rf-can-pan"))
          return;
        event.preventDefault();
        pan = { id: event.pointerId, x: event.clientX, y: event.clientY, pose: { ...pose } };
        viewport.setPointerCapture(event.pointerId);
        viewport.classList.add("rf-panning");
      },
      { signal },
    );
    viewport.addEventListener(
      "pointermove",
      (event) => {
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
        () => {
          pan = null;
          viewport.classList.remove("rf-panning");
        },
        { signal },
      );
    }
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
  for (const id of ["viewer-zoom-out", "viewer-zoom-in"]) {
    const button = document.getElementById(id);
    button?.addEventListener("pointerenter", () => {
      button.title = button.textContent.trim();
    });
  }
  window.rfImageViewer = { mode: setMode, zoom };
  bind();
})();
