(value) => {
  const scale = Math.max(75, Math.min(200, Number(value) || 100));
  document.documentElement.style.setProperty("--rf-user-text-scale", `${scale}%`);
  localStorage.setItem("ruinedfooocus-text-scale", String(scale));
};
