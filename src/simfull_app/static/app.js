/**
 * simfull — Full simulation software front-end.
 *
 * 1. 2D Interactive floorplan canvas editor.
 * 2. 3D spatial building projection colored by simulation node temperatures.
 * 3. 3D STL mesh file import.
 * 4. Multi-zone thermal network simulation with dynamic SVG charts.
 * 5. 2D finite-difference thermal bridge contour visualizer (EN ISO 10211).
 * 6. 1D transient multi-layer wall conduction solver.
 */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

class SimfullApp {
  constructor() {
    this.geometry = null;
    this.simulationResult = null;
    this.activeVertex = null;
    this.dragOffset = { x: 0, y: 0 };
    this.scale = 40; // pixels per meter on 2D canvas
    this.origin = { x: 60, y: 350 };
  }

  async start() {
    this.bindTabs();
    this.bindCanvasEvents();
    this.bindControls();
    await this.loadDefaultGeometry();
    await this.runSimulation();
    await this.runThermalBridge();
    await this.runWall1D();
  }

  // ------------------------------------------------------------------------------------------------- tabs
  bindTabs() {
    $$(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".tab-btn").forEach((b) => b.classList.remove("active"));
        $$(".tab-panel").forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        const panelId = `panel-${btn.dataset.tab}`;
        const panel = $(`#${panelId}`);
        if (panel) panel.classList.add("active");
        if (btn.dataset.tab === "geometry") {
          this.render2DCanvas();
          this.render3DSpatial();
        }
      });
    });
  }

  toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.hidden = false;
    setTimeout(() => { t.hidden = true; }, 3000);
  }

  // ------------------------------------------------------------------------------------------------- geometry & 2D canvas
  async loadDefaultGeometry() {
    try {
      const res = await fetch("/api/geometry/default");
      this.geometry = await res.json();
      this.updateGeometryStats();
      this.render2DCanvas();
      this.render3DSpatial();
    } catch (e) {
      this.toast(`Failed to load default geometry: ${e.message}`);
    }
  }

  updateGeometryStats() {
    if (!this.geometry) return;
    const nZones = this.geometry.zones.length;
    const nWalls = this.geometry.walls.length;
    const area = this.geometry.total_floor_area;
    $("#geometry-stats").textContent = `${nZones} Zones · ${nWalls} Walls · ${area} m² Floor Area`;
  }

  bindCanvasEvents() {
    const canvas = $("#floorplan-canvas");
    if (!canvas) return;

    canvas.addEventListener("mousedown", (e) => {
      const pos = this.getCanvasPos(e, canvas);
      const mPos = this.toMeters(pos);
      // Check if clicking near an existing vertex (threshold ~0.4m)
      for (const z of this.geometry.zones) {
        for (let i = 0; i < z.vertices.length; i++) {
          const v = z.vertices[i];
          const dist = Math.hypot(v.x - mPos.x, v.y - mPos.y);
          if (dist < 0.4) {
            this.activeVertex = { zone: z, index: i };
            this.dragOffset = { x: v.x - mPos.x, y: v.y - mPos.y };
            return;
          }
        }
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.activeVertex) return;
      const pos = this.getCanvasPos(e, canvas);
      const mPos = this.toMeters(pos);
      // Snap to 0.5m grid
      const snappedX = Math.round((mPos.x + this.dragOffset.x) * 2) / 2;
      const snappedY = Math.round((mPos.y + this.dragOffset.y) * 2) / 2;
      const v = this.activeVertex.zone.vertices[this.activeVertex.index];
      v.x = Math.max(0, snappedX);
      v.y = Math.max(0, snappedY);

      // Synchronize wall endpoints
      this.syncWallsWithVertices();
      this.render2DCanvas();
      this.render3DSpatial();
    });

    window.addEventListener("mouseup", () => {
      if (this.activeVertex) {
        this.activeVertex = null;
        this.updateGeometryStats();
      }
    });
  }

  getCanvasPos(e, canvas) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  }

  toMeters(pos) {
    return {
      x: (pos.x - this.origin.x) / this.scale,
      y: (this.origin.y - pos.y) / this.scale,
    };
  }

  toPixels(m) {
    return {
      x: this.origin.x + m.x * this.scale,
      y: this.origin.y - m.y * this.scale,
    };
  }

  syncWallsWithVertices() {
    if (!this.geometry || !this.geometry.zones.length) return;
    let wallIdx = 0;
    this.geometry.zones.forEach((z) => {
      const pts = z.vertices;
      for (let i = 0; i < pts.length; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        if (wallIdx < this.geometry.walls.length) {
          const w = this.geometry.walls[wallIdx];
          w.p1 = p1;
          w.p2 = p2;
          wallIdx++;
        }
      }
    });
  }

  render2DCanvas() {
    const canvas = $("#floorplan-canvas");
    if (!canvas || !this.geometry) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Draw background grid (1m grid lines)
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    for (let x = this.origin.x % this.scale; x < w; x += this.scale) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = this.origin.y % this.scale; y < h; y += this.scale) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Draw coordinate axes
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(this.origin.x, 0); ctx.lineTo(this.origin.x, h);
    ctx.moveTo(0, this.origin.y); ctx.lineTo(w, this.origin.y);
    ctx.stroke();

    // Draw zones
    const zoneColors = ["rgba(37, 99, 235, 0.12)", "rgba(16, 185, 129, 0.12)", "rgba(245, 158, 11, 0.12)"];
    this.geometry.zones.forEach((z, idx) => {
      if (z.vertices.length < 3) return;
      ctx.beginPath();
      const p0 = this.toPixels(z.vertices[0]);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < z.vertices.length; i++) {
        const p = this.toPixels(z.vertices[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.fillStyle = zoneColors[idx % zoneColors.length];
      ctx.fill();

      // Label zone center
      let cx = 0, cy = 0;
      z.vertices.forEach((v) => { cx += v.x; cy += v.y; });
      cx /= z.vertices.length;
      cy /= z.vertices.length;
      const pc = this.toPixels({ x: cx, y: cy });
      ctx.fillStyle = "#1e293b";
      ctx.font = "600 12px 'Space Grotesk', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(z.name || z.id, pc.x, pc.y - 6);
      ctx.font = "10px 'Fira Mono', monospace";
      ctx.fillStyle = "#64748b";
      const zArea = z.floor_area || 0;
      ctx.fillText(`${zArea.toFixed(1)} m²`, pc.x, pc.y + 10);
    });

    // Draw walls
    this.geometry.walls.forEach((wall) => {
      const p1 = this.toPixels(wall.p1);
      const p2 = this.toPixels(wall.p2);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      if (wall.zone_b) {
        // Interior partition
        ctx.strokeStyle = "#94a3b8";
        ctx.lineWidth = 3;
        ctx.setLineDash([4, 4]);
      } else {
        // Exterior wall
        ctx.strokeStyle = "#0f172a";
        ctx.lineWidth = 5;
        ctx.setLineDash([]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw window indication
      if (wall.window_area > 0) {
        const mx = (p1.x + p2.x) * 0.5;
        const my = (p1.y + p2.y) * 0.5;
        ctx.fillStyle = "#38bdf8";
        ctx.beginPath();
        ctx.arc(mx, my, 4, 0, 2 * Math.PI);
        ctx.fill();
      }
    });

    // Draw vertex handles
    this.geometry.zones.forEach((z) => {
      z.vertices.forEach((v) => {
        const p = this.toPixels(v);
        ctx.fillStyle = "#2563eb";
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      });
    });
  }

  // ------------------------------------------------------------------------------------------------- 3D spatial projection & coloring
  render3DSpatial() {
    const canvas = $("#spatial-3d-canvas");
    if (!canvas || !this.geometry) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Isometric projection angles
    const angleX = Math.PI / 6;  // 30 degrees
    const angleY = Math.PI / 6;
    const cx = w * 0.5;
    const cy = h * 0.65;
    const scale3d = 28;

    const project = (x, y, z) => {
      const px = cx + (x - y) * Math.cos(angleX) * scale3d;
      const py = cy - z * scale3d + (x + y) * Math.sin(angleY) * scale3d * 0.5;
      return { x: px, y: py };
    };

    // Current temperature lookup based on time slider
    const currentHour = parseFloat($("#time-slider")?.value || "12");
    const getZoneColor = (zoneId) => {
      if (!this.simulationResult) return "#e2e8f0";
      const net = this.simulationResult.network;
      const nid = `${zoneId}.air`;
      const temps = net.temperatures[nid];
      if (!temps || !temps.length) return "#e2e8f0";
      const hours = net.hours;
      // Interpolate temperature
      let temp = temps[0];
      for (let i = 0; i < hours.length; i++) {
        if (hours[i] >= currentHour) { temp = temps[i]; break; }
      }
      // Temperature colormap from 15°C (blue) to 25°C (red)
      const tNorm = Math.min(1.0, Math.max(0.0, (temp - 15.0) / 10.0));
      return this.colormap(tNorm);
    };

    // Draw ground grid plane
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    for (let gx = -2; gx <= 12; gx += 2) {
      const pA = project(gx, -2, 0);
      const pB = project(gx, 12, 0);
      ctx.beginPath(); ctx.moveTo(pA.x, pA.y); ctx.lineTo(pB.x, pB.y); ctx.stroke();
    }

    // Draw 3D zones (extruded prism walls and roof)
    this.geometry.zones.forEach((z) => {
      const zColor = getZoneColor(z.id);
      const pts = z.vertices;
      const height = z.height || 2.6;

      // Floor slab
      ctx.beginPath();
      const p0 = project(pts[0].x, pts[0].y, 0);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < pts.length; i++) {
        const pt = project(pts[i].x, pts[i].y, 0);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(148, 163, 184, 0.4)";
      ctx.fill();
      ctx.strokeStyle = "#64748b";
      ctx.stroke();

      // Vertical walls
      for (let i = 0; i < pts.length; i++) {
        const pA = pts[i];
        const pB = pts[(i + 1) % pts.length];
        const vA_bot = project(pA.x, pA.y, 0);
        const vB_bot = project(pB.x, pB.y, 0);
        const vB_top = project(pB.x, pB.y, height);
        const vA_top = project(pA.x, pA.y, height);

        ctx.beginPath();
        ctx.moveTo(vA_bot.x, vA_bot.y);
        ctx.lineTo(vB_bot.x, vB_bot.y);
        ctx.lineTo(vB_top.x, vB_top.y);
        ctx.lineTo(vA_top.x, vA_top.y);
        ctx.closePath();
        ctx.fillStyle = zColor;
        ctx.fill();
        ctx.strokeStyle = "#334155";
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

      // Roof slab
      ctx.beginPath();
      const r0 = project(pts[0].x, pts[0].y, height);
      ctx.moveTo(r0.x, r0.y);
      for (let i = 1; i < pts.length; i++) {
        const pt = project(pts[i].x, pts[i].y, height);
        ctx.lineTo(pt.x, pt.y);
      }
      ctx.closePath();
      ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
      ctx.fill();
      ctx.strokeStyle = "#1e293b";
      ctx.stroke();
    });
  }

  colormap(t) {
    // 0.0 (blue #3b82f6) -> 0.5 (yellow #eab308) -> 1.0 (red #ef4444)
    if (t < 0.5) {
      const f = t * 2.0;
      const r = Math.round(59 + (234 - 59) * f);
      const g = Math.round(130 + (179 - 130) * f);
      const b = Math.round(246 + (8 - 246) * f);
      return `rgba(${r}, ${g}, ${b}, 0.75)`;
    }
    const f = (t - 0.5) * 2.0;
    const r = Math.round(234 + (239 - 234) * f);
    const g = Math.round(179 + (68 - 179) * f);
    const b = Math.round(8 + (68 - 8) * f);
    return `rgba(${r}, ${g}, ${b}, 0.75)`;
  }

  // ------------------------------------------------------------------------------------------------- controls & STL import
  bindControls() {
    $("#btn-run-sim")?.addEventListener("click", () => this.runSimulation());

    $("#time-slider")?.addEventListener("input", (e) => {
      const hr = parseFloat(e.target.value);
      const hh = Math.floor(hr);
      const mm = (hr % 1) * 60 === 30 ? "30" : "00";
      $("#time-slider-val").textContent = `${String(hh).padStart(2, "0")}:${mm} (Hour ${hr.toFixed(1)})`;
      this.render3DSpatial();
    });

    $("#btn-preset-2room")?.addEventListener("click", () => this.loadDefaultGeometry());

    $("#btn-preset-lshape")?.addEventListener("click", () => {
      this.geometry = this.createLShapeGeometry();
      this.updateGeometryStats();
      this.render2DCanvas();
      this.render3DSpatial();
    });

    $("#btn-clear-canvas")?.addEventListener("click", () => {
      this.geometry = { zones: [], walls: [], roof_u_value: 0.15, floor_u_value: 0.18, total_floor_area: 0 };
      this.updateGeometryStats();
      this.render2DCanvas();
      this.render3DSpatial();
    });

    // 3D STL file upload
    $("#stl-file-input")?.addEventListener("change", (e) => this.handleSTLUpload(e.target.files && e.target.files[0]));

    // 2D Thermal bridge button
    $("#btn-solve-tb")?.addEventListener("click", () => this.runThermalBridge());

    // 1D Wall conduction button
    $("#btn-solve-wall1d")?.addEventListener("click", () => this.runWall1D());
  }

  createLShapeGeometry() {
    const p1 = { x: 0, y: 0 }, p2 = { x: 8, y: 0 }, p3 = { x: 8, y: 4 }, p4 = { x: 4, y: 4 }, p5 = { x: 4, y: 8 }, p6 = { x: 0, y: 8 };
    const pMid = { x: 4, y: 0 };
    return {
      zones: [
        { id: "main_hall", name: "Main Hall", vertices: [p1, pMid, p4, p6], height: 2.8, floor_area: 32.0, setpoint_temp_c: 21.0, internal_gain_w: 150, ventilation_ach: 0.5 },
        { id: "east_wing", name: "East Wing", vertices: [pMid, p2, p3, p4], height: 2.8, floor_area: 16.0, setpoint_temp_c: 20.0, internal_gain_w: 100, ventilation_ach: 0.5 },
        { id: "north_wing", name: "North Wing", vertices: [p6, p4, p5], height: 2.8, floor_area: 16.0, setpoint_temp_c: 19.0, internal_gain_w: 80, ventilation_ach: 0.5 },
      ],
      walls: [
        { id: "w1", p1, p2: pMid, zone_a: "main_hall", window_area: 2.5, u_value: 0.22, height: 2.8 },
        { id: "w2", p1: pMid, p2, zone_a: "east_wing", window_area: 2.0, u_value: 0.22, height: 2.8 },
        { id: "w3", p1: p2, p2: p3, zone_a: "east_wing", window_area: 1.5, u_value: 0.22, height: 2.8 },
        { id: "w4", p1: p3, p2: p4, zone_a: "east_wing", window_area: 1.0, u_value: 0.22, height: 2.8 },
        { id: "w5", p1: p4, p2: p5, zone_a: "north_wing", window_area: 1.2, u_value: 0.22, height: 2.8 },
        { id: "w6", p1: p5, p2: p6, zone_a: "north_wing", window_area: 1.2, u_value: 0.22, height: 2.8 },
        { id: "w7", p1: p6, p2: p1, zone_a: "main_hall", window_area: 2.0, u_value: 0.22, height: 2.8 },
      ],
      roof_u_value: 0.15,
      floor_u_value: 0.18,
      total_floor_area: 64.0,
    };
  }

  async handleSTLUpload(file) {
    if (!file) return;
    this.toast(`Processing 3D STL mesh: ${file.name}...`);
    const reader = new FileReader();
    reader.onload = async (e) => {
      const buffer = e.target.result;
      const bytes = new Uint8Array(buffer);
      // Check if ASCII or binary
      let isBinary = true;
      let text = "";
      try {
        text = new TextDecoder("utf-8").decode(bytes.slice(0, 100));
        if (text.startsWith("solid") && !bytes.slice(0, 80).some((b) => b === 0)) {
          isBinary = false;
        }
      } catch { isBinary = true; }

      let reqBody;
      if (isBinary) {
        // Base64 encode binary STL
        let binaryStr = "";
        for (let i = 0; i < bytes.length; i++) binaryStr += String.fromCharCode(bytes[i]);
        const b64 = btoa(binaryStr);
        reqBody = { content: b64, is_base64: true, zone_name: file.name.replace(".stl", "") };
      } else {
        const fullText = new TextDecoder("utf-8").decode(bytes);
        reqBody = { content: fullText, is_base64: false, zone_name: file.name.replace(".stl", "") };
      }

      try {
        const res = await fetch("/api/geometry/from-stl", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(reqBody),
        });
        if (!res.ok) throw new Error(await res.text());
        this.geometry = await res.json();
        this.updateGeometryStats();
        this.render2DCanvas();
        this.render3DSpatial();
        this.toast(`STL mesh parsed: ${this.geometry.metadata?.facets_count || "3D"} facets converted to thermal zones!`);
      } catch (err) {
        this.toast(`STL parsing error: ${err.message}`);
      }
    };
    reader.readAsArrayBuffer(file);
  }

  // ------------------------------------------------------------------------------------------------- simulation execution
  async runSimulation() {
    this.toast("Running full simulation network...");
    const req = {
      geometry: this.geometry,
      weather_preset: $("#sim-weather-select")?.value || "winter_design_day",
      construction: $("#sim-construction-select")?.value || "brick_cavity",
      hp_kw: parseFloat($("#sim-hp-kw")?.value || "8"),
      pv_m2: parseFloat($("#sim-pv-m2")?.value || "20"),
      duration_days: 1.0,
      dt_seconds: 300.0,
      run_detailed_fields: true,
    };

    try {
      const res = await fetch("/api/simulate/network", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
      if (!res.ok) throw new Error(await res.text());
      this.simulationResult = await res.json();
      this.updateMetrics(this.simulationResult.metrics);
      this.renderCharts(this.simulationResult.network);
      this.render3DSpatial();
      this.toast("Simulation completed: Energy balance verified!");
    } catch (e) {
      this.toast(`Simulation failed: ${e.message}`);
    }
  }

  updateMetrics(m) {
    if (!m) return;
    $("#m-heat-demand").textContent = `${m.space_heating_demand_kwh} kWh`;
    const fArea = m.total_floor_area_m2 || 50.0;
    $("#m-floor-norm").textContent = `${(m.space_heating_demand_kwh / fArea).toFixed(1)} kWh/m²`;
    $("#m-hp-elec").textContent = `${m.heat_pump_electricity_kwh} kWh`;
    $("#m-scop").textContent = `Seasonal COP: ${m.seasonal_cop}`;
    $("#m-pv-gen").textContent = `${m.solar_pv_generation_kwh} kWh`;
    $("#m-net-grid").textContent = `Net Grid: ${m.net_grid_electricity_kwh} kWh`;
    $("#m-stress").textContent = `${m.max_structural_stress_mpa} MPa`;
  }

  renderCharts(net) {
    if (!net) return;
    const hours = net.hours;
    const temps = net.temperatures;
    const flows = net.flows;
    const summary = net.summary || {};

    // 1. Temperature Chart
    const tempSeries = [];
    if (temps["ambient"]) tempSeries.push({ name: "Ambient", data: temps["ambient"], color: "#64748b" });
    if (temps["plant.tank.layer_0"]) tempSeries.push({ name: "Buffer Tank", data: temps["plant.tank.layer_0"], color: "#ef4444" });
    for (const [nid, arr] of Object.entries(temps)) {
      if (nid.endsWith(".air")) {
        const name = nid.replace(".air", "").replace(/_/g, " ");
        tempSeries.push({ name, data: arr, color: "#2563eb" });
      }
    }
    this.drawSVGLineChart("#chart-temperatures", hours, tempSeries, "°C", 0, 55);

    // 2. Power Chart (kW)
    const powerSeries = [];
    if (summary.hp_power_w) {
      powerSeries.push({ name: "HP Electricity", data: summary.hp_power_w.map((w) => w / 1000), color: "#f59e0b" });
    }
    if (summary.pv_power_w) {
      powerSeries.push({ name: "Solar PV", data: summary.pv_power_w.map((w) => w / 1000), color: "#10b981" });
    }
    // Aggregate heating demand kW
    const heatFlowsKW = new Array(hours.length).fill(0);
    for (const [lid, farr] of Object.entries(flows)) {
      if (lid.endsWith(".emitter")) {
        for (let i = 0; i < hours.length; i++) heatFlowsKW[i] += Math.max(0, (farr[i] || 0) / 1000);
      }
    }
    powerSeries.push({ name: "Heat Demand", data: heatFlowsKW, color: "#dc2626" });
    this.drawSVGLineChart("#chart-powers", hours, powerSeries, "kW", 0, Math.max(8, ...heatFlowsKW));
  }

  drawSVGLineChart(containerSel, xVals, seriesList, unit, yMin, yMax) {
    const box = $(containerSel);
    if (!box) return;
    const w = box.clientWidth || 550;
    const h = 260;
    const padL = 45, padR = 15, padT = 20, padB = 30;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const scaleX = (x) => padL + (x / (xVals[xVals.length - 1] || 24)) * plotW;
    const scaleY = (y) => padT + plotH - ((y - yMin) / Math.max(1, yMax - yMin)) * plotH;

    let svg = `<svg viewBox="0 0 ${w} ${h}">`;
    // Grid lines
    for (let y = yMin; y <= yMax; y += (yMax - yMin) / 4) {
      const sy = scaleY(y);
      svg += `<line x1="${padL}" y1="${sy}" x2="${w - padR}" y2="${sy}" stroke="#e2e8f0" stroke-width="1"/>`;
      svg += `<text x="${padL - 6}" y="${sy + 4}" font-size="10" font-family="'Fira Mono',monospace" text-anchor="end" fill="#64748b">${Math.round(y)} ${unit}</text>`;
    }
    for (let x = 0; x <= 24; x += 6) {
      const sx = scaleX(x);
      svg += `<line x1="${sx}" y1="${padT}" x2="${sx}" y2="${h - padB}" stroke="#e2e8f0" stroke-width="1"/>`;
      svg += `<text x="${sx}" y="${h - padB + 14}" font-size="10" font-family="'Fira Mono',monospace" text-anchor="middle" fill="#64748b">${x}h</text>`;
    }

    // Series lines
    seriesList.forEach((s) => {
      let pathD = "";
      for (let i = 0; i < xVals.length; i++) {
        const sx = scaleX(xVals[i]);
        const sy = scaleY(s.data[i] || 0);
        pathD += (i === 0 ? `M ${sx} ${sy}` : ` L ${sx} ${sy}`);
      }
      svg += `<path d="${pathD}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linecap="round"/>`;
    });

    // Legend
    let legX = padL + 10;
    seriesList.forEach((s) => {
      svg += `<rect x="${legX}" y="${padT - 14}" width="10" height="3" fill="${s.color}"/>`;
      svg += `<text x="${legX + 14}" y="${padT - 11}" font-size="10" font-family="'IBM Plex Sans',sans-serif" font-weight="600" fill="#334155">${s.name}</text>`;
      legX += (s.name.length * 7) + 30;
    });

    svg += "</svg>";
    box.innerHTML = svg;
  }

  // ------------------------------------------------------------------------------------------------- 2D thermal bridge
  async runThermalBridge() {
    const jType = $("#tb-junction-type")?.value || "wall_corner";
    try {
      const res = await fetch("/api/simulate/thermal-bridge", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ junction_type: jType, t_inside: 20.0, t_outside: 0.0, grid_resolution: 30 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      $("#tb-val-psi").textContent = `${data.linear_thermal_transmittance_psi.toFixed(3)} W/(m·K)`;
      $("#tb-val-tsimin").textContent = `${data.t_si_min.toFixed(1)} °C`;
      $("#tb-val-frsi").textContent = data.f_rsi.toFixed(2);
      $("#tb-val-l2d").textContent = `${data.heat_flow_rate_w_m.toFixed(2)} W/m`;

      const riskBadge = $("#tb-val-risk");
      if (data.condensation_risk) {
        riskBadge.textContent = `CONDENSATION RISK (f_Rsi < 0.70)`;
        riskBadge.className = "badge";
        riskBadge.style.background = "#fee2e2";
        riskBadge.style.color = "#b91c1c";
      } else {
        riskBadge.textContent = `PASS (f_Rsi ≥ 0.70)`;
        riskBadge.className = "badge badge-success";
        riskBadge.style.background = "";
        riskBadge.style.color = "";
      }

      this.renderThermalBridgeContour(data.temperature_grid);
    } catch (e) {
      this.toast(`Thermal bridge error: ${e.message}`);
    }
  }

  renderThermalBridgeContour(grid) {
    const canvas = $("#tb-contour-canvas");
    if (!canvas || !grid) return;
    const ctx = canvas.getContext("2d");
    const ny = grid.length;
    const nx = grid[0].length;
    const cellW = canvas.width / nx;
    const cellH = canvas.height / ny;

    for (let r = 0; r < ny; r++) {
      for (let c = 0; c < nx; c++) {
        const t = grid[r][c];
        const tNorm = Math.min(1.0, Math.max(0.0, (t - 0.0) / 20.0));
        ctx.fillStyle = this.colormap(tNorm);
        ctx.fillRect(c * cellW, (ny - 1 - r) * cellH, cellW + 0.5, cellH + 0.5);
      }
    }
  }

  // ------------------------------------------------------------------------------------------------- 1D wall conduction
  async runWall1D() {
    try {
      const res = await fetch("/api/simulate/wall-1d", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ construction: "brick_cavity", t_inside: 21.0, t_outside: 0.0, duration_hours: 24.0 }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      $("#wall1d-uval").textContent = data.u_value.toFixed(2);
      this.renderWall1DLayers(data);
      this.renderWall1DCanvas(data);
    } catch (e) {
      this.toast(`1D wall error: ${e.message}`);
    }
  }

  renderWall1DLayers(data) {
    const tbody = $("#wall1d-layers-table tbody");
    if (!tbody) return;
    tbody.innerHTML = "";
    const uniqueMats = [...new Set(data.materials)];
    uniqueMats.forEach((mat, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${idx + 1}</td><td><strong>${mat}</strong></td><td>-- mm</td><td>--</td>`;
      tbody.append(tr);
    });
  }

  renderWall1DCanvas(data) {
    const canvas = $("#wall1d-canvas");
    if (!canvas || !data) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const padL = 50, padR = 30, padT = 30, padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    const xMax = data.x_positions_m[data.x_positions_m.length - 1];
    const scaleX = (x) => padL + (x / xMax) * plotW;
    const scaleY = (t) => padT + plotH - ((t - 0.0) / 25.0) * plotH;

    // Draw axes
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);

    // Draw temperature curves across depth at multiple time steps
    const numCurves = Math.min(6, data.temperatures.length);
    const stepInterval = Math.floor(data.temperatures.length / numCurves);

    for (let cIdx = 0; cIdx < numCurves; cIdx++) {
      const row = data.temperatures[cIdx * stepInterval];
      ctx.beginPath();
      for (let i = 0; i < data.x_positions_m.length; i++) {
        const sx = scaleX(data.x_positions_m[i]);
        const sy = scaleY(row[i]);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      }
      ctx.strokeStyle = `rgba(37, 99, 235, ${0.3 + (cIdx / numCurves) * 0.7})`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.fillStyle = "#0f172a";
    ctx.font = "11px 'Space Grotesk', sans-serif";
    ctx.fillText("Interior Surface (21°C)", padL, padT - 10);
    ctx.fillText("Exterior (0°C)", w - padR - 70, padT - 10);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new SimfullApp();
  app.start();
});
