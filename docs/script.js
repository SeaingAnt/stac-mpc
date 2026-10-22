/**
 * STAC-MPC Benchmark Explorer Controller
 * Drone obstacle avoidance parameter sliders, highlighted buttons, and video streaming.
 */

// Parameter configurations for Single-Agent Drone
const S1_NOISE_LEVELS = [
    { label: "None", valueStr: "\\sigma = 0.0" },
    { label: "Moderate", valueStr: "\\sigma = 0.25" },
    { label: "Severe", valueStr: "\\sigma = 0.50" }
];

const S1_DIST_LEVELS = [
    { label: "None", valueStr: "d = 0.0\\,\\text{N}" },
    { label: "Moderate", valueStr: "d = 0.2\\,\\text{N}" },
    { label: "Severe", valueStr: "d = 0.4\\,\\text{N}" }
];

const S1_MATRIX = {
    "0_0": { video: "assets/obstacle0.0-0.0.mp4", metric: "⚡ Nominal flight: Both STAC-MPC and baseline navigate smoothly without external disturbances." },
    "0_1": { video: "assets/obstacle0.0-0.2.mp4", metric: "⚡ Moderate disturbance: Baseline shows overshoot and trajectory deviation; STAC-MPC contracts tracking error." },
    "0_2": { video: "assets/obstacle.mp4", metric: "⚡ Severe disturbance: Baseline exhibits sustained oscillations; STAC-MPC guarantees obstacle clearance boundary." },
    "1_0": { video: "assets/obstacle0.25-0.0.mp4", metric: "⚡ Sensor noise: Baseline cost predictions jitter; STAC-MPC contraction bounds network sensitivity." },
    "1_1": { video: "assets/obstacle0.25-0.2.mp4", metric: "⚡ Combined noise & disturbance: Baseline drifts dangerously near obstacle; STAC-MPC maintains certified safe path." },
    "1_2": { video: "assets/obstacle0.25-0.4.mp4", metric: "⚡ Severe disturbance with noise: Baseline struggles with chattering; STAC-MPC enforces safe contraction envelope." },
    "2_0": { video: "assets/obstacle0.5-0.0.mp4", metric: "⚡ High sensor noise: Destabilizes baseline cost matrix; STAC-MPC filters high-frequency noise." },
    "2_1": { video: "assets/obstacle0.5-0.2.mp4", metric: "⚡ High noise & disturbance: Baseline commands chatter heavily; STAC-MPC preserves smooth contraction." },
    "2_2": { video: "assets/obstacle0.5-0.4.mp4", metric: "⚡ Extreme stress regime: Baseline risks obstacle collision; STAC-MPC guarantees invariant safe flight." }
};

function setTicksActive(containerId, activeVal) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const ticks = container.querySelectorAll(".tick");
    ticks.forEach(function (tick) {
        const val = parseInt(tick.getAttribute("data-val"), 10);
        if (val === activeVal) {
            tick.classList.add("active");
            tick.setAttribute("aria-pressed", "true");
        } else {
            tick.classList.remove("active");
            tick.setAttribute("aria-pressed", "false");
        }
    });
}

function updateVideoSource(videoEl, sourceEl, newSrc, isInitial) {
    if (!videoEl || !sourceEl) return;
    try {
        const currentSrc = sourceEl.getAttribute("src");
        if (currentSrc !== newSrc) {
            const wasPlaying = !videoEl.paused;
            sourceEl.setAttribute("src", newSrc);
            videoEl.load();
            if (wasPlaying) {
                videoEl.play().catch(function () {});
            }
        } else if (!isInitial) {
            videoEl.currentTime = 0;
            videoEl.play().catch(function () {});
        }
    } catch (e) {
        console.debug("Video update error:", e);
    }
}

function retypesetMath(elements) {
    try {
        if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
            window.MathJax.typesetPromise(elements.filter(Boolean)).catch(function () {});
        }
    } catch (e) {}
}

function updateScenario1(isInitial) {
    const s1NoiseSlider = document.getElementById("s1NoiseSlider");
    const s1DistSlider = document.getElementById("s1DistSlider");
    const s1NoiseBadge = document.getElementById("s1NoiseBadge");
    const s1DistBadge = document.getElementById("s1DistBadge");
    const s1Status = document.getElementById("s1Status");
    const videoCol1 = document.getElementById("videoCol1");
    const sourceCol1 = document.getElementById("sourceCol1");

    if (!s1NoiseSlider || !s1DistSlider) return;
    const nIdx = parseInt(s1NoiseSlider.value, 10) || 0;
    const dIdx = parseInt(s1DistSlider.value, 10) || 0;
    const item = S1_MATRIX[nIdx + "_" + dIdx] || S1_MATRIX["1_1"];

    if (s1NoiseBadge && S1_NOISE_LEVELS[nIdx]) {
        s1NoiseBadge.innerHTML = S1_NOISE_LEVELS[nIdx].label + " ($" + S1_NOISE_LEVELS[nIdx].valueStr + "$)";
    }
    if (s1DistBadge && S1_DIST_LEVELS[dIdx]) {
        s1DistBadge.innerHTML = S1_DIST_LEVELS[dIdx].label + " ($" + S1_DIST_LEVELS[dIdx].valueStr + "$)";
    }
    if (s1Status && item) {
        s1Status.innerHTML = item.metric;
    }

    setTicksActive("s1NoiseTicks", nIdx);
    setTicksActive("s1DistTicks", dIdx);

    if (videoCol1 && sourceCol1 && item) {
        updateVideoSource(videoCol1, sourceCol1, item.video, isInitial);
    }

    retypesetMath([s1NoiseBadge, s1DistBadge, s1Status]);
}

function setScenarioParam(scenario, param, val) {
    const numericVal = parseInt(val, 10);
    if (param === "noise") {
        const slider = document.getElementById("s1NoiseSlider");
        if (slider) slider.value = numericVal;
    } else if (param === "dist") {
        const slider = document.getElementById("s1DistSlider");
        if (slider) slider.value = numericVal;
    }
    updateScenario1(false);
}

window.setScenarioParam = setScenarioParam;
window.setDroneParam = function (param, val) { setScenarioParam(1, param, val); };
window.updateScenario1 = updateScenario1;
window.updateScenario2 = function () {};

function initBenchmarkExplorer() {
    // BibTeX copy
    const copyBtn = document.getElementById("copyBibtex");
    const bibtexCode = document.getElementById("bibtexCode");
    if (copyBtn && bibtexCode) {
        copyBtn.addEventListener("click", async function () {
            try {
                await navigator.clipboard.writeText(bibtexCode.innerText);
                const originalText = copyBtn.innerText;
                copyBtn.innerText = "Copied!";
                copyBtn.style.background = "rgba(16, 185, 129, 0.2)";
                copyBtn.style.borderColor = "rgba(16, 185, 129, 0.5)";
                setTimeout(function () {
                    copyBtn.innerText = originalText;
                    copyBtn.style.background = "rgba(255, 255, 255, 0.1)";
                    copyBtn.style.borderColor = "rgba(255, 255, 255, 0.2)";
                }, 2000);
            } catch (err) {
                copyBtn.innerText = "Failed";
            }
        });
    }

    // Sliders
    const s1NoiseSlider = document.getElementById("s1NoiseSlider");
    const s1DistSlider = document.getElementById("s1DistSlider");

    if (s1NoiseSlider) s1NoiseSlider.addEventListener("input", function () { updateScenario1(false); });
    if (s1DistSlider) s1DistSlider.addEventListener("input", function () { updateScenario1(false); });

    // Tick buttons (event delegation)
    document.querySelectorAll(".compact-ticks").forEach(function (container) {
        container.addEventListener("click", function (e) {
            const btn = e.target.closest(".tick");
            if (!btn) return;
            const val = parseInt(btn.getAttribute("data-val"), 10);
            if (isNaN(val)) return;
            if (container.id === "s1NoiseTicks") setScenarioParam(1, "noise", val);
            else if (container.id === "s1DistTicks") setScenarioParam(1, "dist", val);
        });
    });

    // Initial render
    updateScenario1(true);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initBenchmarkExplorer);
} else {
    initBenchmarkExplorer();
}
