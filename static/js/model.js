/**
 * Model Specification Logic - Blueprint & Live JSON
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log("Model Blueprint Loaded - Deep Space Glass Mode");

    // Initialize Mermaid
    if (window.mermaid) {
        mermaid.initialize({
            startOnLoad: true,
            theme: 'dark',
            securityLevel: 'loose',
            fontFamily: 'Inter',
        });
    }
});

// Main Initialization Logic
function initModelPage() {
    console.log("Model Blueprint Loaded - Deep Space Glass Mode");

    // Start Live Polling Immediately
    fetchEnsembleStatus();
    fetchGateStatus();
    setInterval(fetchEnsembleStatus, 5000); // 5 seconds
    setInterval(fetchGateStatus, 3000); // 3 seconds

    // Attempt to enable PanZoom periodically until successful
    const panZoomInterval = setInterval(() => {
        // enablePanZoom now returns true on success, false on retry
        if (window.enablePanZoom()) {
            clearInterval(panZoomInterval);
        }
    }, 1000);
}

// Handle DOM Ready (supports late script loading)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initModelPage);
} else {
    initModelPage();
}

// Expose PanZoom for module script to call
// Enhanced PanZoom Logic with Robust Polling
window.enablePanZoom = function () {
    const visualizer = document.querySelector('.arch-visualizer');
    const mermaidContainer = document.querySelector('.mermaid');

    // 1. Sanity Check
    if (!visualizer || !mermaidContainer) return false;

    // 2. Find SVG (Mermaid might take time to generate it)
    const svgElement = mermaidContainer.querySelector('svg');

    if (!svgElement) {
        // SVG not generated yet
        return false;
    }

    // 3. Check if SVG has dimensions (rendered)
    const bbox = svgElement.getBoundingClientRect();
    if (bbox.width === 0 || bbox.height === 0) {
        return false; // Still hidden or unrendered
    }

    // 4. Initialize PanZoom
    // Prepare container and SVG for full interaction
    visualizer.style.overflow = 'hidden'; // Required for pan-zoom
    svgElement.style.width = '100%';
    svgElement.style.height = '100%';
    svgElement.style.maxWidth = 'none';

    if (window.svgPanZoom) {
        // Check if already initialized to avoid double-binding
        if (svgElement.classList.contains('panzoom-enabled')) {
            return true;
        }

        try {
            window.panZoomInstance = svgPanZoom(svgElement, {
                zoomEnabled: true,
                controlIconsEnabled: true,
                fit: true,
                center: true,
                minZoom: 0.1, // Allow zooming out further
                maxZoom: 10,
                dblClickZoomEnabled: false // Prevent interference
            });

            svgElement.classList.add('panzoom-enabled');
            console.log("✅ Pan/Zoom enabled successfully on Architecture View");

            // Fix initial resize if needed
            window.panZoomInstance.resize();
            window.panZoomInstance.fit();
            window.panZoomInstance.center();

            // User Feedback: "Too small". Zoom in a bit more than 'fit'
            window.panZoomInstance.zoomBy(1.2);
            window.panZoomInstance.center(); // Re-center after zoom

            return true;
        } catch (e) {
            console.warn("svgPanZoom init failed (will retry):", e);
            return false;
        }
    }
    return false; // Lib not ready
};

/* --- 1. Ensemble Status (5 Models) --- */
async function fetchEnsembleStatus() {
    try {
        console.log("Fetching ensemble status...");
        const response = await fetch('/api/model/ensemble-status');
        const data = await response.json();
        console.log("Ensemble data received:", data);

        if (data.success) {
            renderEnsembleGrid(data.data);
        }
    } catch (error) {
        console.error("Failed to fetch ensemble status:", error);
        // Show error in grid
        const grid = document.getElementById('ensemble-grid');
        if (grid) grid.innerHTML = `<div class="text-danger p-3">Connection Error: ${error.message}</div>`;
    }
}

function renderEnsembleGrid(data) {
    const grid = document.getElementById('ensemble-grid');
    if (!grid) return;

    try {
        const weights = data.weights || [0.2, 0.2, 0.2, 0.2, 0.2];
        const stats = data.stats || {};
        const metadata = data.metadata || {};

        // Find leader (max weight)
        const maxWeight = Math.max(...weights);

        let html = '';

        weights.forEach((weight, index) => {
            const isLeader = weight === maxWeight && weight > 0;
            const modelKey = `model_${index}`;
            const modelStat = stats[modelKey] || { accuracy: 0.0 };
            const modelMeta = metadata[modelKey] || { role: 'General', last_updated: '-' };

            // Safety check for accuracy numericality
            let accVal = parseFloat(modelStat.accuracy);
            if (isNaN(accVal)) accVal = 0.0;
            const accuracy = (accVal * 100).toFixed(1);

            // Calculate weight percentage
            let weightVal = parseFloat(weight);
            if (isNaN(weightVal)) weightVal = 0.0;

            html += `
                <div class="ensemble-card ${isLeader ? 'active-leader' : ''}">
                    <!-- Leader Badge (Top Right) -->
                    ${isLeader ? '<div class="leader-badge">LEADER</div>' : ''}
                    
                    <div class="ensemble-icon">
                        <i class="bi bi-cpu${isLeader ? '-fill' : ''}"></i>
                    </div>
                    <div class="ensemble-name">모델 0${index + 1}</div>
                    
                    <div class="ensemble-metric">${accuracy}%</div>
                    <div class="ensemble-label mb-2">정확도 (Acc)</div>
                    
                    <hr class="my-2 opacity-25">
                    
                    <!-- New Metadata Section -->
                    <div class="model-role text-primary small fw-bold mb-1" style="font-size: 0.75rem;">
                        ${modelMeta.role}
                    </div>
                    <div class="model-date text-secondary position-relative" style="font-size: 0.65rem;">
                        <i class="bi bi-clock-history me-1"></i>${modelMeta.last_updated}
                    </div>
                    
                    <div class="mt-2 small text-secondary" style="font-size: 0.7em;">
                        가중치: ${(weightVal * 100).toFixed(0)}%
                    </div>
                </div>
            `;
        });

        grid.innerHTML = html;

    } catch (e) {
        console.error("Error rendering ensemble grid:", e);
        grid.innerHTML = `<div class="col-12 text-center text-danger py-3">
            <i class="bi bi-exclamation-triangle me-2"></i>
            렌더링 오류: ${e.message}
        </div>`;
    }
}

/* --- 2. Neural State Monitor (Gate) --- */
async function fetchGateStatus() {
    try {
        const response = await fetch('/api/model/gate-status');
        const data = await response.json();

        if (data.success) {
            updateGateVisualizer(data);
        }
    } catch (error) {
        console.error("Failed to fetch gate status:", error);
    }
}

function updateGateVisualizer(data) {
    const gateValue = parseFloat(data.gate_value || 0.5); // 0 (CNN) to 1 (Transformer)

    const transPct = (gateValue * 100).toFixed(0);
    const cnnPct = (100 - transPct).toFixed(0);

    const barTrans = document.getElementById('gate-bar-trans');
    const barCnn = document.getElementById('gate-bar-cnn');
    const display = document.getElementById('gate-value-display');

    if (barTrans && barCnn) {
        barTrans.style.width = `${transPct}%`;
        barCnn.style.width = `${cnnPct}%`;
    }

    if (display) {
        let label = "하이브리드";
        if (gateValue > 0.6) label = "추세 추종형 (Trend)";
        if (gateValue < 0.4) label = "패턴 매칭형 (Pattern)";

        display.innerHTML = `${label}: ${gateValue.toFixed(2)}`;
    }
}

