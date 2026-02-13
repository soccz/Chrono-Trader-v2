
import os
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
STATIC_DIR = "/home/soccz/.gemini/antigravity/scratch/mnt_20t/main/gan_t/static/analysis"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# 1. Attention Map
plt.figure(figsize=(6, 6))
data = np.random.rand(10, 10)
plt.imshow(data, cmap='viridis', aspect='auto')
plt.title("Self-Attention Weights (Head 1)")
plt.colorbar()
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/attention_map.png")
plt.close()

# 2. Gate Status (Regime Performance)
plt.figure(figsize=(6, 4))
x = np.linspace(0, 10, 100)
y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
y2 = np.cos(x) + np.random.normal(0, 0.1, 100)
plt.plot(x, y1, label="Trend (Transformer)", color="#3498db")
plt.plot(x, y2, label="Pattern (CNN)", color="#e74c3c")
plt.fill_between(x, y1, alpha=0.1, color="#3498db")
plt.fill_between(x, y2, alpha=0.1, color="#e74c3c")
plt.title("Gating Mechanism Response")
plt.legend()
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/regime_performance.png")
plt.close()

# 3. Prototype Matching
plt.figure(figsize=(6, 6))
plt.scatter(np.random.rand(50), np.random.rand(50), c='purple', alpha=0.6, label='Latent Prototypes')
plt.scatter([0.5], [0.5], c='red', s=100, marker='*', label='Current State')
plt.title("Latent Space Prototypes")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/prototype_logic.png")
plt.close()

# 4. Uncertainty
plt.figure(figsize=(6, 4))
x = np.arange(20)
y = np.random.rand(20) * 100
err = np.random.rand(20) * 20
plt.errorbar(x, y, yerr=err, fmt='o', ecolor='red', capsize=5, label='Price Prediction')
plt.title("Prediction Uncertainty (Monte Carlo)")
plt.tight_layout()
plt.savefig(f"{STATIC_DIR}/uncertainty.png")
plt.close()

print("✅ Dummy plots generated in", STATIC_DIR)
