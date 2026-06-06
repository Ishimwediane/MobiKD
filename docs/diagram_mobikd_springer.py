"""
Academic Flowchart: MobiKD Distillation Training Pipeline
Based on the clean Springer academic publication flowchart format.
No bright colors. Solid line-art, grayscale, clear engineering connections.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(18, 12))
ax.set_xlim(0, 23); ax.set_ylim(0, 12); ax.axis('off')
fig.patch.set_facecolor('white')

def T(x, y, s, sz=9.5, bold=False, ha='center', va='center'):
    ax.text(x, y, s, fontsize=sz, color='black',
        fontweight='bold' if bold else 'normal',
        ha=ha, va=va, zorder=8, fontfamily='DejaVu Sans')

def A(x1, y1, x2, y2, lw=1.2, ls='solid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color='black', lw=lw,
            linestyle=ls, connectionstyle='arc3,rad=0.0'), zorder=6)

def draw_cylinder(x, y, w, h, label):
    rect = mpatches.Rectangle((x, y), w, h, fc='#FFFFFF', ec='#000000', lw=1.5, zorder=2)
    ax.add_patch(rect)
    ellipse_b = mpatches.Arc((x + w/2, y), w, h*0.25, angle=0, theta1=180, theta2=360, color='#000000', lw=1.5, zorder=3)
    ax.add_patch(ellipse_b)
    ellipse_t = mpatches.Ellipse((x + w/2, y + h), w, h*0.25, fc='#EAEAEA', ec='#000000', lw=1.5, zorder=3)
    ax.add_patch(ellipse_t)
    ax.plot([x, x], [y, y + h], color='#000000', lw=1.5, zorder=3)
    ax.plot([x + w, x + w], [y, y + h], color='#000000', lw=1.5, zorder=3)
    T(x + w/2, y + h/2, label, sz=9, bold=True)

def draw_box(x, y, w, h, label, header=None, fill='#FFFFFF'):
    rect = mpatches.Rectangle((x, y), w, h, fc=fill, ec='#000000', lw=1.5, zorder=3)
    ax.add_patch(rect)
    if header:
        hdr_rect = mpatches.Rectangle((x, y + h - 0.35), w, 0.35, fc='#F0F0F0', ec='#000000', lw=1.5, zorder=3)
        ax.add_patch(hdr_rect)
        T(x + w/2, y + h - 0.17, header, sz=8.5, bold=True)
        T(x + w/2, y + (h - 0.35)/2, label, sz=8.5)
    else:
        T(x + w/2, y + h/2, label, sz=9)

def draw_diamond(cx, cy, w, h, label):
    pts = [[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]]
    poly = Polygon(pts, closed=True, fc='#FFFFFF', ec='#000000', lw=1.5, zorder=3)
    ax.add_patch(poly)
    T(cx, cy, label, sz=8.5, bold=True)

# ─────────────────────────────────────────────
# 1. DATA PREPARATION (Left Columns)
# ─────────────────────────────────────────────
draw_cylinder(0.8, 7.5, 2.0, 1.8, 'CIFAR-100\nDataset')
draw_box(3.4, 7.6, 2.2, 1.2, 'Filter 5 Tree Classes\nMap to Binary:\nTree vs Not-Tree')

A(2.8, 8.4, 3.4, 8.2)

draw_cylinder(6.4, 9.5, 2.0, 1.0, 'Training Set\n(Binary labels)')
draw_cylinder(6.4, 5.0, 2.0, 1.0, 'Testing Set\n(Binary labels)')

# Split connection
ax.plot([5.6, 6.0, 6.0], [8.2, 8.2, 10.0], color='black', lw=1.2)
A(6.0, 10.0, 6.4, 10.0)
ax.plot([5.6, 6.0, 6.0], [8.2, 8.2, 5.5], color='black', lw=1.2)
A(6.0, 5.5, 6.4, 5.5)

# Robustness Mix
draw_box(9.0, 9.5, 2.8, 1.0, 'Robustness Mix (4x size):\nClean + Blur +\nNoise + Low-Light')
A(8.4, 10.0, 9.0, 10.0)

# ─────────────────────────────────────────────
# 2. DISTILLATION MODELS & TRAINING LOOP BORDER
# ─────────────────────────────────────────────
# Highlighted background box for the Knowledge Distillation Loop
kd_rect = mpatches.Rectangle((12.2, 0.8), 10.0, 10.4, fill=True, fc='#FAFAFA', ec='#000000', ls='--', lw=1.5, zorder=1)
ax.add_patch(kd_rect)
T(17.2, 10.95, 'KNOWLEDGE DISTILLATION (KD) TRAINING LOOP', sz=10, bold=True)

draw_box(12.5, 9.5, 2.8, 1.1, 'ResNet50\n(Knowledge Source)\n[FROZEN]', header='Teacher Model')
draw_box(12.5, 7.4, 2.8, 1.1, 'MobileNetV2\n(Receives Knowledge)\n[TRAINABLE]', header='Student Model')

# Distribute training data to models
ax.plot([11.8, 12.1, 12.1], [10.0, 10.0, 10.05], color='black', lw=1.2)
A(12.1, 10.05, 12.5, 10.05)
ax.plot([11.8, 12.1, 12.1], [10.0, 10.0, 7.95], color='black', lw=1.2)
A(12.1, 7.95, 12.5, 7.95)

# ─────────────────────────────────────────────
# 3. TEMPERATURE SCALING & LOGITS
# ─────────────────────────────────────────────
draw_box(16.0, 9.5, 2.6, 1.1, 'Soft Targets:\np_t = softmax(logits_t / T)\nTemperature T = 4.0')
draw_box(16.0, 7.4, 2.6, 1.1, 'Soft Predictions:\np_s = softmax(logits_s / T)\nTemperature T = 4.0')
draw_box(16.0, 5.8, 2.6, 1.1, 'Hard Predictions:\np_student_hard\n(standard softmax)')

A(15.3, 10.05, 16.0, 10.05)
ax.plot([15.3, 15.6, 15.6], [7.95, 7.95, 7.95], color='black', lw=1.2)
A(15.6, 7.95, 16.0, 7.95)
ax.plot([15.3, 15.6, 15.6], [7.95, 7.95, 6.35], color='black', lw=1.2)
A(15.6, 6.35, 16.0, 6.35)

# ─────────────────────────────────────────────
# 4. LOSS BRANCHES
# ─────────────────────────────────────────────
draw_box(19.4, 8.3, 2.4, 1.0, 'KL Divergence Loss\nKL(p_t || p_s)')
draw_box(19.4, 5.8, 2.4, 1.0, 'Categorical\nCross-Entropy Loss\nCE(y_true, p_student_hard)')

# Soft targets/predictions to KL
ax.plot([17.3, 17.3], [9.5, 9.0], color='black', lw=1.2)
A(17.3, 9.0, 19.4, 9.0)

ax.plot([17.3, 17.3], [8.5, 8.6], color='black', lw=1.2)
A(17.3, 8.6, 19.4, 8.6)

# Hard predictions to CE
A(18.6, 6.35, 19.4, 6.3)

# Y_true ground truth labels from data to CE
ax.plot([10.4, 10.4, 19.0, 19.0], [9.5, 5.4, 5.4, 6.0], color='black', lw=1.2, ls='--')
A(19.0, 6.0, 19.4, 6.0)
T(14.7, 5.65, 'y_true (Ground Truth Labels)', sz=8)

# ─────────────────────────────────────────────
# 5. TOTAL LOSS & OPTIMIZATION LOOP
# ─────────────────────────────────────────────
draw_box(19.4, 3.0, 2.4, 1.2, 'Total Distillation Loss\nL = α·CE + (1-α)·T²·KL\nα = 0.3')

# Connect CE and KL to Total Loss
ax.plot([20.6, 20.6], [8.3, 4.2], color='black', lw=1.2)
A(20.6, 5.8, 20.6, 4.2)

# Diamond Decision
draw_diamond(16.0, 3.6, 2.4, 1.4, 'Lowest Loss\nor Max Epochs?')
A(19.4, 3.6, 17.2, 3.6)

# NO -> Backprop & Update
draw_box(12.5, 1.2, 2.8, 1.1, 'Gradient Backpropagation\n(Update student weights)', header='Adam Optimizer')
ax.plot([16.0, 16.0], [2.9, 1.75], color='black', lw=1.2)
A(16.0, 1.75, 15.3, 1.75)
T(16.3, 2.4, 'No', sz=9, bold=True)

# Loop backprop back up to Student Model
A(13.9, 2.3, 13.9, 7.4)
T(13.0, 4.7, 'Knowledge Transfer\n(Update student weights)', sz=8, bold=True)

# YES -> Save Model
draw_box(9.0, 3.0, 2.8, 1.1, 'Optimal MobiKD model')
A(14.8, 3.6, 11.8, 3.55)
T(13.3, 3.8, 'Yes', sz=9, bold=True)

# ─────────────────────────────────────────────
# 6. EVALUATION
# ─────────────────────────────────────────────
# Connect Testing Set to Optimal Model
ax.plot([7.4, 7.8, 7.8], [5.0, 5.0, 3.55], color='black', lw=1.2)
A(7.8, 3.55, 9.0, 3.55)
T(8.4, 4.2, 'Evaluate Accuracy', sz=8)

# Caption at bottom
T(11.5, 0.4, 'Figure: Technical flowchart of the proposed MobiKD knowledge distillation training architecture.', sz=10, bold=True)

plt.tight_layout(pad=0.2)
plt.savefig('docs/mobikd_springer_flowchart.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: docs/mobikd_springer_flowchart.png")
