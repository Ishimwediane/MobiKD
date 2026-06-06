"""
Academic Flowchart: MobiKD Application Area Pipeline
Generates a Springer-style, grayscale MLOps pipeline showing data preparation,
model development, edge deployment, and mobile application stages.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(22, 13))
ax.set_xlim(0, 22); ax.set_ylim(0, 11.5); ax.axis('off')
fig.patch.set_facecolor('white')

def T(x, y, s, sz=9.5, bold=False, ha='center', va='center'):
    ax.text(x, y, s, fontsize=sz, color='black',
        fontweight='bold' if bold else 'normal',
        ha=ha, va=va, zorder=8, fontfamily='DejaVu Sans')

def A(x1, y1, x2, y2, lw=1.2, ls='solid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color='black', lw=lw,
            linestyle=ls, connectionstyle='arc3,rad=0.0'), zorder=6)

def R(x, y, w, h, fc, ec, lw=1.2, z=2, radius=0.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw, zorder=z))

def draw_cylinder(x, y, w, h, label):
    rect = mpatches.Rectangle((x, y), w, h, fc='#FFFFFF', ec='#000000', lw=1.5, zorder=2)
    ax.add_patch(rect)
    ellipse_b = mpatches.Arc((x + w/2, y), w, h*0.25, angle=0, theta1=180, theta2=360, color='#000000', lw=1.5, zorder=3)
    ax.add_patch(ellipse_b)
    ellipse_t = mpatches.Ellipse((x + w/2, y + h), w, h*0.25, fc='#EAEAEA', ec='#000000', lw=1.5, zorder=3)
    ax.add_patch(ellipse_t)
    ax.plot([x, x], [y, y + h], color='#000000', lw=1.5, zorder=3)
    ax.plot([x + w, x + w], [y, y + h], color='#000000', lw=1.5, zorder=3)
    T(x + w/2, y + h/2, label, sz=9.5, bold=True)

def draw_box(x, y, w, h, label, header=None, fill='#FFFFFF'):
    rect = mpatches.Rectangle((x, y), w, h, fc=fill, ec='#000000', lw=1.5, zorder=3)
    ax.add_patch(rect)
    if header:
        hdr_rect = mpatches.Rectangle((x, y + h - 0.35), w, 0.35, fc='#EAEAEA', ec='#000000', lw=1.5, zorder=3)
        ax.add_patch(hdr_rect)
        T(x + w/2, y + h - 0.17, header, sz=8.5, bold=True)
        T(x + w/2, y + (h - 0.35)/2, label, sz=8.5)
    else:
        T(x + w/2, y + h/2, label, sz=9)

def draw_diamond(cx, cy, w, h, label):
    pts = [[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]]
    poly = Polygon(pts, closed=True, fc='#FFFFFF', ec='#000000', lw=1.5, zorder=3)
    ax.add_patch(poly)
    T(cx, cy, label, sz=9, bold=True)

def draw_chevron(x, y, w, h, text):
    n = 0.35
    if x == 0.5:
        pts = [[x, y], [x + w - n, y], [x + w, y + h/2], [x + w - n, y + h], [x, y + h]]
    elif x > 15.0:
        pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x + n, y + h/2]]
    else:
        pts = [[x, y], [x + w - n, y], [x + w, y + h/2], [x + w - n, y + h], [x, y + h], [x + n, y + h/2]]
        
    poly = Polygon(pts, closed=True, fc='#EAEAEA', ec='#000000', lw=1.5, zorder=3)
    ax.add_patch(poly)
    T(x + w/2, y + h/2, text, sz=11, bold=True)

# ─────────────────────────────────────────────
# 1. CHEVRON HEADERS
# ─────────────────────────────────────────────
draw_chevron(0.5, 10.1, 5.0, 0.9, '1. Process & Prepare Data')
draw_chevron(5.5, 10.1, 6.2, 0.9, '2. Develop MobiKD Transfer Models')
draw_chevron(11.7, 10.1, 4.2, 0.9, '3. Edge Deployment')
draw_chevron(15.9, 10.1, 5.6, 0.9, '4. Mobile Application')

# ─────────────────────────────────────────────
# 2. PHASE BACKGROUND REGIONS
# ─────────────────────────────────────────────
R(0.4, 0.3, 5.0, 9.6, '#FCFCFC', '#000000', lw=1.0, z=1, radius=0.1)
R(5.5, 0.3, 6.2, 9.6, '#FCFCFC', '#000000', lw=1.0, z=1, radius=0.1)
R(11.7, 0.3, 4.2, 9.6, '#FCFCFC', '#000000', lw=1.0, z=1, radius=0.1)
R(15.9, 0.3, 5.7, 9.6, '#FCFCFC', '#000000', lw=1.0, z=1, radius=0.1)

# ─────────────────────────────────────────────
# COLUMN 1: DATA PREPARATION
# ─────────────────────────────────────────────
draw_cylinder(0.8, 8.0, 2.0, 1.2, 'PlantVillage\nPotato Leaves')
draw_cylinder(3.0, 8.0, 2.0, 1.2, 'Rejection Set\n(Non-Leaf Images)')

draw_box(0.8, 6.0, 4.2, 1.2, 'Data Preprocessing\n- Center Crop & Resize\n  (32x32 for Leaf, 96x96 for Disease)\n- Normalization to [0, 1]')
draw_box(0.8, 4.4, 4.2, 1.1, 'Robustness Augmentation\n- Gaussian Blur | Sensor Noise\n- Low-Light Degradation\n- Combined Degradations')

# Connections in Column 1
ax.plot([1.8, 1.8], [8.0, 7.2], color='black', lw=1.2)
A(1.8, 7.2, 1.8, 7.2)
ax.plot([4.0, 4.0], [8.0, 7.2], color='black', lw=1.2)
A(4.0, 7.2, 4.0, 7.2)
A(2.9, 6.0, 2.9, 5.5)

# Route augmented training data to model training columns
ax.plot([2.9, 2.9, 7.2, 7.2], [4.4, 3.7, 3.7, 5.2], color='black', lw=1.2, ls='--')
A(7.2, 3.7, 7.2, 5.2)
ax.plot([7.2, 10.0, 10.0], [3.7, 3.7, 5.2], color='black', lw=1.2, ls='--')
A(10.0, 3.7, 10.0, 5.2)
T(5.0, 3.9, 'Augmented Training Data', sz=8, bold=True)

# ─────────────────────────────────────────────
# COLUMN 2: MODEL DEVELOPMENT
# ─────────────────────────────────────────────
draw_box(6.0, 8.2, 5.2, 1.1, 'MobiKD Distilled Backbone\n(kd_mobilenetv2.keras)\n[FROZEN PRETRAINED BACKBONE]')

draw_box(6.0, 5.2, 2.4, 2.3, 'Binary Head:\nDense 128 (ReLU)\n→ Dropout (0.4)\n→ Dense 64 (ReLU)\n→ Dropout (0.3)\n→ Softmax (2 cls)\n\nPhase 1: Train Head\nPhase 2: Fine-tune Top 30', header='MobiKD Leaf Validator')
draw_box(8.8, 5.2, 2.4, 2.3, '4-Class Head:\nDense 128 (ReLU)\n→ Dropout (0.3)\n→ Softmax (4 cls)\n\nPhase 1: Train Head\nPhase 2: Fine-tune Top 30', header='MobiKD Potato Disease')

# Connect Backbone to Leaf Validator and Disease Classifier
A(7.2, 8.2, 7.2, 7.5)
A(10.0, 8.2, 10.0, 7.5)
T(8.6, 7.85, 'Transfer Backbone Weights', sz=8, bold=True)

# ─────────────────────────────────────────────
# COLUMN 3: EDGE DEPLOYMENT
# ─────────────────────────────────────────────
draw_box(12.1, 7.8, 3.4, 1.2, 'Convert MobiKD Models\n- Float16 Quantization\n- Offline Edge Serialization\n- Optimize for Mobile I/O')
draw_cylinder(12.1, 5.2, 3.4, 1.4, 'Model Registry\n(MobiKD TFLite Assets)')

# Connect Dev columns to TFLite converter
ax.plot([8.4, 8.6, 8.6, 11.6, 11.6], [5.5, 5.5, 4.8, 4.8, 8.4], color='black', lw=1.2)
ax.plot([11.2, 11.6, 11.6], [5.5, 5.5, 4.8], color='black', lw=1.2)
A(11.6, 8.4, 12.1, 8.4)
T(10.6, 5.0, 'Export MobiKD Models', sz=8, bold=True, ha='left')

# Connect TFLite converter to Registry
A(13.8, 7.8, 13.8, 6.6)

# Route Registry to Mobile Application
ax.plot([13.8, 13.8, 16.3, 16.3], [5.2, 4.7, 4.7, 7.55], color='black', lw=1.2, ls='--')
A(16.3, 7.55, 16.65, 7.55)
ax.plot([16.3, 16.3], [4.7, 3.55], color='black', lw=1.2, ls='--')
A(16.3, 3.55, 16.65, 3.55)
T(15.1, 4.5, 'Deploy MobiKD TFLite Models', sz=8, bold=True)

# ─────────────────────────────────────────────
# COLUMN 4: MOBILE APPLICATION
# ─────────────────────────────────────────────
draw_box(16.65, 8.6, 4.2, 0.8, 'User Input Photo\n(Mobile Camera Capture)')
draw_box(16.65, 7.0, 4.2, 1.1, 'Stage 1: MobiKD Leaf Validator\n- Input: 32×32 pixels\n- Runs local TFLite model', header='Offline Inference Gate 1')

A(18.75, 8.6, 18.75, 8.1)

draw_diamond(18.75, 5.25, 2.4, 1.2, 'Is Leaf?\n(Conf \u2265 50%)')
A(18.75, 7.0, 18.75, 5.85)

# Reject path (No) - Placed on the right of the column to avoid line intersection
draw_box(20.05, 4.85, 1.35, 0.8, 'REJECT PHOTO\n(Prompt: Retake)')
A(19.95, 5.25, 20.05, 5.25)
T(20.0, 5.45, 'No', sz=8.5, bold=True)

# Accept path (Yes)
draw_box(16.65, 3.0, 4.2, 1.1, 'Stage 2: MobiKD Potato Disease\n- Input: 96×96 pixels\n- Runs local TFLite model', header='Offline Inference Gate 2')
A(18.75, 4.65, 18.75, 4.1)
T(19.15, 4.4, 'Yes', sz=8.5, bold=True)

# Diagnosis Box
draw_box(16.65, 1.0, 4.2, 1.1, 'Output Diagnosis & Advice\n[Early Blight]  |  [Late Blight]\n[Healthy]  |  [Not Potato Leaf]')
A(18.75, 3.0, 18.75, 2.1)

# ─────────────────────────────────────────────
# FEEDBACK LOOP
# ─────────────────────────────────────────────
ax.plot([18.75, 18.75, 1.8], [1.0, 0.25, 0.25], color='black', lw=1.2, ls='--')
A(1.8, 0.25, 1.8, 6.0, ls='--')
T(10.0, 0.4, 'Model Retraining / Active Learning Feedback Loop', sz=9.5, bold=True)


# Figure Caption
T(11.0, 11.2, 'Figure: End-to-end MLOps pipeline and edge application architecture for MobiKD leaf disease detection.', sz=11, bold=True)

plt.tight_layout(pad=0.2)
plt.savefig('docs/mobikd_application_flowchart.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: docs/mobikd_application_flowchart.png")

