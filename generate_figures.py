"""
generate_figures.py
===================
Generates all 6 documentation figures for the Academic Advisor Agent project.

Usage:
    python generate_figures.py

Output:
    figures/figure1_roles_architecture.png
    figures/figure2_role_detection_flowchart.png
    figures/figure3_rag_pipeline.png
    figures/figure4_performance_metrics.png
    figures/figure5_routing_accuracy.png
    figures/figure6_quality_ratings.png

Requirements:
    pip install matplotlib numpy
"""

import os
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Output directory ────────────────────────────────────────────
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 200

# ── Consistent color palette (matches ROLES dict colors) ────────
ROLE_COLORS = {
    "academic_guidance": "#534AB7",
    "career_planning":   "#0F6E56",
    "job_market":        "#185FA5",
    "higher_studies":    "#854F0B",
    "academic_progress": "#639922",
    "mentorship":        "#D85A30",
    "administrative":    "#993556",
}
ROLE_BG = {
    "academic_guidance": "#EEEDFE",
    "career_planning":   "#E1F5EE",
    "job_market":        "#E6F1FB",
    "higher_studies":    "#FAEEDA",
    "academic_progress": "#EAF3DE",
    "mentorship":        "#FAECE7",
    "administrative":    "#FBEAF0",
}
ROLES_META = [
    ("academic_guidance", "AG", "Academic Guidance\n& Course Planning",
     "Courses, electives, curriculum,\nsemester planning"),
    ("career_planning",   "CP", "Career Planning\n& Development",
     "Job paths, resume,\ninternships, skill development"),
    ("job_market",        "JM", "Job Market\n& Opportunities",
     "2025 hiring trends, salaries,\nhot skills, companies"),
    ("higher_studies",    "HS", "Higher Studies\n& Competitive Exams",
     "GATE, GRE, MS abroad,\nPhD, MBA, IIT MTech"),
    ("academic_progress", "AP", "Academic Progress\n& Performance",
     "CGPA, study strategies,\nexam preparation"),
    ("mentorship",        "MW", "Mentorship\n& Wellness",
     "Stress, burnout,\nmotivation, personal support"),
    ("administrative",    "AD", "Administrative\n& Logistics",
     "Forms, fees, scholarships,\ncertificates"),
]

# ════════════════════════════════════════════════════════════════
# FIGURE 1 — Seven Advisor Roles Architecture (hub-spoke)
# ════════════════════════════════════════════════════════════════

def figure1_roles_architecture():
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#F7F8FC")

    # Central hub
    cx, cy = 5, 5
    hub_radius = 1.05
    hub = plt.Circle((cx, cy), hub_radius, color="#1A1A2E", zorder=4)
    ax.add_patch(hub)
    ax.text(cx, cy, "Academic\nAdvisor\nAgent", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=5, linespacing=1.45)

    # Spoke positions — evenly spaced around hub
    n = len(ROLES_META)
    spoke_r = 3.8
    angles_deg = [90 - i * (360 / n) for i in range(n)]

    for i, (role_id, emoji, name, desc) in enumerate(ROLES_META):
        theta = np.radians(angles_deg[i])
        sx = cx + spoke_r * np.cos(theta)
        sy = cy + spoke_r * np.sin(theta)

        color  = ROLE_COLORS[role_id]
        bg     = ROLE_BG[role_id]

        # Draw spoke line
        # From edge of hub to edge of node box
        arrow_start_x = cx + hub_radius * np.cos(theta)
        arrow_start_y = cy + hub_radius * np.sin(theta)
        node_r = 1.15
        arrow_end_x = sx - node_r * np.cos(theta)
        arrow_end_y = sy - node_r * np.sin(theta)

        ax.annotate("", xy=(arrow_end_x, arrow_end_y),
                    xytext=(arrow_start_x, arrow_start_y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=2, mutation_scale=18),
                    zorder=2)

        # Role node box
        box_w, box_h = 2.5, 1.9
        box = FancyBboxPatch((sx - box_w / 2, sy - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.08",
                             facecolor=bg, edgecolor=color, linewidth=2.5, zorder=3)
        ax.add_patch(box)

        # Abbreviation badge (colored circle with 2-letter code)
        badge = plt.Circle((sx, sy + 0.58), 0.34, color=color, zorder=5)
        ax.add_patch(badge)
        ax.text(sx, sy + 0.58, emoji, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white", zorder=6)
        # Name
        ax.text(sx, sy + 0.05, name, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=color, zorder=5, linespacing=1.3)
        # Description
        ax.text(sx, sy - 0.62, desc, ha="center", va="center",
                fontsize=6.2, color="#444444", zorder=5, linespacing=1.25)

    ax.set_title("Seven Advisor Roles Architecture — Academic Advisor Agent",
                 fontsize=15, fontweight="bold", color="#1A1A2E", pad=18)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure1_roles_architecture.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════
# FIGURE 2 — Role Detection Algorithm Flowchart
# ════════════════════════════════════════════════════════════════

def _draw_box(ax, x, y, w, h, text, shape="rect",
              facecolor="#EEEDFE", edgecolor="#534AB7",
              fontsize=9, fontcolor="#1A1A2E", bold=False):
    """Draw a rectangle or diamond flowchart node."""
    if shape == "diamond":
        dx, dy = w / 2, h / 2
        diamond = plt.Polygon([[x, y + dy], [x + dx, y],
                                [x, y - dy], [x - dx, y]],
                               facecolor=facecolor, edgecolor=edgecolor, lw=2, zorder=3)
        ax.add_patch(diamond)
    else:
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.04",
                             facecolor=facecolor, edgecolor=edgecolor, lw=2, zorder=3)
        ax.add_patch(box)

    fw = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=fontcolor, fontweight=fw, zorder=4,
            multialignment="center", linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2, label="", color="#666666"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16))
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, fontsize=8, color=color, va="center")


def figure2_role_detection_flowchart():
    fig, ax = plt.subplots(figsize=(10, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F8FC")

    cx = 5      # center x
    bw = 6.0    # box width
    bh = 0.80   # box height

    # ── Nodes (y positions, top → bottom) ──
    nodes = [
        (18.5, "START:\nStudent Query + StudentProfile",
         "rect",  "#1A1A2E", "#FFFFFF", 9, True),
        (16.8, "Step 1\nLowercase the query → q = query.lower()",
         "rect",  "#EEEDFE", "#534AB7", 9, False),
        (15.0, "Step 2\nFor each role, scan its keyword list\n"
               "score += 1 + len(keyword.split()) × 0.5  per match",
         "rect",  "#E1F5EE", "#0F6E56", 8.5, False),
        (13.0, "Step 3a — Boost job_market (+2)\nif query contains:\n"
               '"vs", "versus", "better", "trending",\n"demand", "market", "2025"',
         "rect",  "#E6F1FB", "#185FA5", 8, False),
        (11.1, "Step 3b — Boost higher_studies (+3)\nif query contains:\n"
               '"gate", "gre", "gmat", "ielts", "toefl",\n"nus", "tum", "cmu"',
         "rect",  "#FAEEDA", "#854F0B", 8, False),
        (9.2, "Step 3c — Boost career_planning (+2)\nif query contains:\n"
              '"salary", "lpa", "ctc", "package",\n"offer", "placed", "placement"',
         "rect",  "#EAF3DE", "#639922", 8, False),
        (7.3, "Step 4\nSelect role with highest score\nbest = max(scores, key=scores.get)",
         "rect",  "#FAEEDA", "#854F0B", 9, False),
        (5.5, "max score > 0 ?",
         "diamond", "#FFF3CD", "#CC8800", 9, True),
        (3.8, "OUTPUT\nReturn best-fit role ID",
         "rect",  "#1A1A2E", "#FFFFFF", 10, True),
        (3.8, "OUTPUT\nDefault → academic_guidance",
         "rect",  "#FAECE7", "#D85A30", 9, False),
    ]

    # Draw all nodes except the two parallel outputs
    for i, (y, text, shape, fc, tc, fs, bold) in enumerate(nodes[:-1]):
        _draw_box(ax, cx, y, bw, 0.95 if shape == "diamond" else 1.0 if "\n\n" in text else 0.85,
                  text, shape=shape, facecolor=fc, fontcolor=tc, fontsize=fs, bold=bold,
                  edgecolor=tc if fc != "#1A1A2E" else "#AAAAAA")

    # Parallel outputs after decision
    yes_x, no_x = 3.2, 7.2
    _draw_box(ax, yes_x, 3.8, 2.8, 0.75, "OUTPUT\nReturn best-fit role ID",
              facecolor="#1A1A2E", fontcolor="#FFFFFF", edgecolor="#888888", fontsize=8.5, bold=True)
    _draw_box(ax, no_x, 3.8, 2.8, 0.75, "OUTPUT\nDefault → academic_guidance",
              facecolor="#FAECE7", fontcolor="#D85A30", edgecolor="#D85A30", fontsize=8.5, bold=False)

    # ── Arrows ──
    ys = [n[0] for n in nodes]
    for i in range(6):
        _arrow(ax, cx, ys[i] - 0.5, cx, ys[i + 1] + 0.5)

    # After step 3c → step 4
    _arrow(ax, cx, ys[6] - 0.5, cx, ys[7] + 0.55)

    # Decision → yes/no
    _arrow(ax, yes_x - 0.15, ys[7] - 0.55, yes_x, 3.8 + 0.4, label="Yes (score>0)", color="#0F6E56")
    _arrow(ax, no_x + 0.15, ys[7] - 0.35, no_x, 3.8 + 0.4, label="No (score=0)", color="#D85A30")

    ax.set_title("Role Detection Algorithm — detect_role() Flowchart",
                 fontsize=14, fontweight="bold", color="#1A1A2E", y=0.985)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure2_role_detection_flowchart.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════
# FIGURE 3 — RAG Pipeline Design
# ════════════════════════════════════════════════════════════════

def figure3_rag_pipeline():
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#F7F8FC")

    stages = [
        # (x_center, y_center, width, height, title, body, facecolor, edgecolor)
        (1.5,  4.0, 2.6, 5.5,
         "Knowledge\nSources",
         ".txt files:\n career_planning.txt\n higher_studies.txt\n job_market.txt\n etc.\n\n[Web Scraper]\n (scraper.py\n via ingest_urls.py)",
         "#EAF3DE", "#639922"),
        (4.4,  4.0, 2.4, 3.5,
         "Text\nSplitter",
         "RecursiveCharacter\nTextSplitter\n\nchunk_size = 900\nchunk_overlap = 150",
         "#EEEDFE", "#534AB7"),
        (7.1,  4.0, 2.4, 3.5,
         "Embeddings",
         "SimpleHashEmbeddings\n(deterministic 1536-d\nchar trigram, MD5)\n\n⟳ Swappable for\nOpenAI embeddings",
         "#E6F1FB", "#185FA5"),
        (9.9,  4.0, 2.4, 3.5,
         "Vector Store",
         "ChromaDB\n(local persist)\nchroma_db/\n\nTop-K similarity\nsearch",
         "#FAEEDA", "#854F0B"),
        (12.7, 4.0, 2.6, 4.0,
         "Prompt\nAssembly",
         "Role-specific\nsystem prompt\n+\nStudent profile\ncontext\n+\nRAG context (top-5)\n+\nConversation history",
         "#FAECE7", "#D85A30"),
        (15.6, 4.0, 2.4, 3.5,
         "LLM",
         "OpenRouter API\n→ Claude 3.5 Sonnet\ntemperature = 0.4\nmax_tokens = 2048",
         "#FBEAF0", "#993556"),
    ]

    prev_x2 = None
    for (x, y, w, h, title, body, fc, ec) in stages:
        # Box
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.12",
                             facecolor=fc, edgecolor=ec, lw=2.5, zorder=3)
        ax.add_patch(box)
        # Title bar
        title_h = 0.65
        tbar = FancyBboxPatch((x - w / 2, y + h / 2 - title_h), w, title_h,
                              boxstyle="round,pad=0.0",
                              facecolor=ec, edgecolor=ec, lw=0, zorder=4)
        ax.add_patch(tbar)
        ax.text(x, y + h / 2 - title_h / 2, title,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", zorder=5, multialignment="center")
        ax.text(x, y - 0.1, body,
                ha="center", va="center", fontsize=7.5, color="#333333",
                zorder=5, multialignment="center", linespacing=1.4)
        # Arrow from previous stage
        if prev_x2 is not None:
            ax.annotate("", xy=(x - w / 2, y),
                        xytext=(prev_x2, y),
                        arrowprops=dict(arrowstyle="-|>", color="#666666",
                                        lw=2, mutation_scale=18),
                        zorder=2)
        prev_x2 = x + w / 2

    # RAG_TOP_K annotation
    ax.text(9.9, 1.05, "RAG_TOP_K = 5", ha="center", fontsize=8.5,
            color="#854F0B", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FAEEDA", edgecolor="#854F0B", lw=1.2))

    # Output arrow + label
    ax.annotate("", xy=(17.5, 4.0), xytext=(16.8, 4.0),
                arrowprops=dict(arrowstyle="-|>", color="#993556", lw=2.2, mutation_scale=18))
    ax.text(17.6, 4.0, "Structured\nMarkdown\nResponse",
            ha="left", va="center", fontsize=8, fontweight="bold",
            color="#993556", linespacing=1.4)

    ax.set_title("RAG Pipeline Design — Retrieval-Augmented Generation",
                 fontsize=14, fontweight="bold", color="#1A1A2E", pad=14)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure3_rag_pipeline.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════
# FIGURE 4 — System Performance Metrics (2×2 grid)
# ════════════════════════════════════════════════════════════════

def figure4_performance_metrics():
    role_names_short = [
        "Academic\nGuidance", "Career\nPlanning", "Job\nMarket",
        "Higher\nStudies", "Academic\nProgress", "Mentorship", "Administrative"
    ]
    role_colors = list(ROLE_COLORS.values())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#F7F8FC")
    fig.suptitle("System Performance Metrics\n"
                 "(Illustrative / Representative Data)",
                 fontsize=14, fontweight="bold", color="#1A1A2E", y=0.98)

    # ── Panel A: Average Response Time by Role ──
    ax = axes[0, 0]
    times = [2.1, 2.8, 3.2, 2.5, 1.8, 1.5, 1.3]
    bars = ax.bar(range(7), times, color=role_colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(7))
    ax.set_xticklabels(role_names_short, fontsize=8)
    ax.set_ylabel("Response Time (seconds)", fontsize=9)
    ax.set_title("(A)  Average Response Time by Role", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 4.0)
    ax.yaxis.grid(True, color="#DDDDDD", zorder=0)
    ax.set_facecolor("#FAFAFA")
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Panel B: Query Distribution by Role ──
    ax = axes[0, 1]
    dist = [25, 20, 18, 15, 10, 7, 5]
    wedge_props = dict(width=0.55, edgecolor="white", linewidth=2)
    wedges, texts, autotexts = ax.pie(
        dist, labels=None, autopct="%1.0f%%",
        colors=role_colors, startangle=140,
        wedgeprops=wedge_props, pctdistance=0.75)
    for at in autotexts:
        at.set_fontsize(8)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("(B)  Query Distribution by Role", fontsize=10, fontweight="bold")
    legend_labels = [
        f"{rn.replace(chr(10), ' ')} ({d}%)"
        for rn, d in zip(role_names_short, dist)
    ]
    ax.legend(wedges, legend_labels, loc="lower left", bbox_to_anchor=(-0.35, -0.12),
              fontsize=7.5, framealpha=0.7)

    # ── Panel C: Knowledge Base Coverage ──
    ax = axes[1, 0]
    chunks = [142, 118, 135, 126, 98, 84, 77]
    bars = ax.bar(range(7), chunks, color=role_colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(7))
    ax.set_xticklabels(role_names_short, fontsize=8)
    ax.set_ylabel("Number of Chunks", fontsize=9)
    ax.set_title("(C)  Knowledge Base Coverage", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, color="#DDDDDD", zorder=0)
    ax.set_facecolor("#FAFAFA")
    for bar, val in zip(bars, chunks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(val), ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ── Panel D: Response Quality over Sessions ──
    ax = axes[1, 1]
    sessions = np.arange(1, 11)
    quality  = [3.6, 3.7, 3.85, 3.9, 4.0, 4.05, 4.12, 4.18, 4.22, 4.28]
    ax.plot(sessions, quality, marker="o", color="#534AB7", linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgewidth=2.5, zorder=3)
    ax.fill_between(sessions, quality, alpha=0.12, color="#534AB7")
    ax.set_xticks(sessions)
    ax.set_xlabel("Session Number", fontsize=9)
    ax.set_ylabel("Quality Score (1–5)", fontsize=9)
    ax.set_ylim(3.0, 5.0)
    ax.set_title("(D)  Response Quality over Sessions", fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, color="#DDDDDD", zorder=0)
    ax.set_facecolor("#FAFAFA")
    for s, q in zip(sessions, quality):
        ax.text(s, q + 0.05, f"{q:.2f}", ha="center", va="bottom", fontsize=7)

    fig.text(0.5, 0.01,
             "Note: All metrics are illustrative/representative figures for documentation purposes.",
             ha="center", fontsize=9, color="#888888", style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = os.path.join(FIGURES_DIR, "figure4_performance_metrics.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════
# FIGURE 5 — Query Routing Accuracy Comparison
# ════════════════════════════════════════════════════════════════

def figure5_routing_accuracy():
    role_names_short = [
        "Academic\nGuidance", "Career\nPlanning", "Job\nMarket",
        "Higher\nStudies", "Academic\nProgress", "Mentorship", "Administrative", "Overall"
    ]

    approaches = [
        "Keyword Only",
        "Keyword +\nWeighting",
        "Keyword + Weighting\n+ Boost Rules",
        "Full System\n(+Profile Context)",
    ]
    approach_colors = ["#AAAAAA", "#534AB7", "#185FA5", "#0F6E56"]

    # Accuracy per role per approach (illustrative)
    data = np.array([
        # KW-only  KW+W   KW+W+B  Full
        [68, 80, 88, 93],   # academic_guidance
        [65, 78, 85, 92],   # career_planning
        [72, 84, 93, 96],   # job_market
        [66, 80, 94, 97],   # higher_studies
        [70, 81, 87, 93],   # academic_progress
        [71, 82, 88, 94],   # mentorship
        [74, 86, 89, 93],   # administrative
        [70, 82, 90, 94],   # overall
    ])

    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor("#F7F8FC")
    ax.set_facecolor("#FAFAFA")

    n_roles = len(role_names_short)
    n_approaches = len(approaches)
    bar_w = 0.18
    group_w = bar_w * n_approaches
    x = np.arange(n_roles)

    for j, (approach, color) in enumerate(zip(approaches, approach_colors)):
        offset = (j - (n_approaches - 1) / 2) * bar_w
        bars = ax.bar(x + offset, data[:, j], bar_w,
                      label=approach, color=color,
                      edgecolor="white", linewidth=0.6, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(role_names_short, fontsize=9)
    ax.set_ylabel("Routing Accuracy (%)", fontsize=10)
    ax.set_ylim(55, 102)
    ax.yaxis.grid(True, color="#DDDDDD", zorder=0, linestyle="--", alpha=0.8)
    ax.set_title("Query Routing Accuracy Comparison — detect_role() Approaches",
                 fontsize=13, fontweight="bold", color="#1A1A2E", pad=14)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, ncol=2)

    # Accuracy labels on top of the "Full System" bars
    for j in range(n_approaches):
        offset = (j - (n_approaches - 1) / 2) * bar_w
        for i in range(n_roles):
            ax.text(x[i] + offset, data[i, j] + 0.5,
                    f"{data[i, j]}%", ha="center", va="bottom",
                    fontsize=6.5, color="#333333")

    fig.text(0.5, 0.01,
             "Note: All accuracy figures are illustrative/representative for documentation purposes.",
             ha="center", fontsize=9, color="#888888", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(FIGURES_DIR, "figure5_routing_accuracy.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════
# FIGURE 6 — Student Response Quality Ratings
# ════════════════════════════════════════════════════════════════

def figure6_quality_ratings():
    role_names = [
        "Academic Guidance", "Career Planning", "Job Market",
        "Higher Studies", "Academic Progress", "Mentorship", "Administrative"
    ]
    role_colors = list(ROLE_COLORS.values())
    avg_ratings = [4.5, 4.4, 4.3, 4.6, 4.2, 4.7, 4.0]

    # Distribution of 1-5 star ratings per role (rows = roles, cols = 1..5 stars)
    rating_dist = np.array([
        [1,  3,  8, 30, 58],   # academic_guidance
        [1,  4,  9, 35, 51],   # career_planning
        [2,  4, 10, 36, 48],   # job_market
        [0,  2,  7, 27, 64],   # higher_studies
        [1,  5, 12, 38, 44],   # academic_progress
        [0,  2,  5, 23, 70],   # mentorship
        [2,  6, 15, 40, 37],   # administrative
    ], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#F7F8FC")

    # ── Left: Horizontal bar chart — average ratings ──
    y_pos = np.arange(len(role_names))
    hbars = ax1.barh(y_pos, avg_ratings, color=role_colors,
                     edgecolor="white", linewidth=0.8, height=0.55, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(role_names, fontsize=10)
    ax1.set_xlabel("Average Rating (1–5 scale)", fontsize=10)
    ax1.set_xlim(0, 5.2)
    ax1.xaxis.grid(True, color="#DDDDDD", zorder=0, linestyle="--", alpha=0.8)
    ax1.set_facecolor("#FAFAFA")
    ax1.set_title("(A)  Average Student Satisfaction Rating\nby Advisor Role",
                  fontsize=11, fontweight="bold", color="#1A1A2E")
    ax1.axvline(x=4.0, color="#999999", lw=1.2, ls="--", zorder=2)
    ax1.text(4.02, 6.6, "4.0 baseline", fontsize=7.5, color="#999999")
    for i, (bar, val) in enumerate(zip(hbars, avg_ratings)):
        ax1.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2,
                 f"★ {val:.1f}", va="center", fontsize=9.5, fontweight="bold",
                 color=role_colors[i])

    # ── Right: Stacked bar chart — rating distribution ──
    star_colors = ["#D32F2F", "#F57C00", "#FBC02D", "#7CB342", "#1B5E20"]
    star_labels = ["1 ★", "2 ★", "3 ★", "4 ★", "5 ★"]
    bottoms = np.zeros(len(role_names))
    x_pos = np.arange(len(role_names))

    for star_idx in range(5):
        col = rating_dist[:, star_idx]
        bars = ax2.bar(x_pos, col, bottom=bottoms,
                       color=star_colors[star_idx], label=star_labels[star_idx],
                       edgecolor="white", linewidth=0.5, width=0.6, zorder=3)
        # Label if segment > 8%
        for i, (b, c) in enumerate(zip(bottoms, col)):
            if c >= 8:
                ax2.text(x_pos[i], b + c / 2, f"{int(c)}%",
                         ha="center", va="center", fontsize=7.5,
                         color="white", fontweight="bold")
        bottoms += col

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([n.replace(" ", "\n") for n in role_names], fontsize=8.5)
    ax2.set_ylabel("Percentage of Responses (%)", fontsize=10)
    ax2.set_ylim(0, 108)
    ax2.yaxis.grid(True, color="#DDDDDD", zorder=0, linestyle="--", alpha=0.7)
    ax2.set_facecolor("#FAFAFA")
    ax2.set_title("(B)  Rating Distribution by Advisor Role\n(1–5 star breakdown)",
                  fontsize=11, fontweight="bold", color="#1A1A2E")
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9,
               ncol=5, bbox_to_anchor=(1.0, 1.14))

    fig.text(0.5, 0.01,
             "Note: All ratings are illustrative/representative figures for documentation purposes.",
             ha="center", fontsize=9, color="#888888", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(FIGURES_DIR, "figure6_quality_ratings.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures for Academic Advisor Agent documentation…\n")

    print("Figure 1: Seven Advisor Roles Architecture")
    figure1_roles_architecture()

    print("Figure 2: Role Detection Algorithm Flowchart")
    figure2_role_detection_flowchart()

    print("Figure 3: RAG Pipeline Design")
    figure3_rag_pipeline()

    print("Figure 4: System Performance Metrics")
    figure4_performance_metrics()

    print("Figure 5: Query Routing Accuracy Comparison")
    figure5_routing_accuracy()

    print("Figure 6: Student Response Quality Ratings")
    figure6_quality_ratings()

    print(f"\n✅ All 6 figures saved to '{FIGURES_DIR}/'")
