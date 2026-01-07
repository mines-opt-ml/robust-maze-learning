"""Utilities for generating architecture diagrams."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Color scheme for diagrams
COLORS = {
    'input': '#E3F2FD',
    'projection': '#90CAF9',
    'recur': '#4CAF50',
    'residual': '#A5D6A7',
    'head': '#FFE0B2',
    'output': '#FFF3E0',
    'arrow': '#37474F',
    'concat': '#7B1FA2',
    'text': '#212121',
    'gray': '#757575',
}


def _draw_block(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str | None = None,
    color: str = '#90CAF9',
) -> None:
    """Draw a rounded rectangle block with title and optional subtitle."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color,
        edgecolor='#424242',
        linewidth=2.5,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x, y + 0.15, title, ha='center', va='center', fontsize=13, fontweight='bold', color=COLORS['text'])
        ax.text(x, y - 0.2, subtitle, ha='center', va='center', fontsize=10, color=COLORS['gray'])
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=13, fontweight='bold', color=COLORS['text'])


def _draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    """Draw a simple arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle='-|>',
        mutation_scale=18,
        color=COLORS['arrow'],
        linewidth=2,
    )
    ax.add_patch(arrow)


def generate_dt_net_diagram(output_path: Path | str) -> None:
    """Generate architecture diagram for DTNet (DeepThinking Network).

    Args:
        output_path: Path to save the PDF diagram.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Figure setup - slightly narrower
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(1, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # === TITLE ===
    ax.text(7, 5.5, 'DTNet', ha='center', va='center', fontsize=18, fontweight='bold')

    # === MAIN FLOW ===
    y = 3.5
    bw, bh = 1.4, 1.0  # block width, height

    # Input block
    input_x = 1.0
    _draw_block(ax, input_x, y, bw, bh, 'Input', 'H×W\n3 ch', COLORS['input'])

    # Arrow: Input → Projection
    _draw_arrow(ax, input_x + bw / 2, y, 2.8 - bw / 2, y)

    # Projection block
    proj_x = 2.8
    _draw_block(ax, proj_x, y, bw, bh, 'Projection', 'Conv2d, ReLU\n3→128 ch', COLORS['projection'])

    # === RECURRENT BLOCK ===
    rec_x, rec_w, rec_h = 6.8, 5.2, 2.2
    rec_box = FancyBboxPatch(
        (rec_x - rec_w / 2, y - rec_h / 2),
        rec_w,
        rec_h,
        boxstyle='round,pad=0.03,rounding_size=0.15',
        facecolor='#E8F5E9',
        edgecolor=COLORS['recur'],
        linewidth=2.5,
        linestyle='--',
    )
    ax.add_patch(rec_box)
    ax.text(
        rec_x,
        y + rec_h / 2 + 0.2,
        r'Recurrent Block  (× $K$ iterations)',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color=COLORS['recur'],
    )

    # Concat circle (input injection point)
    concat_x = 5.0
    circle = plt.Circle((concat_x, y), 0.22, facecolor=COLORS['concat'], edgecolor='#424242', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(concat_x, y, '+', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=6)

    # Arrow: Projection → Concat circle
    _draw_arrow(ax, proj_x + bw / 2, y, concat_x - 0.22, y)

    # Arrow: Concat → Conv Recall
    _draw_arrow(ax, concat_x + 0.22, y, 5.9, y)

    # Conv Recall block
    _draw_block(ax, 6.5, y, 1.1, 0.8, 'Conv', '131→128 ch', COLORS['residual'])

    # Arrow: Conv → Residual
    _draw_arrow(ax, 7.05, y, 7.45, y)

    # Residual block
    residual_x = 8.0
    _draw_block(ax, residual_x, y, 1.0, 0.8, 'Residual', '2× ResBlock', COLORS['residual'])

    # Latent feedback loop: down from bottom center of Residual, left, up to concat
    loop_y_bottom = y - rec_h / 2 + 0.15
    residual_bottom = y - 0.4 - 0.08  # Bottom edge of Residual block plus offset for border
    # Down from bottom center of Residual
    ax.plot(
        [residual_x, residual_x], [residual_bottom, loop_y_bottom], color=COLORS['recur'], lw=2, solid_capstyle='round'
    )
    # Horizontal left to just right of concat
    concat_arrow_x_green = concat_x + 0.1  # Slightly right of center
    ax.plot(
        [residual_x, concat_arrow_x_green],
        [loop_y_bottom, loop_y_bottom],
        color=COLORS['recur'],
        lw=2,
        solid_capstyle='round',
    )
    # Up to concat from below (right side)
    ax.annotate(
        '',
        xy=(concat_arrow_x_green, y - 0.22),
        xytext=(concat_arrow_x_green, loop_y_bottom),
        arrowprops=dict(arrowstyle='-|>', color=COLORS['recur'], lw=2),
    )
    ax.text(
        6.8, loop_y_bottom + 0.18, 'Latent', ha='center', va='center', fontsize=9, style='italic', color=COLORS['recur']
    )

    # === HEAD ===
    head_x = 10.6
    _draw_block(ax, head_x, y, bw, bh, 'Head', '3× Conv2d\n128→32→8→2 ch', COLORS['head'])

    # Arrow: Residual → Head
    _draw_arrow(ax, residual_x + 0.5, y, head_x - bw / 2, y)

    # === OUTPUT ===
    output_x = 12.4
    _draw_block(ax, output_x, y, bw, bh, 'Prediction', 'Argmax\nH×W, 2 cls', COLORS['output'])

    # Arrow: Head → Output
    _draw_arrow(ax, head_x + bw / 2, y, output_x - bw / 2, y)

    # === INPUT INJECTION + MASK PATH (purple, below main flow) ===
    # This single path shows input going to both: recurrent concat AND output mask
    inject_y = y - 1.8
    concat_arrow_x_purple = concat_x - 0.1  # Slightly left of center (symmetric with green)
    input_bottom = y - bh / 2 - 0.08  # Bottom edge of Input block plus offset for border
    output_bottom = y - bh / 2 - 0.08  # Bottom edge of Output block plus offset for border
    # Down from Input (starting at bottom edge)
    ax.plot([input_x, input_x], [input_bottom, inject_y], color=COLORS['concat'], lw=2.5, solid_capstyle='round')
    # Horizontal across to output
    ax.plot([input_x, output_x], [inject_y, inject_y], color=COLORS['concat'], lw=2.5, solid_capstyle='round')
    # Up to concat from below (left side)
    ax.annotate(
        '',
        xy=(concat_arrow_x_purple, y - 0.22),
        xytext=(concat_arrow_x_purple, inject_y),
        arrowprops=dict(arrowstyle='-|>', color=COLORS['concat'], lw=2.5),
    )
    # Up to output (for masking, ending at bottom edge)
    ax.annotate(
        '',
        xy=(output_x, output_bottom),
        xytext=(output_x, inject_y),
        arrowprops=dict(arrowstyle='-|>', color=COLORS['concat'], lw=2.5),
    )
    # Labels
    ax.text(
        3.0,
        inject_y - 0.35,
        'Input Injection (every iteration)',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=COLORS['concat'], edgecolor='none'),
    )
    ax.text(
        output_x - 4.6,
        inject_y - 0.35,
        'Mask (zero out walls)',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=COLORS['concat'], edgecolor='none'),
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {output_path}')


def generate_it_net_diagram(output_path: Path | str) -> None:
    """Generate architecture diagram for ITNet (Implicit Thinking Network).

    Args:
        output_path: Path to save the PDF diagram.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Figure setup
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(1, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # === TITLE ===
    ax.text(7, 5.5, 'ITNet', ha='center', va='center', fontsize=18, fontweight='bold')

    # === MAIN FLOW ===
    y = 3.5
    bw, bh = 1.4, 1.0  # block width, height

    # Input block
    input_x = 1.0
    _draw_block(ax, input_x, y, bw, bh, 'Input', 'H×W\n3 ch', COLORS['input'])

    # Arrow: Input → Projection
    _draw_arrow(ax, input_x + bw / 2, y, 2.8 - bw / 2, y)

    # Projection block
    proj_x = 2.8
    _draw_block(ax, proj_x, y, bw, bh, 'Projection', 'Conv2d, ReLU\n3→128 ch', COLORS['projection'])

    # === RECURRENT BLOCK ===
    rec_x, rec_w, rec_h = 6.8, 5.2, 2.2
    rec_box = FancyBboxPatch(
        (rec_x - rec_w / 2, y - rec_h / 2),
        rec_w,
        rec_h,
        boxstyle='round,pad=0.03,rounding_size=0.15',
        facecolor='#E8F5E9',
        edgecolor=COLORS['recur'],
        linewidth=2.5,
        linestyle='--',
    )
    ax.add_patch(rec_box)
    ax.text(
        rec_x,
        y + rec_h / 2 + 0.2,
        r'Recurrent Block  (× $K$ iters, until $\|z^{(k)} - z^{(k-1)}\| < \epsilon$)',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color=COLORS['recur'],
    )

    # Concat circle (input injection point)
    concat_x = 5.0
    circle = plt.Circle((concat_x, y), 0.22, facecolor=COLORS['concat'], edgecolor='#424242', linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    ax.text(concat_x, y, '+', ha='center', va='center', fontsize=14, fontweight='bold', color='white', zorder=6)

    # Arrow: Projection → Concat circle
    _draw_arrow(ax, proj_x + bw / 2, y, concat_x - 0.22, y)

    # Arrow: Concat → Conv Recall
    _draw_arrow(ax, concat_x + 0.22, y, 5.9, y)

    # Conv Recall block
    _draw_block(ax, 6.5, y, 1.1, 0.8, 'Conv', '131→128 ch', COLORS['residual'])

    # Arrow: Conv → Residual
    _draw_arrow(ax, 7.05, y, 7.45, y)

    # Residual block
    residual_x = 8.0
    _draw_block(ax, residual_x, y, 1.0, 0.8, 'Residual', '4× ResBlock', COLORS['residual'])

    # Latent feedback loop: down from bottom center of Residual, left, up to concat
    loop_y_bottom = y - rec_h / 2 + 0.15
    residual_bottom = y - 0.4 - 0.08  # Bottom edge of Residual block plus offset for border
    # Down from bottom center of Residual
    ax.plot(
        [residual_x, residual_x], [residual_bottom, loop_y_bottom], color=COLORS['recur'], lw=2, solid_capstyle='round'
    )
    # Horizontal left to just right of concat
    concat_arrow_x_green = concat_x + 0.1  # Slightly right of center
    ax.plot(
        [residual_x, concat_arrow_x_green],
        [loop_y_bottom, loop_y_bottom],
        color=COLORS['recur'],
        lw=2,
        solid_capstyle='round',
    )
    # Up to concat from below (right side)
    ax.annotate(
        '',
        xy=(concat_arrow_x_green, y - 0.22),
        xytext=(concat_arrow_x_green, loop_y_bottom),
        arrowprops=dict(arrowstyle='-|>', color=COLORS['recur'], lw=2),
    )
    ax.text(
        6.8, loop_y_bottom + 0.18, 'Latent', ha='center', va='center', fontsize=9, style='italic', color=COLORS['recur']
    )

    # === HEAD ===
    head_x = 10.6
    _draw_block(ax, head_x, y, bw, bh, 'Head', '3× Conv2d\n128→32→8→2 ch', COLORS['head'])

    # Arrow: Residual → Head
    _draw_arrow(ax, residual_x + 0.5, y, head_x - bw / 2, y)

    # === OUTPUT ===
    output_x = 12.4
    _draw_block(ax, output_x, y, bw, bh, 'Prediction', 'Argmax\nH×W, 2 cls', COLORS['output'])

    # Arrow: Head → Output
    _draw_arrow(ax, head_x + bw / 2, y, output_x - bw / 2, y)

    # === INPUT INJECTION + MASK PATH (purple, below main flow) ===
    inject_y = y - 1.8
    concat_arrow_x_purple = concat_x - 0.1  # Slightly left of center (symmetric with green)
    input_bottom = y - bh / 2 - 0.08  # Bottom edge of Input block plus offset for border
    output_bottom = y - bh / 2 - 0.08  # Bottom edge of Output block plus offset for border
    # Down from Input (starting at bottom edge)
    ax.plot([input_x, input_x], [input_bottom, inject_y], color=COLORS['concat'], lw=2.5, solid_capstyle='round')
    # Horizontal across to output
    ax.plot([input_x, output_x], [inject_y, inject_y], color=COLORS['concat'], lw=2.5, solid_capstyle='round')
    # Up to concat from below (left side)
    ax.annotate(
        '',
        xy=(concat_arrow_x_purple, y - 0.22),
        xytext=(concat_arrow_x_purple, inject_y),
        arrowprops=dict(arrowstyle='-|>', color=COLORS['concat'], lw=2.5),
    )
    # Up to output (for masking, ending at bottom edge)
    ax.annotate(
        '',
        xy=(output_x, output_bottom),
        xytext=(output_x, inject_y),
        arrowprops=dict(arrowstyle='-|>', color=COLORS['concat'], lw=2.5),
    )
    # Labels
    ax.text(
        3.0,
        inject_y - 0.35,
        'Input Injection (every iteration)',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=COLORS['concat'], edgecolor='none'),
    )
    ax.text(
        output_x - 4.6,
        inject_y - 0.35,
        'Mask (zero out walls)',
        ha='center',
        va='center',
        fontsize=10,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=COLORS['concat'], edgecolor='none'),
    )

    # Save
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    # Default output location
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'outputs' / 'diagrams'
    generate_dt_net_diagram(output_dir / 'dt_net_architecture.pdf')
    generate_it_net_diagram(output_dir / 'it_net_architecture.pdf')
