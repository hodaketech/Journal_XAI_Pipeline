from __future__ import annotations

import argparse
import html

from common import OUT_DIR, ensure_dirs, read_json, require_file


def labels(summary: dict[str, object]) -> dict[str, str]:
    exclusions = dict(summary["exclusions"])
    return {
        "identification": f"Identification\nRecords identified: {summary['n_identified']}",
        "dedupe": f"De-duplication\nDuplicates removed: {summary['n_duplicates']}\nRecords screened: {summary['n_after_dedup']}",
        "screening": f"Screening\nTitle/abstract pass: {summary['n_title_abstract_pass']}",
        "eligibility": "Full-text exclusions\n" + "\n".join(f"{code}: {exclusions.get(code, 0)}" for code in ["E1", "E2", "E3", "E4"]),
        "included": f"Included\nStudies retained: {summary['n_fulltext_pass']}",
    }


def write_svg(path, summary: dict[str, object]) -> None:
    items = labels(summary)
    boxes = {
        "identification": (60, 40),
        "dedupe": (60, 180),
        "screening": (60, 320),
        "included": (60, 470),
        "eligibility": (410, 320),
    }
    width, height = 760, 620
    box_w, box_h = 250, 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"8\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L9,3 z\" fill=\"#333\"/></marker></defs>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for key, (x, y) in boxes.items():
        parts.append(f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="6" fill="#f7f9fb" stroke="#334155" stroke-width="1.5"/>')
        for idx, line in enumerate(items[key].splitlines()):
            weight = "700" if idx == 0 else "400"
            parts.append(f'<text x="{x + 18}" y="{y + 28 + idx * 18}" font-family="Arial, sans-serif" font-size="14" font-weight="{weight}" fill="#111827">{html.escape(line)}</text>')
    arrows = [
        ((185, 132), (185, 180)),
        ((185, 272), (185, 320)),
        ((185, 412), (185, 470)),
        ((310, 366), (410, 366)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="1.5" marker-end="url(#arrow)"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_png(path, summary: dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        items = labels(summary)
        boxes = {
            "identification": (0.12, 0.78),
            "dedupe": (0.12, 0.56),
            "screening": (0.12, 0.34),
            "included": (0.12, 0.10),
            "eligibility": (0.58, 0.34),
        }
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        ax.set_axis_off()
        for key, (x, y) in boxes.items():
            patch = FancyBboxPatch((x, y), 0.32, 0.14, boxstyle="round,pad=0.015,rounding_size=0.012", facecolor="#f7f9fb", edgecolor="#334155", linewidth=1.2)
            ax.add_patch(patch)
            for idx, line in enumerate(items[key].splitlines()):
                ax.text(x + 0.02, y + 0.105 - idx * 0.032, line, fontsize=10.5, weight="bold" if idx == 0 else "normal", va="center", ha="left", color="#111827")
        arrows = [
            ((0.28, 0.78), (0.28, 0.70)),
            ((0.28, 0.56), (0.28, 0.48)),
            ((0.28, 0.34), (0.28, 0.24)),
            ((0.44, 0.41), (0.58, 0.41)),
        ]
        for start, end in arrows:
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#333"))
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return
    except ModuleNotFoundError:
        write_png_with_pillow(path, summary)


def write_png_with_pillow(path, summary: dict[str, object]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    scale = 3
    width, height = 760 * scale, 620 * scale
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        regular = ImageFont.truetype("arial.ttf", 14 * scale)
        bold = ImageFont.truetype("arialbd.ttf", 14 * scale)
    except Exception:
        regular = ImageFont.load_default()
        bold = regular

    items = labels(summary)
    boxes = {
        "identification": (60 * scale, 40 * scale),
        "dedupe": (60 * scale, 180 * scale),
        "screening": (60 * scale, 320 * scale),
        "included": (60 * scale, 470 * scale),
        "eligibility": (410 * scale, 320 * scale),
    }
    box_w, box_h = 250 * scale, 92 * scale
    for key, (x, y) in boxes.items():
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=6 * scale, fill="#f7f9fb", outline="#334155", width=2 * scale)
        for idx, line in enumerate(items[key].splitlines()):
            draw.text((x + 18 * scale, y + (14 + idx * 18) * scale), line, fill="#111827", font=bold if idx == 0 else regular)
    arrows = [
        ((185, 132), (185, 180)),
        ((185, 272), (185, 320)),
        ((185, 412), (185, 470)),
        ((310, 366), (410, 366)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        start = (x1 * scale, y1 * scale)
        end = (x2 * scale, y2 * scale)
        draw.line([start, end], fill="#333333", width=2 * scale)
        if y2 > y1:
            points = [(end[0], end[1]), (end[0] - 5 * scale, end[1] - 10 * scale), (end[0] + 5 * scale, end[1] - 10 * scale)]
        else:
            points = [(end[0], end[1]), (end[0] - 10 * scale, end[1] - 5 * scale), (end[0] - 10 * scale, end[1] + 5 * scale)]
        draw.polygon(points, fill="#333333")
    image.save(path, dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render PRISMA-style Figure 1 from screening_summary.json.")
    parser.parse_args()
    ensure_dirs()
    require_file(OUT_DIR / "screening_summary.json", "Run survey/scripts/02_screen_report.py first.")
    summary = read_json(OUT_DIR / "screening_summary.json")
    write_svg(OUT_DIR / "figure1_prisma.svg", summary)
    try:
        write_png(OUT_DIR / "figure1_prisma.png", summary)
    except Exception as exc:
        raise SystemExit(f"Could not write PNG. Error: {exc}") from exc
    print(f"Wrote {OUT_DIR / 'figure1_prisma.svg'}")
    print(f"Wrote {OUT_DIR / 'figure1_prisma.png'}")


if __name__ == "__main__":
    main()
