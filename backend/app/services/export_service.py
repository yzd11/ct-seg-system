"""生成中文 PDF 诊断报告（纯 ReportLab，无需 matplotlib）。"""
import json
import math
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import List, Optional

from reportlab.graphics.shapes import Drawing, PolyLine, Line, Rect, String

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import settings

# ── 中文字体注册 ──────────────────────────────────────────────────────────────
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
]

_CN_FONT = "Helvetica"
_CN_BOLD = "Helvetica-Bold"

for _path in _FONT_CANDIDATES:
    if not Path(_path).exists():
        continue
    try:
        pdfmetrics.registerFont(TTFont("CnRegular", _path))
        pdfmetrics.registerFont(TTFont("CnBold",    _path))
        _CN_FONT = "CnRegular"
        _CN_BOLD = "CnBold"
        break
    except Exception:
        continue

# ── 颜色常量 ─────────────────────────────────────────────────────────────────
BLUE_DARK   = colors.HexColor("#1677ff")
BLUE_LIGHT  = colors.HexColor("#e8f0fe")
GREEN_DARK  = colors.HexColor("#16a34a")
GREEN_LIGHT = colors.HexColor("#f0fdf4")
TEAL_DARK   = colors.HexColor("#0d9488")
TEAL_LIGHT  = colors.HexColor("#f0fdfa")
RED_DARK    = colors.HexColor("#dc2626")
RED_LIGHT   = colors.HexColor("#fff5f5")
AMBER_DARK  = colors.HexColor("#d97706")
AMBER_LIGHT = colors.HexColor("#fffbeb")
GRAY_BG     = colors.HexColor("#f8fafc")
GRAY_BORDER = colors.HexColor("#e2e8f0")
TEXT_DARK   = colors.HexColor("#1a202c")
TEXT_GRAY   = colors.HexColor("#64748b")
WHITE       = colors.white


# ── 样式工厂 ─────────────────────────────────────────────────────────────────
def _sty(name, size=10, bold=False, color=None, align=0, leading=None,
         space_before=0, space_after=0):
    return ParagraphStyle(
        name,
        fontName=_CN_BOLD if bold else _CN_FONT,
        fontSize=size,
        leading=leading or size * 1.45,
        textColor=color or TEXT_DARK,
        alignment=align,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )


# ── 页眉 / 页脚 ───────────────────────────────────────────────────────────────
def _make_cb(case, job):
    ts = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    def cb(canvas, doc):
        w, h = A4
        canvas.saveState()
        # 顶部蓝条
        canvas.setFillColor(BLUE_DARK)
        canvas.rect(0, h - 1.2 * cm, w, 1.2 * cm, fill=1, stroke=0)
        canvas.setFillColor(WHITE)
        canvas.setFont(_CN_BOLD, 10)
        canvas.drawString(1.5 * cm, h - 0.85 * cm, "CT 肝脏与肿瘤分割系统")
        canvas.setFont(_CN_FONT, 9)
        canvas.drawRightString(w - 1.5 * cm, h - 0.85 * cm, "AI 辅助诊断报告")
        # 底部
        canvas.setStrokeColor(GRAY_BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(1.5 * cm, 1.5 * cm, w - 1.5 * cm, 1.5 * cm)
        canvas.setFillColor(TEXT_GRAY)
        canvas.setFont(_CN_FONT, 8)
        canvas.drawString(1.5 * cm, 0.9 * cm,
                          f"患者：{case.patient_id}　模型：{job.model_name}　生成：{ts}")
        canvas.drawRightString(w - 1.5 * cm, 0.9 * cm, f"第 {doc.page} 页")
        canvas.restoreState()

    return cb


# ── 临床指标计算 ──────────────────────────────────────────────────────────────
def _compute_clinical(job, case, slice_results):
    """返回所有临床量化指标。spacing = [sx, sy, sz] in mm。"""
    spacing = json.loads(case.voxel_spacing) if case.voxel_spacing else [1.0, 1.0, 1.0]
    sx, sy, sz = spacing                        # mm per voxel

    liver_vol  = job.liver_volume_ml or 0.0
    tumor_vol  = job.tumor_volume_ml or 0.0
    flr        = (liver_vol - tumor_vol) / liver_vol * 100 if liver_vol else 0

    # 含肿瘤切片
    tumor_slices = [(r.slice_index, r.tumor_area_px)
                    for r in slice_results if r.tumor_area_px > 0]
    n_tumor = len(tumor_slices)
    n_total = len(slice_results)

    if tumor_slices:
        first_z  = min(s for s, _ in tumor_slices)
        last_z   = max(s for s, _ in tumor_slices)
        axial_mm = (last_z - first_z + 1) * sz       # 轴向跨度 mm
        peak_idx, peak_area = max(tumor_slices, key=lambda x: x[1])
        # RECIST 最大径估算：假设截面近似圆形
        recist_mm = 2 * math.sqrt(peak_area * sx * sy / math.pi)
        mid_z    = (first_z + last_z) // 2
    else:
        first_z = last_z = peak_idx = mid_z = None
        axial_mm = recist_mm = peak_area = 0

    tumor_burden = tumor_vol / liver_vol * 100 if liver_vol else 0

    return {
        "liver_vol":     liver_vol,
        "tumor_vol":     tumor_vol,
        "flr":           flr,
        "tumor_burden":  tumor_burden,
        "n_total":       n_total,
        "n_tumor":       n_tumor,
        "recist_mm":     recist_mm,
        "axial_mm":      axial_mm,
        "peak_idx":      peak_idx,
        "peak_area":     peak_area,
        "first_z":       first_z,
        "last_z":        last_z,
        "mid_z":         mid_z,
        "spacing":       spacing,
        "tumor_slices":  tumor_slices,
    }


# ── 临床摘要文本生成 ──────────────────────────────────────────────────────────
def _clinical_summary(c):
    """生成一段临床语言描述（不做诊断，只描述 AI 测量结果）。"""
    if not c["n_tumor"]:
        return "AI 分割模型未在本次扫描中检出肿瘤区域，肝脏分割结果见下方数据。"

    resectable = "估算剩余肝体积（FLR）{:.1f}%，{}手术切除参考阈值（25%）。".format(
        c["flr"],
        "高于" if c["flr"] >= 25 else "低于",
    )

    return (
        f"AI 分割模型于本次 CT 扫描中检出肝脏病灶。"
        f"病灶最大轴向截面位于第 {c['peak_idx']} 切片，"
        f"估算最大径约 {c['recist_mm']:.1f} mm（RECIST 参考值，圆形截面假设）；"
        f"病灶轴向跨度约 {c['axial_mm']:.1f} mm，"
        f"累及 {c['n_tumor']} 个切片（占总切片数 {c['n_tumor']/c['n_total']*100:.1f}%）。"
        f"肿瘤体积 {c['tumor_vol']:.1f} mL，占肝脏总体积 {c['tumor_burden']:.2f}%。"
        f"{resectable}"
        f"以上数据均为 AI 辅助计算结果，仅供参考，请结合临床判断。"
    )


# ── ReportLab 原生折线图（无需 matplotlib） ───────────────────────────────────
def _make_area_chart(slice_results, c, width_cm=15.8, height_cm=5.5) -> Drawing:
    """
    用 ReportLab Drawing 绘制肝脏/肿瘤截面面积折线图。
    返回 Drawing 对象，可直接嵌入 story。
    """
    W   = width_cm  * cm
    H   = height_cm * cm
    PAD_L, PAD_R, PAD_T, PAD_B = 1.2 * cm, 0.5 * cm, 0.4 * cm, 0.9 * cm
    plot_w = W - PAD_L - PAD_R
    plot_h = H - PAD_T - PAD_B

    indices     = [r.slice_index for r in slice_results]
    liver_areas = [r.liver_area_px for r in slice_results]
    tumor_areas = [r.tumor_area_px for r in slice_results]

    if not indices:
        return Drawing(W, H)

    x_min, x_max = indices[0], indices[-1]
    y_max = max(max(liver_areas, default=1), 1)

    def px(i):   # slice index → drawing x
        return PAD_L + (i - x_min) / max(x_max - x_min, 1) * plot_w

    def py(v):   # area value → drawing y
        return PAD_B + v / y_max * plot_h

    d = Drawing(W, H)

    # ── 背景
    d.add(Rect(0, 0, W, H, fillColor=colors.HexColor("#f8fafc"),
               strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=0.5))

    # ── 水平网格线（4条）
    for i in range(1, 5):
        yv = i / 4 * y_max
        yp = py(yv)
        d.add(Line(PAD_L, yp, W - PAD_R, yp,
                   strokeColor=colors.HexColor("#e2e8f0"), strokeWidth=0.4))
        label = f"{int(yv/1000)}k" if yv >= 1000 else str(int(yv))
        d.add(String(PAD_L - 4, yp - 3, label,
                     fontName=_CN_FONT, fontSize=6, fillColor=colors.HexColor("#94a3b8"),
                     textAnchor="end"))

    # ── 下采样（最多 300 点，避免折线点过密）
    step = max(1, len(indices) // 300)
    si   = indices[::step]
    sl   = liver_areas[::step]
    st   = tumor_areas[::step]

    # 肝脏折线（青绿）
    liver_pts = []
    for i, v in zip(si, sl):
        liver_pts += [px(i), py(v)]
    if len(liver_pts) >= 4:
        d.add(PolyLine(liver_pts, strokeColor=colors.HexColor("#0d9488"),
                       strokeWidth=1.0, fillColor=None))

    # 肿瘤折线（红）
    tumor_pts = []
    for i, v in zip(si, st):
        tumor_pts += [px(i), py(v)]
    if len(tumor_pts) >= 4:
        d.add(PolyLine(tumor_pts, strokeColor=colors.HexColor("#dc2626"),
                       strokeWidth=1.4, fillColor=None))

    # ── 关键切片竖线标注（错落高度，防止文字重叠）
    marker_spec = []
    if c["peak_idx"] is not None:
        marker_spec.append((c["peak_idx"], "#dc2626", "峰值"))
    if c["first_z"] is not None and c["first_z"] != c["peak_idx"]:
        marker_spec.append((c["first_z"], "#f59e0b", "起始"))
    if c["last_z"] is not None and c["last_z"] != c["peak_idx"]:
        marker_spec.append((c["last_z"], "#f59e0b", "终止"))

    for i, (z_idx, hex_color, label) in enumerate(marker_spec):
        xp = px(z_idx)
        cl = colors.HexColor(hex_color)
        d.add(Line(xp, PAD_B, xp, H - PAD_T,
                   strokeColor=cl, strokeWidth=0.7,
                   strokeDashArray=[3, 2]))
        y_offset = PAD_T + 6 + i * 10   # 每条标注错落 10pt
        d.add(String(xp + 2, H - y_offset, f"{label}({z_idx})",
                     fontName=_CN_FONT, fontSize=6, fillColor=cl))

    # ── 坐标轴
    d.add(Line(PAD_L, PAD_B, W - PAD_R, PAD_B,
               strokeColor=colors.HexColor("#94a3b8"), strokeWidth=0.6))
    d.add(Line(PAD_L, PAD_B, PAD_L, H - PAD_T,
               strokeColor=colors.HexColor("#94a3b8"), strokeWidth=0.6))

    # ── X 轴刻度标签（5个）
    for i in range(5):
        zi = x_min + int(i / 4 * (x_max - x_min))
        xp = px(zi)
        d.add(Line(xp, PAD_B, xp, PAD_B - 3,
                   strokeColor=colors.HexColor("#94a3b8"), strokeWidth=0.5))
        d.add(String(xp, PAD_B - 10, str(zi),
                     fontName=_CN_FONT, fontSize=6,
                     fillColor=colors.HexColor("#94a3b8"), textAnchor="middle"))

    # ── 轴标签
    d.add(String(W / 2, 2, "切片索引",
                 fontName=_CN_FONT, fontSize=7,
                 fillColor=colors.HexColor("#64748b"), textAnchor="middle"))

    return d


# ── 代表性切片图选取（仅含肿瘤） ─────────────────────────────────────────────
def _pick_representative_images(masks_dir: Path, c) -> list:
    """
    返回最多 3 张含肿瘤的代表性切片路径：
    [最大截面, 肿瘤起始, 肿瘤终止]
    如果起始/终止与最大截面相同则去重。
    """
    if not masks_dir.exists() or not c["peak_idx"] is not None:
        return []

    def _path(z):
        p = masks_dir / f"{z:04d}.png"
        return p if p.exists() else None

    candidates = {}
    if c["peak_idx"] is not None:
        p = _path(c["peak_idx"])
        if p:
            candidates["最大截面"] = (c["peak_idx"], p)
    if c["first_z"] is not None and c["first_z"] != c["peak_idx"]:
        p = _path(c["first_z"])
        if p:
            candidates["肿瘤起始"] = (c["first_z"], p)
    if c["last_z"] is not None and c["last_z"] != c["peak_idx"] and c["last_z"] != c["first_z"]:
        p = _path(c["last_z"])
        if p:
            candidates["肿瘤终止"] = (c["last_z"], p)

    return [(label, z_idx, path) for label, (z_idx, path) in candidates.items()]


# ── 主函数 ────────────────────────────────────────────────────────────────────
async def generate_pdf(job, case, slice_results: Optional[List] = None) -> bytes:
    buf     = BytesIO()
    W       = A4[0] - 3.6 * cm   # 可用宽度

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
    )

    sr   = slice_results or []
    c    = _compute_clinical(job, case, sr)
    story = []

    # ── 标题 ──────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        "CT 肝脏与肿瘤 AI 分割诊断报告",
        _sty("title", size=19, bold=True, align=1, color=BLUE_DARK),
    ))
    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph(
        f"报告编号：{job.id[:8].upper()}　　生成日期：{datetime.now().strftime('%Y年%m月%d日')}",
        _sty("sub", size=9, align=1, color=TEXT_GRAY),
    ))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE_DARK, spaceAfter=10))

    # ── 基本信息 ──────────────────────────────────────────────────────────────
    story.append(Paragraph("■ 基本信息", _sty("s1", size=12, bold=True, color=BLUE_DARK,
                                              space_before=2, space_after=6)))
    finished  = job.finished_at.strftime("%Y年%m月%d日 %H:%M") if job.finished_at else "—"
    sp        = c["spacing"]
    sp_str    = f"{sp[0]:.2f} × {sp[1]:.2f} × {sp[2]:.2f} mm"

    def _cell(text, lbl=False, clr=None):
        actual = clr if clr is not None else (TEXT_GRAY if lbl else TEXT_DARK)
        st = _sty(f"_c{id(text)}", size=9, bold=lbl, color=actual)
        return Paragraph(text, st)

    cw4 = [3.0 * cm, 5.8 * cm, 3.2 * cm, 5.8 * cm]
    meta = Table([
        [_cell("项目", True, WHITE), _cell("内容", True, WHITE),
         _cell("项目", True, WHITE), _cell("内容", True, WHITE)],
        [_cell("患者编号", True), _cell(case.patient_id),
         _cell("分析模型", True), _cell(job.model_name)],
        [_cell("文件名称", True), _cell(case.filename),
         _cell("分析完成", True), _cell(finished)],
        [_cell("切片数量", True), _cell(f"{case.slice_count or '—'} 张"),
         _cell("体素间距", True), _cell(sp_str)],
    ], colWidths=cw4, repeatRows=1)
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BLUE_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), _CN_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BACKGROUND",    (0, 1), (0, -1), BLUE_LIGHT),
        ("BACKGROUND",    (2, 1), (2, -1), BLUE_LIGHT),
        ("FONTNAME",      (0, 1), (0, -1), _CN_BOLD),
        ("FONTNAME",      (2, 1), (2, -1), _CN_BOLD),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, GRAY_BORDER),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_BG]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(meta)
    story.append(Spacer(1, 0.5 * cm))

    # ── AI 摘要 ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_BORDER, spaceAfter=8))
    story.append(Paragraph("■ AI 分割结果摘要", _sty("s2", size=12, bold=True, color=BLUE_DARK,
                                                     space_before=4, space_after=8)))
    summary_box = Table(
        [[Paragraph(_clinical_summary(c), _sty("smry", size=9, leading=14, color=TEXT_DARK))]],
        colWidths=[W],
    )
    summary_box.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE_LIGHT),
        ("BOX",           (0, 0), (-1, -1), 1.0, BLUE_DARK),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_box)
    story.append(Spacer(1, 0.5 * cm))

    # ── 关键量化指标 ──────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_BORDER, spaceAfter=8))
    story.append(Paragraph("■ 关键量化指标", _sty("s3", size=12, bold=True, color=TEAL_DARK,
                                                  space_before=4, space_after=8)))

    def _fmt(v, fmt=".1f", suffix="", none_val="—"):
        if v is None:
            return none_val
        return f"{v:{fmt}}{suffix}"

    # FLR 风险等级
    if c["flr"] >= 30:
        flr_note = "✓ 良好（≥30%）"
        flr_color = GREEN_DARK
    elif c["flr"] >= 25:
        flr_note = "△ 临界（25–30%）"
        flr_color = AMBER_DARK
    else:
        flr_note = "✗ 偏低（<25%）"
        flr_color = RED_DARK

    def _kv(key, val, note="", highlight=None):
        """生成一行 [指标, 数值, 说明/风险] 的三列行。"""
        val_color = highlight or TEXT_DARK
        return [
            Paragraph(key,  _sty(f"k{id(key)}", size=9, bold=True, color=TEXT_GRAY)),
            Paragraph(val,  _sty(f"v{id(val)}", size=9, bold=True, color=val_color)),
            Paragraph(note, _sty(f"n{id(note)}", size=8, color=TEXT_GRAY)),
        ]

    kpi_rows = [
        [Paragraph("指标",   _sty("kh1", size=9, bold=True, color=WHITE)),
         Paragraph("数值",   _sty("kh2", size=9, bold=True, color=WHITE)),
         Paragraph("临床参考", _sty("kh3", size=9, bold=True, color=WHITE))],
        _kv("肝脏总体积",    _fmt(c["liver_vol"], ".1f", " mL"), ""),
        _kv("肿瘤总体积",    _fmt(c["tumor_vol"], ".1f", " mL"), ""),
        _kv("肿瘤负荷",      _fmt(c["tumor_burden"], ".2f", "%"),
            "< 5% 轻度 | 5–25% 中度 | > 25% 重度"),
        _kv("剩余肝体积（FLR）", _fmt(c["flr"], ".1f", "%"),
            flr_note, highlight=flr_color),
        _kv("最大径估算（RECIST）",
            _fmt(c["recist_mm"], ".1f", " mm") if c["recist_mm"] else "—",
            "基于最大截面面积，圆形假设；供参考"),
        _kv("病灶轴向跨度",
            _fmt(c["axial_mm"], ".1f", " mm") if c["axial_mm"] else "—",
            f"第 {c['first_z']}–{c['last_z']} 切片"
            if c["first_z"] is not None else ""),
        _kv("含肿瘤切片数",
            f"{c['n_tumor']} / {c['n_total']} 张",
            f"占总切片 {c['n_tumor']/c['n_total']*100:.1f}%"
            if c["n_total"] else ""),
        _kv("最大截面切片",
            f"第 {c['peak_idx']} 张" if c["peak_idx"] is not None else "—",
            f"截面积 {c['peak_area']:,} px²" if c["peak_area"] else ""),
    ]

    kpi_t = Table(kpi_rows, colWidths=[5.0 * cm, 4.0 * cm, W - 9.0 * cm], repeatRows=1)
    kpi_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TEAL_DARK),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), _CN_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("BACKGROUND",    (0, 1), (0, -1), TEAL_LIGHT),
        ("FONTNAME",      (0, 1), (0, -1), _CN_BOLD),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("GRID",          (0, 0), (-1, -1), 0.5, GRAY_BORDER),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_BG]),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_t)
    story.append(Spacer(1, 0.5 * cm))

    # ── 肿瘤截面面积分布图（从第二页开始）────────────────────────────────────
    if sr:
        chart_drawing = _make_area_chart(sr, c, width_cm=15.8, height_cm=5.5)
        story.append(PageBreak())
        story.append(Paragraph("■ 肿瘤截面面积分布",
                                _sty("s4", size=12, bold=True, color=BLUE_DARK,
                                     space_before=0, space_after=6)))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_BORDER, spaceAfter=8))
        story.append(Paragraph(
            "下图展示各切片的肝脏与肿瘤截面面积变化。"
            "纵轴为像素面积，横轴为切片索引。垂直虚线标注关键切片位置（峰值/起始/终止）。",
            _sty("chart_desc", size=8, color=TEXT_GRAY, space_after=6),
        ))
        story.append(chart_drawing)
        # 图例单独置于图表正下方，不与图像重叠
        legend_para = Paragraph(
            '<font color="#0d9488"><b>━━</b></font>&nbsp;&nbsp;肝脏面积'
            '&nbsp;&nbsp;&nbsp;&nbsp;'
            '<font color="#dc2626"><b>━━</b></font>&nbsp;&nbsp;肿瘤面积',
            _sty("chart_legend", size=8, align=1, color=TEXT_GRAY, space_after=4),
        )
        story.append(legend_para)
        story.append(Spacer(1, 0.3 * cm))

    # ── 代表性切片图 ──────────────────────────────────────────────────────────
    masks_dir = settings.upload_dir / job.case_id / "results" / job.id / "masks"
    picks     = _pick_representative_images(masks_dir, c)

    if picks:
        story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_BORDER, spaceAfter=8))
        story.append(Paragraph("■ 代表性切片分割图",
                                _sty("s5", size=12, bold=True, color=BLUE_DARK,
                                     space_before=4, space_after=6)))
        story.append(Paragraph(
            "以下切片均含有肿瘤分割结果，按临床意义选取：最大截面（病灶最大横径所在层）、"
            "肿瘤起始层与终止层（反映病灶纵向范围）。绿色为肝脏分割，红色为肿瘤分割。",
            _sty("img_desc", size=8, color=TEXT_GRAY, space_after=8),
        ))

        n      = len(picks)
        col_w  = [W / n] * n
        img_sz = min(5.2 * cm, W / n - 0.3 * cm)

        label_row = []
        idx_row   = []
        img_row   = []

        for label, z_idx, path in picks:
            # 该切片肿瘤面积
            area_at = next((r.tumor_area_px for r in sr if r.slice_index == z_idx), 0)
            label_row.append(Paragraph(label,
                _sty(f"il{z_idx}", size=9, bold=True, color=BLUE_DARK, align=1)))
            idx_row.append(Paragraph(
                f"切片 {z_idx}　肿瘤截面 {area_at:,} px²",
                _sty(f"ii{z_idx}", size=7.5, color=TEXT_GRAY, align=1)))
            img_row.append(RLImage(str(path), width=img_sz, height=img_sz))

        img_t = Table([label_row, idx_row, img_row], colWidths=col_w)
        img_t.setStyle(TableStyle([
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("BACKGROUND",   (0, 2), (-1, 2), colors.black),
            ("BOX",          (0, 0), (-1, -1), 0.8, GRAY_BORDER),
            ("INNERGRID",    (0, 0), (-1, -1), 0.5, GRAY_BORDER),
            ("LINEBELOW",    (0, 1), (-1, 1), 0.5, GRAY_BORDER),
        ]))
        story.append(img_t)
        story.append(Spacer(1, 0.5 * cm))

    # ── 免责声明（紧贴上方内容，减少留白）────────────────────────────────────
    story.append(Spacer(1, 0.15 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_BORDER, spaceAfter=5))
    disclaimer = Table(
        [[Paragraph(
            "【免责声明】本报告由 CT 肝脏与肿瘤 AI 智能分割系统自动生成。"
            "所有测量值（含 RECIST 直径、FLR 等）均为基于 2D 切片分割的估算结果，"
            "非标准临床测量，不构成临床诊断依据。"
            "最终诊断及治疗决策请以专业放射科/外科医师的评估意见为准。",
            _sty("dis", size=8, color=TEXT_GRAY),
        )]],
        colWidths=[W],
    )
    disclaimer.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), GRAY_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, GRAY_BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(disclaimer)

    # ── 构建 ──────────────────────────────────────────────────────────────────
    cb = _make_cb(case, job)
    doc.build(story, onFirstPage=cb, onLaterPages=cb)
    return buf.getvalue()
