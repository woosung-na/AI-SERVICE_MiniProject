"""
Formatting Node
- 입력: draft_report (마크다운 초안)
- 역할: 섹션 헤더·표·참고문헌 형식 통일, 최종 보고서 저장
- PDF: ReportLab Platypus 기반 전문 보고서 스타일
"""

import re
import time
import logging
from datetime import datetime
from pathlib import Path
from state import ResearchState
from metrics import MetricsTracker

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# A4 기준 좌우 여백 제외 실제 콘텐츠 너비 (mm)
_MARGIN = 20
_CONTENT_W_MM = 210 - _MARGIN * 2  # 170mm


def formatting_node(state: ResearchState, metrics: MetricsTracker = None) -> ResearchState:
    """Formatting Node: 마크다운 정제 + 파일 저장"""
    t0 = time.time()
    logger.info("[Formatting Node] 최종 보고서 포맷팅 시작...")

    draft = state.get("draft_report", "")
    if not draft:
        logger.warning("[Formatting Node] draft_report 없음.")
        return {**state, "final_report": "", "next_action": "end"}

    final_report = _format_report(draft)

    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = OUTPUT_DIR / f"report_{timestamp}.md"
    md_path.write_text(final_report, encoding="utf-8")
    logger.info(f"[Formatting Node] 마크다운 저장: {md_path}")

    _try_export_pdf(final_report, timestamp)

    elapsed = time.time() - t0
    if metrics:
        metrics.record("formatting_node", {
            "report_length": len(final_report),
            "output_path": str(md_path),
            "elapsed_sec": round(elapsed, 2),
        })

    return {**state, "final_report": final_report, "next_action": "end"}


# ── 마크다운 정제 ──────────────────────────────────────────────

def _format_report(draft: str) -> str:
    if "생성일" not in draft and "작성일" not in draft:
        date_str = datetime.now().strftime("%Y년 %m월 %d일")
        draft = f"> 작성일: {date_str} | 생성: HBM4/PIM/CXL 분석 AI Agent\n\n" + draft

    draft = re.sub(r'\n{3,}', '\n\n', draft)
    draft = _fix_markdown_tables(draft)
    draft = _normalize_references(draft)
    return draft.strip()


def _fix_markdown_tables(text: str) -> str:
    """헤더 행 직후에만 구분선 삽입 (데이터 행 사이 삽입 방지)"""
    lines = text.split('\n')
    result = []
    for i, line in enumerate(lines):
        result.append(line)
        if '|' in line and i + 1 < len(lines):
            next_line = lines[i + 1]
            prev_line = lines[i - 1] if i > 0 else ''
            is_first_table_row = '|' not in prev_line
            if (is_first_table_row
                    and '|' in next_line
                    and '---' not in next_line
                    and '---' not in line):
                cols = line.count('|') - 1
                result.append('|' + '---|' * max(cols, 1))
    return '\n'.join(result)


def _normalize_references(text: str) -> str:
    if 'REFERENCE' not in text.upper():
        text += "\n\n## REFERENCE\n\n> 본 보고서는 공개된 논문, 특허, 뉴스 기사를 기반으로 작성되었습니다.\n"
    return text


# ── PDF 생성 ───────────────────────────────────────────────────

def _try_export_pdf(report_text: str, timestamp: str):
    """ReportLab Platypus 기반 전문 보고서 PDF 생성"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.platypus import SimpleDocTemplate
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # ── 폰트 ────────────────────────────────────────────
        FONT = _register_korean_font(pdfmetrics, TTFont)

        # ── 색상 팔레트 ──────────────────────────────────────
        NAVY       = colors.HexColor("#1B2A4A")
        BLUE       = colors.HexColor("#2563EB")
        BLUE_LIGHT = colors.HexColor("#EFF6FF")
        GRAY_D     = colors.HexColor("#374151")
        GRAY_M     = colors.HexColor("#6B7280")
        GRAY_L     = colors.HexColor("#F3F4F6")
        BORDER     = colors.HexColor("#D1D5DB")
        WHITE      = colors.white

        # ── 스타일 ──────────────────────────────────────────
        def ps(name, **kw):
            return ParagraphStyle(name=name, **kw)

        S = {
            "title": ps("title",
                fontName=FONT, fontSize=20, textColor=NAVY,
                leading=28, spaceAfter=2*mm, wordWrap='CJK'),
            "meta": ps("meta",
                fontName=FONT, fontSize=8.5, textColor=GRAY_M,
                leading=13, spaceAfter=6*mm, wordWrap='CJK'),
            "h2": ps("h2",
                fontName=FONT, fontSize=12, textColor=WHITE,
                leading=17, wordWrap='CJK'),
            "h3": ps("h3",
                fontName=FONT, fontSize=11, textColor=BLUE,
                leading=16, spaceBefore=4*mm, spaceAfter=2*mm,
                leftIndent=0, wordWrap='CJK'),
            "h4": ps("h4",
                fontName=FONT, fontSize=10, textColor=NAVY,
                leading=15, spaceBefore=3*mm, spaceAfter=1*mm,
                leftIndent=3*mm, wordWrap='CJK'),
            "body": ps("body",
                fontName=FONT, fontSize=9.5, textColor=GRAY_D,
                leading=15, spaceAfter=2*mm,
                alignment=TA_JUSTIFY, wordWrap='CJK'),
            "bullet": ps("bullet",
                fontName=FONT, fontSize=9.5, textColor=GRAY_D,
                leading=15, spaceAfter=1.5*mm,
                leftIndent=8*mm, firstLineIndent=-4*mm, wordWrap='CJK'),
            "sub_bullet": ps("sub_bullet",
                fontName=FONT, fontSize=9, textColor=GRAY_M,
                leading=14, spaceAfter=1*mm,
                leftIndent=16*mm, firstLineIndent=-4*mm, wordWrap='CJK'),
            "quote": ps("quote",
                fontName=FONT, fontSize=9, textColor=GRAY_M,
                leading=14, spaceAfter=2*mm,
                leftIndent=6*mm, wordWrap='CJK'),
            "th": ps("th",
                fontName=FONT, fontSize=8.5, textColor=WHITE,
                leading=12, alignment=TA_CENTER, wordWrap='CJK'),
            "td": ps("td",
                fontName=FONT, fontSize=8.5, textColor=GRAY_D,
                leading=12, alignment=TA_CENTER, wordWrap='CJK'),
            "td_left": ps("td_left",
                fontName=FONT, fontSize=8.5, textColor=GRAY_D,
                leading=12, alignment=TA_LEFT, wordWrap='CJK'),
        }

        PAGE_W, PAGE_H = A4
        MG = _MARGIN * mm
        CW = _CONTENT_W_MM * mm  # 콘텐츠 너비

        # ── 페이지 헤더/푸터 콜백 ───────────────────────────
        def _page_deco(canvas, doc):
            canvas.saveState()
            canvas.setStrokeColor(NAVY)
            canvas.setLineWidth(0.4)
            # 상단 선 + 헤더
            canvas.line(MG, PAGE_H - 13*mm, PAGE_W - MG, PAGE_H - 13*mm)
            canvas.setFont(FONT, 7.5)
            canvas.setFillColor(GRAY_M)
            canvas.drawString(MG, PAGE_H - 10*mm,
                              "HBM4 / PIM / CXL  경쟁사 R&D 동향 분석 보고서")
            canvas.drawRightString(PAGE_W - MG, PAGE_H - 10*mm,
                                   datetime.now().strftime("%Y.%m.%d"))
            # 하단 선 + 페이지 번호
            canvas.line(MG, 13*mm, PAGE_W - MG, 13*mm)
            canvas.drawCentredString(PAGE_W / 2, 9*mm, f"— {doc.page} —")
            canvas.restoreState()

        # ── 문서 생성 ────────────────────────────────────────
        pdf_path = OUTPUT_DIR / f"report_{timestamp}.pdf"
        doc = SimpleDocTemplate(
            str(pdf_path), pagesize=A4,
            leftMargin=MG, rightMargin=MG,
            topMargin=MG + 6*mm, bottomMargin=MG + 2*mm,
        )

        story = _md_to_story(report_text, S,
                              NAVY, BLUE, BLUE_LIGHT, GRAY_L, BORDER, WHITE, CW, mm)

        doc.build(story, onFirstPage=_page_deco, onLaterPages=_page_deco)
        logger.info(f"[Formatting Node] PDF 저장 완료: {pdf_path}")

    except ImportError:
        logger.debug("[Formatting Node] reportlab 없음. PDF 건너뜀.")
    except Exception as e:
        logger.warning(f"[Formatting Node] PDF 변환 실패: {e}", exc_info=True)


def _register_korean_font(pdfmetrics, TTFont) -> str:
    """한글 폰트 등록. 성공 시 폰트명, 실패 시 'Helvetica' 반환."""
    candidates = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont("KoreanFont", path))
                logger.info(f"[PDF] 폰트 등록: {path}")
                return "KoreanFont"
            except Exception:
                continue
    logger.warning("[PDF] 한글 폰트 없음. Helvetica 사용.")
    return "Helvetica"


# ── 마크다운 → Platypus 변환 ──────────────────────────────────

def _md_to_story(text, S, navy, blue, blue_light, gray_l, border, white, cw, mm):
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, HRFlowable

    story = []
    lines = text.split('\n')
    i = 0

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # ── 빈 줄 ──────────────────────────────────────────
        if not line.strip():
            story.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # ── 수평선 ─────────────────────────────────────────
        if re.match(r'^[-*_]{3,}$', line.strip()):
            story.append(Spacer(1, 1 * mm))
            story.append(HRFlowable(
                width='100%', thickness=0.5,
                color=border, spaceAfter=2 * mm))
            i += 1
            continue

        # ── 제목 H1 ────────────────────────────────────────
        if re.match(r'^# [^#]', line):
            txt = _inline(line[2:])
            story.append(Paragraph(txt, S["title"]))
            i += 1
            continue

        # ── 섹션 H2 (색배경 박스) ───────────────────────────
        if re.match(r'^## [^#]', line):
            txt = _inline(line[3:])
            p = Paragraph(txt, S["h2"])
            t = Table([[p]], colWidths=[cw])
            t.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), navy),
                ('TOPPADDING',    (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('LEFTPADDING',   (0, 0), (-1, -1), 8),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
            ]))
            story.append(Spacer(1, 3 * mm))
            story.append(t)
            story.append(Spacer(1, 2 * mm))
            i += 1
            continue

        # ── 소제목 H3 (파란 왼쪽 선 효과) ──────────────────
        if re.match(r'^### [^#]', line):
            txt = _inline(line[4:])
            # 왼쪽 파란 bar: 1칸짜리 테이블로 구현
            p = Paragraph(txt, S["h3"])
            accent = Table([[p]], colWidths=[cw])
            accent.setStyle(TableStyle([
                ('LEFTPADDING',    (0, 0), (-1, -1), 8),
                ('TOPPADDING',     (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING',  (0, 0), (-1, -1), 3),
                ('LINEBEFORE',     (0, 0), (0, -1), 3, blue),
            ]))
            story.append(accent)
            i += 1
            continue

        # ── 소소제목 H4 ─────────────────────────────────────
        if re.match(r'^#### [^#]', line):
            txt = _inline(line[5:])
            story.append(Paragraph(txt, S["h4"]))
            i += 1
            continue

        # ── 마크다운 표 ─────────────────────────────────────
        if line.startswith('|'):
            tbl_lines = []
            while i < len(lines) and lines[i].startswith('|'):
                tbl_lines.append(lines[i])
                i += 1
            tbl = _make_table(tbl_lines, S, navy, gray_l, blue_light, border, white, cw)
            if tbl:
                story.append(tbl)
                story.append(Spacer(1, 3 * mm))
            continue

        # ── 인용 blockquote ─────────────────────────────────
        if line.startswith('>'):
            inner = line.lstrip('> ')
            txt = _inline(inner)
            p = Paragraph(txt, S["quote"])
            box = Table([[p]], colWidths=[cw])
            box.setStyle(TableStyle([
                ('BACKGROUND',    (0, 0), (-1, -1), gray_l),
                ('LEFTPADDING',   (0, 0), (-1, -1), 10),
                ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
                ('TOPPADDING',    (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LINEBEFORE',    (0, 0), (0, -1), 3, border),
            ]))
            story.append(box)
            i += 1
            continue

        # ── 하위 불릿 (2+ 스페이스) ─────────────────────────
        if re.match(r'^  +[-*] ', line):
            inner = re.sub(r'^  +[-*] ', '', line)
            story.append(Paragraph(f"◦  {_inline(inner)}", S["sub_bullet"]))
            i += 1
            continue

        # ── 불릿 ───────────────────────────────────────────
        if re.match(r'^[-*] ', line):
            inner = line[2:]
            story.append(Paragraph(f"•  {_inline(inner)}", S["bullet"]))
            i += 1
            continue

        # ── 번호 리스트 ─────────────────────────────────────
        if re.match(r'^\d+\. ', line):
            inner = re.sub(r'^\d+\. ', '', line)
            story.append(Paragraph(_inline(inner), S["bullet"]))
            i += 1
            continue

        # ── 일반 본문 ───────────────────────────────────────
        txt = _inline(line)
        if txt:
            story.append(Paragraph(txt, S["body"]))
        i += 1

    return story


def _make_table(tbl_lines, S, navy, gray_l, blue_light, border, white, cw):
    """마크다운 표 → ReportLab Table"""
    from reportlab.platypus import Table, TableStyle, Paragraph

    rows_raw = []
    for ln in tbl_lines:
        # 구분선 행 스킵
        if re.match(r'^\|[\s\-:|]+\|$', ln.strip()):
            continue
        cells = [c.strip() for c in ln.strip('|').split('|')]
        rows_raw.append(cells)

    if not rows_raw:
        return None

    n_cols = max(len(r) for r in rows_raw)
    # 열 너비: 균등 배분
    col_w = cw / n_cols

    data = []
    for ri, row in enumerate(rows_raw):
        # 열 수 맞춤 패딩
        row = row + [''] * (n_cols - len(row))
        style = S["th"] if ri == 0 else S["td"]
        data.append([Paragraph(_inline(c), style) for c in row])

    t = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)

    cmd = [
        # 헤더
        ('BACKGROUND',    (0, 0), (-1, 0), navy),
        ('TEXTCOLOR',     (0, 0), (-1, 0), white),
        # 짝수 행 배경
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, blue_light]),
        # 전체 테두리
        ('GRID',          (0, 0), (-1, -1), 0.4, border),
        ('BOX',           (0, 0), (-1, -1), 0.8, navy),
        # 패딩
        ('TOPPADDING',    (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]
    t.setStyle(TableStyle(cmd))
    return t


def _inline(text: str) -> str:
    """마크다운 인라인 기호 → ReportLab XML 변환.

    순서:
      1. XML 특수문자 이스케이프 (&, <, >)
      2. **bold** → <b>...</b>
      3. *italic* → <i>...</i>
      4. `code` → 텍스트만 유지
      5. ~~strikethrough~~ → 제거
    """
    # 1. XML escape
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    # 2. Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # 3. Italic (bold 처리 후 남은 단일 *)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # 4. Inline code
    text = re.sub(r'`(.+?)`', r'\1', text)
    # 5. Strikethrough
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    return text
