"""
build_docx.py
Converts the LaTeX manuscript into the CEE 176E submission format:
  * .docx (editable Word), 12 pt Cambria body text, black
  * ACS-style numbered references (citeproc + american-chemical-society.csl)
  * native Word equations (OMML) numbered in order; native Word tables
  * figures from ../figures/course/ (rubric-compliant styling)
  * "Figure n." / "Table n." captions; all cross-references resolved to text

    python build_docx.py      # writes Chicago_LSLR_Manuscript.docx
Requires pandoc >= 3 and python-docx.
"""
from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
OUT = HERE / "Chicago_LSLR_Manuscript.docx"
FONT = "Cambria"


# ----------------------------------------------------------------- helpers
def read(p):
    return (HERE / p).read_text()


def _clean_table(name: str, txt: str) -> str:
    if name == "tab_consent.tex":
        txt = re.sub(r"\$(-?\d+\.\d+)(\^\{[^}]*\})?\$", lambda m: m.group(1) + (m.group(2) or ""), txt)
        txt = re.sub(r"\$\((\d+\.\d+)\)\$", r"(\1)", txt)
    return txt


def inline_inputs(tex: str) -> str:
    tex = tex.replace(r"\input{results_text.tex}", read("results_text.tex"))
    tex = re.sub(r"\\csname @@input\\endcsname\s+(tab_[a-z]+\.tex)\s*", lambda m: _clean_table(m.group(1), read(m.group(1))) + "\n", tex)
    return tex


def strip_preamble(tex: str) -> tuple[str, str]:
    pre, body = tex.split(r"\begin{document}", 1)
    macros = read("results.tex")
    keep = "\n".join(l for l in pre.splitlines() if l.startswith(r"\newcommand"))
    return macros + "\n" + keep, body.split(r"\end{document}")[0]


def brace_block(s: str, start: int) -> int:
    """Index just past the brace group that opens at s[start] == '{'."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


# ---------------------------------------------------------- numbering pass
def number_everything(body: str):
    labels: dict[str, str] = {}
    sec = [0, 0]
    appendix = False
    app_letter = 0
    eq = fig = tab = prop = 0
    out = []
    pos = 0
    token = re.compile(r"\\appendix|\\section\*?\{|\\subsection\*?\{|\\begin\{(equation|align|figure|table|proposition|algorithm)\}")
    while True:
        m = token.search(body, pos)
        if not m:
            out.append(body[pos:])
            break
        out.append(body[pos:m.start()])
        t = m.group(0)
        if t == r"\appendix":
            appendix = True
            pos = m.end()
            continue
        if t.startswith(r"\section") or t.startswith(r"\subsection"):
            star = "*" in t
            end = brace_block(body, m.end() - 1)
            title = body[m.end():end - 1]
            lab = re.match(r"\s*\\label\{([^}]+)\}", body[end:])
            if t.startswith(r"\section"):
                if appendix:
                    app_letter += 1
                    num = "ABCDEFGHIJ"[app_letter - 1]
                    out.append(r"\section*{Appendix " + num + ". " + title + "}")
                    if lab:
                        labels[lab.group(1)] = f"Appendix {num}"
                elif star:
                    out.append(r"\section*{" + title + "}")
                else:
                    sec = [sec[0] + 1, 0]
                    out.append(r"\section{" + title + "}")
                    if lab:
                        labels[lab.group(1)] = f"Section {sec[0]}"
            else:
                sec[1] += 1
                out.append(r"\subsection{" + title + "}")
                if lab:
                    labels[lab.group(1)] = f"Section {sec[0]}.{sec[1]}"
            pos = end + (lab.end() if lab else 0)
            continue
        env = m.group(1)
        endtag = r"\end{" + env + "}"
        e = body.index(endtag, m.end()) + len(endtag)
        block = body[m.start():e]
        if env == "equation":
            eq += 1
            for lb in re.findall(r"\\label\{([^}]+)\}", block):
                labels[lb] = f"({eq})"
            inner = re.sub(r"\\label\{[^}]+\}", "", block[len(r"\begin{equation}"):-len(endtag)])
            block = r"\begin{equation*}" + inner.rstrip() + rf"\qquad\qquad ({eq})" + "\n" + r"\end{equation*}"
        elif env == "align":
            rows = block[len(r"\begin{align}"):-len(endtag)].split(r"\\")
            eqs = []
            for r in rows:
                if not r.strip():
                    continue
                lb = re.search(r"\\label\{([^}]+)\}", r)
                r = r.replace(lb.group(0), "") if lb else r
                parts = [p.strip() for p in r.split("&") if p.strip()]
                # parts: [lhs-relation, forall, (description)]
                desc = ""
                if parts and parts[-1].startswith(r"\text{("):
                    desc = parts.pop()
                    desc = re.sub(r"\\text\{\((.*)\)\}$", r"\1", desc)
                    desc = desc.replace(r"\text{", "").replace("}", "").replace(r"\bar T", "T").strip("() ")
                math = r"\qquad ".join(parts)
                if lb:
                    eq += 1
                    labels[lb.group(1)] = f"({eq})"
                    math += rf"\qquad\text{{{desc}}}\qquad ({eq})" if desc else rf"\qquad ({eq})"
                eqs.append(r"\begin{equation*}" + math + r"\end{equation*}")
            block = "\n".join(eqs)
        elif env == "figure":
            fig += 1
            for lb in re.findall(r"\\label\{([^}]+)\}", block):
                labels[lb] = f"Figure {fig}"
            block = re.sub(r"\\label\{[^}]+\}", "", block)
            block = block.replace(r"\caption{", r"\caption{Figure " + str(fig) + ". ", 1)
            block = re.sub(r"\\includegraphics\[[^\]]*\]\{([^}]+)\.pdf\}",
                           r"\\includegraphics[width=6.5in]{../figures/course/\1.png}", block)
        elif env == "table":
            tab += 1
            for lb in re.findall(r"\\label\{([^}]+)\}", block):
                labels[lb] = f"Table {tab}"
            block = re.sub(r"\\label\{[^}]+\}", "", block)
            block = block.replace(r"\caption{", r"\caption{Table " + str(tab) + ". ", 1)
        elif env == "proposition":
            prop += 1
            for lb in re.findall(r"\\label\{([^}]+)\}", block):
                labels[lb] = f"Proposition {prop}"
            head = re.match(r"\\begin\{proposition\}\[([^\]]*)\]", block)
            inner = block[head.end() if head else len(r"\begin{proposition}"):-len(endtag)]
            inner = re.sub(r"\\label\{[^}]+\}", "", inner)
            block = (r"\textbf{Proposition " + str(prop) + (f" ({head.group(1)})." if head else ".") + "} "
                     + r"\emph{" + inner.strip() + "}")
        elif env == "algorithm":
            labels["alg:pipeline"] = "Box 1"
            steps = [
                r"Fetch the raw data and verify checksums; join the inventory to its geocodes.",
                r"Train the tract-blocked classifier on field-verified lines and impute $P(\text{lead})$ for unknown lines.",
                r"Build tract features (ACS, SVI, HOLC, CDPH) and cluster action lines into the zones $\mathcal{Z}$.",
                r"Compute $h_z$ by \cref{eq:harm}; define the EJ zones and the most vulnerable quintile.",
                r"Simulate the heuristics: empirical status quo, proportional pace and harm priority.",
                r"Solve \cref{eq:obj} subject to \eqref{c:complete}--\eqref{c:d} for CM, HW, EW, EW-convex, DW and the robustness variants.",
                r"Set $G^\star$ to the HW gap; solve \cref{eq:eps} for $\varepsilon = G^\star-1,\dots,G^\star-6$ and the reversed problem for $\varepsilon \in \{G^\star+1.5,\ G^\star+3,\ 0,\ 2\}$.",
                r"Evaluate all schedules on a common yardstick (central harm, $r=3\%$).",
                r"Draw 2,000 parameter vectors, re-evaluate the fixed schedules, and estimate the consent models.",
            ]
            items = "\n".join(r"\item " + s for s in steps)
            block = (r"\textbf{Box 1. End-to-end pipeline.} \emph{Input:} public raw files (Table 1) and "
                     r"parameters (config.py). \emph{Output:} schedules, frontier, sensitivity distributions, "
                     r"tables and figures." + "\n" + r"\begin{enumerate}" + "\n" + items + "\n" + r"\end{enumerate}")
        out.append(block)
        pos = e
    return "".join(out), labels


def resolve_refs(body: str, labels: dict) -> str:
    def one(key, cap=False):
        v = labels.get(key, "??")
        return v

    def multi(m):
        keys = [k.strip() for k in m.group(2).split(",")]
        vals = [one(k) for k in keys]
        missing = [k for k, v in zip(keys, vals) if v == "??"]
        if missing:
            print("UNRESOLVED:", missing)
        if len(vals) > 1 and not missing and len({v.split()[0] for v in vals}) == 1 and not vals[0].startswith("("):
            word = vals[0].split()[0]
            plural = {"Section": "Sections", "Figure": "Figures", "Table": "Tables",
                      "Appendix": "Appendices", "Proposition": "Propositions"}.get(word, word)
            nums = [v.split(" ", 1)[1] for v in vals]
            return f"{plural} {' and '.join(nums)}"
        vals = [("Eq. " + v if v.startswith("(") and m.group(1) in ("cref", "Cref") else v) for v in vals]
        return " and ".join(vals)
    body = re.sub(r"\\(cref|Cref)\{([^}]+)\}", multi, body)
    body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: labels.get(m.group(1), "??"), body)
    body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "??").split(" ")[-1], body)
    return body


def listings(body: str) -> str:
    def rep(m):
        opts, path = m.group(1) or "", m.group(2)
        text = (HERE / path).read_text()
        fl = re.search(r"firstline=(\d+)", opts)
        ll = re.search(r"lastline=(\d+)", opts)
        if fl and ll:
            text = "\n".join(text.splitlines()[int(fl.group(1)) - 1:int(ll.group(1))])
        return "\n\\begin{verbatim}\n" + text.rstrip() + "\n\\end{verbatim}\n"
    return re.sub(r"\\lstinputlisting(?:\[([^\]]*)\])?\{([^}]+)\}", rep, body)


def simplify(body: str) -> str:
    # data statement box -> quotation
    i = body.find(r"\medskip\noindent\fbox{")
    if i >= 0:
        j = brace_block(body, body.index("{", i + len(r"\medskip\noindent\fbox")))
        inner = body[i:j]
        txt = inner[inner.index(r"\small") + len(r"\small"):].rstrip("}").strip()
        body = body[:i] + "\\begin{quote}\n" + txt + "\n\\end{quote}\n" + body[j:]
    body = re.sub(r"\\paragraph\{([^}]*)\}", r"\\textbf{\1} ", body)
    body = body.replace(r"\symbf", r"\mathbf")
    for a_, b_ in ((r"\Bigl(", r"\left("), (r"\Bigr)", r"\right)"), (r"\bigl(", r"\left("), (r"\bigr)", r"\right)"),
                   (r"\bigl[", r"\left["), (r"\bigr]", r"\right]"), (r"\Bigl[", r"\left["), (r"\Bigr]", r"\right]"),
                   (r"\!", "")):
        body = body.replace(a_, b_)
    body = re.sub(r"\\begin\{proof\}(\[[^\]]*\])?", r"\\emph{Proof.} ", body).replace(r"\end{proof}", "")
    body = body.replace(r"\bibliographystyle{plainnat}", "")
    body = body.replace(r"\bibliography{../bib/references}", "\n\\section*{References}\n\nREFSPLACEHOLDER\n\n")
    body = re.sub(r"\\begin\{itemize\}\[[^\]]*\]", r"\\begin{itemize}", body)
    body = re.sub(r"\\begin\{enumerate\}\[[^\]]*\]", r"\\begin{enumerate}", body)
    body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)
    body = body.replace(r"\maketitle", "")
    body = body.replace(r"\addlinespace", "")
    body = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", body)
    for a_, b_ in (("^{**}", "**"), ("^{*}", "*"), ("^{\\dagger}", "\u2020")):
        body = body.replace(a_, b_)
    body = body.replace("$\u2020p<0.10$, $*p<0.05$, $**p<0.01$", "\u2020 p < 0.10, * p < 0.05, ** p < 0.01")
    # tabularx -> tabular with explicit paragraph columns (pandoc reads p{} widths)
    def tx(m):
        start = m.end() - 1
        end = brace_block(body_ref[0], start)
        spec = body_ref[0][start + 1:end - 1]
        if "2.7cm" in spec:
            new = r"\begin{tabular}{p{2.8cm}p{3.4cm}p{1.6cm}p{7.2cm}}"
        else:
            new = r"\begin{tabular}{p{3.6cm}p{11.4cm}}"
        return new, end
    body_ref = [body]
    while True:
        m = re.search(r"\\begin\{tabularx\}\{\\linewidth\}\{", body_ref[0])
        if not m:
            break
        new, end = tx(m)
        body_ref[0] = body_ref[0][:m.start()] + new + body_ref[0][end:]
    body = body_ref[0].replace(r"\end{tabularx}", r"\end{tabular}")
    for _w in ("Sets", "Parameters", "Decision variables"):
        body = body.replace(r"\multicolumn{2}{@{}l}{\emph{" + _w + "}}", r"\emph{" + _w + "} & ")
    return body


# --------------------------------------------------------- reference docx
def make_reference_docx(path: Path):
    default = HERE / "ref_default.docx"
    subprocess.run(["pandoc", "-o", str(default), "--print-default-data-file", "reference.docx"], check=True)
    zin = zipfile.ZipFile(default)
    styles = zin.read("word/styles.xml").decode()
    rfont = f'<w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}" w:eastAsia="{FONT}" w:cs="{FONT}"/>'
    # document defaults: Cambria 12 pt black
    styles = re.sub(r"<w:rPrDefault>.*?</w:rPrDefault>",
                    f"<w:rPrDefault><w:rPr>{rfont}<w:color w:val=\"000000\"/><w:sz w:val=\"24\"/>"
                    f"<w:szCs w:val=\"24\"/><w:lang w:val=\"en-US\"/></w:rPr></w:rPrDefault>", styles, flags=re.S)
    # every style: force Cambria + black (code styles keep monospace)

    def fix_style(m):
        st = m.group(0)
        if any(k in st for k in ('w:styleId="SourceCode"', 'w:styleId="VerbatimChar"')) or re.search(r'w:styleId="[A-Za-z]*Tok"', st):
            return st
        st = re.sub(r"<w:rFonts[^>]*/>", "", st)
        st = re.sub(r'<w:color [^>]*/>', '<w:color w:val="000000"/>', st)
        if "<w:rPr>" in st:
            st = st.replace("<w:rPr>", "<w:rPr>" + rfont, 1)
        else:
            st = st.replace("</w:style>", f"<w:rPr>{rfont}<w:color w:val=\"000000\"/></w:rPr></w:style>")
        return st
    styles = re.sub(r"<w:style [^>]*>.*?</w:style>", fix_style, styles, flags=re.S)
    zout = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED)
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/styles.xml":
            data = styles.encode()
        zout.writestr(item, data)
    zout.close()
    default.unlink()


# ------------------------------------------------------------ post-process
def postprocess(path: Path):
    import docx
    from docx.shared import Pt, RGBColor, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d = docx.Document(str(path))
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    for s in d.sections:
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Inches(1)

    def setfont(run, size=None, mono=False):
        if mono:
            return
        run.font.name = FONT
        rpr = run._element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.insert(0, rf)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), FONT)
        run.font.color.rgb = RGBColor(0, 0, 0)
        if size:
            run.font.size = Pt(size)

    for p in d.paragraphs:
        mono = p.style.name in ("Source Code",)
        for r in p.runs:
            setfont(r, None, mono)
        if p.style.name in ("Image Caption", "Table Caption", "Caption"):
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for r in p.runs:
                setfont(r, 11)
                r.font.italic = False
        if p.style.name == "Captioned Figure" or any(r._element.findall(".//" + qn("w:drawing")) for r in p.runs):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # tables: 9 pt text, horizontal rules (booktabs look), autofit to page width
    for t in d.tables:
        ncol0 = len(t.columns)
        if ncol0 >= 5:
            first = 1.55 if ncol0 >= 10 else 1.9
            widths = [Inches(first)] + [Inches((6.5 - first) / (ncol0 - 1))] * (ncol0 - 1)
            tblPr0 = t._tbl.tblPr
            lay = OxmlElement("w:tblLayout")
            lay.set(qn("w:type"), "fixed")
            tblPr0.append(lay)
            for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
                gc.set(qn("w:w"), str(int(w.inches * 1440)))
            for row in t.rows:
                for cell, w in zip(row.cells, widths):
                    cell.width = w
        t.autofit = ncol0 < 5
        tbl = t._tbl
        tblPr = tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge, val in (("top", "single"), ("bottom", "single"), ("insideH", "nil"), ("insideV", "nil"),
                          ("left", "nil"), ("right", "nil")):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), val)
            el.set(qn("w:sz"), "8")
            el.set(qn("w:color"), "000000")
            borders.append(el)
        old = tblPr.find(qn("w:tblBorders"))
        if old is not None:
            tblPr.remove(old)
        tblPr.append(borders)
        ncol = len(t.columns)
        size = 9 if ncol <= 7 else 8
        for i, row in enumerate(t.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(0)
                    p.paragraph_format.space_before = Pt(0)
                    for r in p.runs:
                        setfont(r, size)
                        if i == 0:
                            r.font.bold = True
            if i == 0:
                for cell in row.cells:
                    tcPr = cell._tc.get_or_add_tcPr()
                    b = OxmlElement("w:tcBorders")
                    bt = OxmlElement("w:bottom")
                    bt.set(qn("w:val"), "single")
                    bt.set(qn("w:sz"), "6")
                    bt.set(qn("w:color"), "000000")
                    b.append(bt)
                    tcPr.append(b)
    d.save(str(path))


# ------------------------------------------------------------------- main
def main():
    tex = inline_inputs(read("main.tex"))
    macros, body = strip_preamble(tex)
    body, labels = number_everything(body)
    body = resolve_refs(body, labels)
    body = simplify(body)
    body = listings(body)
    pre = (HERE / "_docx_src.tex")
    pre.write_text(r"\documentclass{article}" + "\n" + macros + "\n"
                   + r"\title{Algorithmic Equity and Capital Allocation: A Computational Optimization Framework for "
                     "Chicago's {\\textasciitilde}400,000 Lead Service Line Replacements" r" Across Economics, Environmental Justice, "
                     r"and Municipal Policy}\author{Vikram Menon}\date{October 2026}"
                   + "\n" + r"\begin{document}" + body + r"\end{document}")
    md = HERE / "_docx_src.md"
    subprocess.run(["pandoc", str(pre), "-f", "latex", "-t", "markdown", "-s", "-o", str(md), "--wrap=none"],
                   check=True, cwd=HERE)
    text = md.read_text()
    text = text.replace("REFSPLACEHOLDER", "::: {#refs}\n:::")
    md.write_text(text)
    ref = HERE / "_reference.docx"
    make_reference_docx(ref)
    subprocess.run(["pandoc", str(md), "-f", "markdown", "-o", str(OUT), "--citeproc",
                    "--bibliography", str(HERE.parent / "bib" / "references.bib"), "--csl", str(HERE / "acs.csl"),
                    "--reference-doc", str(ref), "--number-sections", "--resource-path", str(HERE),
                    "-M", "link-citations=true", "-M", "reference-section-title=References"],
                   check=True, cwd=HERE)
    postprocess(OUT)
    missing = sorted({k for k in re.findall(r"\?\?", text)})
    print(f"wrote {OUT.name}; unresolved refs: {text.count('??')}")
    for f in (pre, ref):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
