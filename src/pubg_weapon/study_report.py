"""Generate result-driven report sources, a PDF and presentation from one completed study."""
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from .io import read_json, write_json


REPO = 'https://github.com/kttt294/PUBG-Weapon-Groups'
FONT_SIZE = 12.5


def table(frame, digits=4):
    def render(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.{digits}f}'.replace('.', ',')
        return str(value)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---']*len(frame.columns)) + ' |',
                      *['| ' + ' | '.join(render(value) for value in row) + ' |' for row in frame.itertuples(index=False, name=None)]])


def frame_table(frame, title):
    return f'**{title}**\n\n{table(frame)}\n'


def full_summary(root, seeds):
    rows, validations = [], []
    for seed in seeds:
        folder = root / 'full_models' / str(seed)
        val = pd.read_csv(folder / 'validation_metrics.csv')
        test = pd.read_csv(folder / 'test' / 'test_metrics.csv').set_index('name')
        groups = {
            'Majority': val.name.eq('majority_D'),
            'Gaussian NB': val.name.eq('naive_bayes_D'),
            'Logistic SGD': val.name.str.startswith('sgd'),
            'XGBoost Hist': val.name.str.startswith('xgboost'),
        }
        for family, keep in groups.items():
            choices = val.loc[keep & val.feature_set.eq('loadout_history')]
            chosen = choices.sort_values(['macro_f1', 'name'], ascending=[False, True]).iloc[0]
            score = test.loc[chosen['name']]
            rows.append({'seed': seed, 'family': family, 'configuration': chosen['name'],
                         'validation_macro_f1': chosen.macro_f1, **score.to_dict()})
        validations.append(val.assign(seed=seed))
    full = pd.DataFrame(rows)
    pd.concat(validations).to_csv(root / 'full_validation_all_seeds.csv', index=False)
    full.to_csv(root / 'full_test_selected_by_family.csv', index=False)
    return full


def plot_curves(root, seed, out):
    sgd = pd.read_csv(root / 'full_models' / str(seed) / 'sgd_learning_curves.csv')
    xgb = pd.read_csv(root / 'full_models' / str(seed) / 'xgboost_learning_curves.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout='constrained')
    for name, part in sgd.groupby('model'):
        axes[0].plot(part.epoch, part.validation_sample_macro_f1, marker='o', label=name)
    axes[0].set(xlabel='Epoch', ylabel='Validation Macro-F1')
    axes[0].legend(fontsize=7)
    for name, part in xgb.groupby('model'):
        line, = axes[1].plot(part['round'], part.validation_sample_log_loss, label=name+' val')
        axes[1].plot(part['round'], part.train_sample_log_loss, linestyle='--', color=line.get_color(), label=name+' train')
    axes[1].set(xlabel='Boosting round', ylabel='Validation log-loss')
    axes[1].legend(fontsize=7)
    fig.savefig(out / 'learning_curves.png', dpi=175)
    plt.close(fig)


def charts(root, manifest, full, modern, out):
    counts = manifest['splits']['train']['class_counts']
    labels = sorted(counts, key=counts.get)
    fig, ax = plt.subplots(figsize=(7, 3.4), layout='constrained')
    ax.barh(labels, [counts[c] for c in labels], color='#315e7b')
    ax.set_xlabel('Training events')
    fig.savefig(out / 'classes.png', dpi=170)
    plt.close(fig)
    summary = modern.groupby('name').macro_f1.agg(['mean', 'std']).sort_values('mean')
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.barh(summary.index, summary['mean'], xerr=summary['std'].fillna(0), color='#315e7b', capsize=3)
    ax.set_xlabel('Macro-F1, mean and standard deviation over training seeds')
    fig.savefig(out / 'modern_comparison.png', dpi=170)
    plt.close(fig)
    seed = read_json(root / 'study_config.json')['seeds'][0]
    confusion = pd.read_csv(root / 'full_models' / str(seed) / 'test' / 'confusion_matrix.csv', index_col=0)
    normalized = confusion.to_numpy() / confusion.sum(axis=1).to_numpy()[:, None]
    fig, ax = plt.subplots(figsize=(6.5, 4.8), layout='constrained')
    im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(confusion)), confusion.columns)
    ax.set_yticks(np.arange(len(confusion)), confusion.index)
    for (i, j), value in np.ndenumerate(normalized):
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if value > .5 else '#18344a')
    ax.set(xlabel='Predicted class', ylabel='True class')
    fig.colorbar(im, ax=ax)
    fig.savefig(out / 'confusion.png', dpi=170)
    plt.close(fig)
    importance = pd.read_csv(root / 'diagnostics' / str(seed) / 'permutation_importance.csv').sort_values('macro_f1_drop_mean').tail(10)
    fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
    ax.barh(importance.feature, importance.macro_f1_drop_mean, xerr=importance.macro_f1_drop_std, color='#315e7b', capsize=3)
    ax.set_xlabel('Macro-F1 drop on the post-selection interpretation sample')
    fig.savefig(out / 'importance.png', dpi=170)
    plt.close(fig)
    plot_curves(root, seed, out)


def render_pdf(markdown, destination, out):
    from html import escape
    from matplotlib.font_manager import findfont, FontProperties
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
                                   PageBreak, Image, LongTable, TableStyle, KeepTogether)
    from reportlab.platypus.tableofcontents import TableOfContents
    from PIL import Image as PILImage
    for name, weight in [('StudySerif', 'normal'), ('StudyBold', 'bold')]:
        path = findfont(FontProperties(family='DejaVu Serif', weight=weight))
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily('StudySerif', normal='StudySerif', bold='StudyBold', italic='StudySerif', boldItalic='StudyBold')
    width, height = 595.28, 841.89
    content_width = width-85-57
    styles = {
        'body': ParagraphStyle('Body', fontName='StudySerif', fontSize=FONT_SIZE, leading=17, alignment=TA_JUSTIFY, spaceAfter=8),
        'heading': ParagraphStyle('Heading', fontName='StudyBold', fontSize=FONT_SIZE, leading=17, spaceBefore=8, spaceAfter=7, keepWithNext=True),
        'cell': ParagraphStyle('Cell', fontName='StudySerif', fontSize=FONT_SIZE, leading=16),
        'center': ParagraphStyle('Center', fontName='StudySerif', fontSize=FONT_SIZE, leading=17, alignment=TA_CENTER, spaceAfter=8),
    }
    def formatted(text):
        value = escape(text)
        value = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)
        return value
    class Report(BaseDocTemplate):
        def afterFlowable(self, flow):
            if getattr(flow, 'heading_level', None) is not None:
                self.canv.bookmarkPage(flow.key)
                self.notify('TOCEntry', (flow.heading_level, flow.getPlainText(), self.page, flow.key))
        def footer(self, canvas, doc):
            canvas.setFont('StudySerif', FONT_SIZE)
            canvas.drawCentredString(width/2, 33, str(doc.page))
    doc = Report(str(destination), pagesize=(width, height), title='Đồ án Khai phá dữ liệu PUBG',
                 leftMargin=85, rightMargin=57, topMargin=65, bottomMargin=57)
    frame = Frame(85, 57, content_width, height-122, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='study', frames=[frame], onPage=doc.footer))
    story = [Spacer(1, 55)]
    for line in ['ĐẠI HỌC PHENIKAA', 'TRƯỜNG CÔNG NGHỆ THÔNG TIN PHENIKAA',
                 'BÁO CÁO BÀI TẬP LỚN', 'MÔN KHAI PHÁ DỮ LIỆU',
                 'PHÂN LOẠI NHÓM VŨ KHÍ TRONG PUBG', 'TỪ NGỮ CẢNH VÀ LỊCH SỬ SỰ KIỆN',
                 'Giảng viên hướng dẫn - ThS. Phạm Kim Thành', 'Nhóm 3',
                 f'GitHub dự án - {REPO}']:
        story.append(Paragraph(formatted(line), styles['center']))
        story.append(Spacer(1, 12))
    story.extend([PageBreak(), Paragraph('MỤC LỤC', styles['heading'])])
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('Toc0', parent=styles['body'], fontSize=FONT_SIZE, leading=17),
                       ParagraphStyle('Toc1', parent=styles['body'], fontSize=FONT_SIZE, leading=17, leftIndent=12)]
    story.extend([toc, PageBreak()])
    lines = markdown.splitlines()
    index, heading_count = 0, 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line.startswith('#'):
            depth = min(1, len(line)-len(line.lstrip('#'))-1)
            para = Paragraph(formatted(line.lstrip('#').strip()), styles['heading'])
            para.heading_level, para.key = depth, f'heading_{heading_count}'
            heading_count += 1
            story.append(para)
        elif line.startswith('|'):
            rows = [line]
            while index < len(lines) and lines[index].startswith('|'):
                rows.append(lines[index])
                index += 1
            values = [[v.strip() for v in row.strip('|').split('|')] for row in rows]
            values = [r for r in values if not all(set(v) <= set('-: ') for v in r)]
            weights = [min(48, max(10, max(len(r[c]) for r in values))) for c in range(len(values[0]))]
            widths = [content_width*w/sum(weights) for w in weights]
            data = [[Paragraph(formatted(v), styles['cell']) for v in row] for row in values]
            tab = LongTable(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
            tab.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), .35, colors.grey),
                        ('BACKGROUND', (0, 0), (-1, -1), colors.white), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
            story.extend([tab, Spacer(1, 10)])
        elif line.startswith('!['):
            caption, name = re.match(r'!\[(.*?)\]\((.*?)\)', line).groups()
            path = out / name
            with PILImage.open(path) as image:
                w, h = image.size
            factor = min(content_width/w, 280/h)
            story.append(KeepTogether([Image(str(path), width=w*factor, height=h*factor),
                         Paragraph(formatted(caption), styles['center'])]))
        else:
            story.append(Paragraph(formatted(line), styles['body']))
    doc.multiBuild(story)


def latex_source(markdown, out):
    def escape(text):
        values = {'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}', '^': r'\textasciicircum{}'}
        return ''.join(values.get(c, c) for c in text)
    lines = [r'\documentclass[12pt,a4paper]{article}',
             r'\usepackage{fontspec,geometry,longtable,array,graphicx,amsmath,hyperref}',
             r'\geometry{left=30mm,right=20mm,top=23mm,bottom=23mm}',
             r'\setmainfont{TeX Gyre Termes}',
             r'\usepackage{tocloft}', r'\renewcommand{\contentsname}{MỤC LỤC}',
             r'\makeatletter', r'\renewcommand\normalsize{\@setfontsize\normalsize{12.5}{17}}',
             r'\renewcommand\small{\@setfontsize\small{12.5}{17}}',
             r'\renewcommand\footnotesize{\@setfontsize\footnotesize{12.5}{17}}',
             r'\renewcommand\large{\@setfontsize\large{12.5}{17}}',
             r'\renewcommand\Large{\@setfontsize\Large{12.5}{17}}',
             r'\renewcommand\LARGE{\@setfontsize\LARGE{12.5}{17}}', r'\makeatother',
             r'\setlength{\parindent}{0pt}', r'\setlength{\parskip}{7pt}',
             r'\begin{document}\normalsize', r'\begin{titlepage}\centering',
             r'\textbf{ĐẠI HỌC PHENIKAA}\\[15mm]', r'\textbf{BÁO CÁO BÀI TẬP LỚN KHAI PHÁ DỮ LIỆU}\\[15mm]',
             r'\textbf{PHÂN LOẠI NHÓM VŨ KHÍ TRONG PUBG}\\[10mm]',
             r'Giảng viên hướng dẫn - ThS. Phạm Kim Thành\\[10mm]',
             r'Nhóm 3\\[15mm]', r'GitHub dự án - \url{' + REPO + r'}',
             r'\end{titlepage}\tableofcontents\clearpage']
    source = markdown.splitlines()
    index = 0
    while index < len(source):
        line = source[index].strip()
        index += 1
        if not line:
            continue
        if line.startswith('#'):
            level = min(2, len(line)-len(line.lstrip('#'))-1)
            kind = ('section', 'subsection', 'subsubsection')[level]
            title = escape(line.lstrip('#').strip())
            lines.extend(['\\'+kind+'*{\\fontsize{12.5}{17}\\selectfont '+title+'}',
                          r'\addcontentsline{toc}{'+kind+'}{'+title+'}'])
        elif line.startswith('|'):
            rows = [line]
            while index < len(source) and source[index].startswith('|'):
                rows.append(source[index])
                index += 1
            values = [[v.strip() for v in row.strip('|').split('|')] for row in rows]
            values = [r for r in values if not all(set(v) <= set('-: ') for v in r)]
            width = .88 / len(values[0])
            spec = ''.join(r'>{\raggedright\arraybackslash}p{'+f'{width:.4f}'+r'\linewidth}' for _ in values[0])
            lines.append(r'\begin{longtable}{'+spec+'}')
            lines.extend(' & '.join(escape(v) for v in row)+r' \\' for row in values)
            lines.append(r'\end{longtable}')
        elif line.startswith('!['):
            caption, path = re.match(r'!\[(.*?)\]\((.*?)\)', line).groups()
            lines.extend([r'\begin{center}\includegraphics[width=.95\linewidth]{'+path+'}',
                          escape(caption), r'\end{center}'])
        else:
            lines.append(escape(line.replace('**', ''))+'\n')
    lines.append(r'\end{document}')
    (out / 'main.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def build_deliverables(root):
    root = Path(root)
    out = root / 'deliverables'
    out.mkdir(exist_ok=True)
    config = read_json(root / 'study_config.json')
    manifest = read_json(root / 'prepared' / 'dataset_manifest.json')
    full = full_summary(root, config['seeds'])
    modern = pd.read_csv(root / 'modern_comparison' / 'test_metrics.csv')
    modern_val = pd.read_csv(root / 'modern_comparison' / 'validation_metrics.csv')
    seed = config['seeds'][0]
    representative = root / 'full_models' / str(seed)
    cohorts = read_json(root / 'comparison_cohorts' / 'manifest.json')
    cohort_table = pd.DataFrame(cohorts['cohorts']).query('seed == @seed')[['split', 'rows', 'matches']]
    charts(root, manifest, full, modern, out)
    summary = full.groupby('family')[['macro_f1', 'accuracy', 'weighted_f1', 'top_2_accuracy']].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    compact_full = summary[['family', 'macro_f1_mean', 'macro_f1_std', 'accuracy_mean', 'top_2_accuracy_mean']]
    modern_summary = modern.groupby('name')[['macro_f1', 'accuracy', 'fit_seconds', 'predict_seconds']].agg(['mean', 'std'])
    modern_summary.columns = ['_'.join(c) for c in modern_summary.columns]
    modern_summary = modern_summary.reset_index()
    compact_modern = modern_summary[['name', 'macro_f1_mean', 'macro_f1_std', 'accuracy_mean', 'fit_seconds_mean']]
    compact_full.to_csv(out / 'full_summary.csv', index=False)
    compact_modern.to_csv(out / 'modern_summary.csv', index=False)
    winner = full.groupby('family').validation_macro_f1.mean().idxmax()
    modern_winner = modern_val.groupby('name').macro_f1.mean().idxmax()
    ablation = pd.concat([pd.read_csv(root / 'full_models' / str(s) / 'test' / 'ablation.csv').assign(seed=s) for s in config['seeds']])
    abl_summary = ablation.groupby('ablation')[['macro_f1', 'accuracy']].agg(['mean', 'std'])
    abl_summary.columns = ['_'.join(c) for c in abl_summary.columns]
    abl_summary = abl_summary.reset_index()
    raw_quality = pd.read_csv(root / 'prepared' / 'raw_column_quality.csv')
    cleaning = pd.read_csv(root / 'prepared' / 'cleaning_counts.csv')[['rule', 'before', 'removed', 'remaining']]
    numeric = pd.read_csv(root / 'eda' / 'train_numeric_summary.csv')
    quantile = pd.read_csv(root / 'eda' / 'train_distance_quantiles.csv')
    stats = pd.DataFrame(manifest['splits']).T[['rows', 'matches']].reset_index(names='split')
    per_class = pd.read_csv(representative / 'test' / 'per_class_metrics.csv')
    selected = read_json(representative / 'selection.json')['best']
    per_class = per_class[per_class.name.eq(selected)].drop(columns='name')
    history = pd.read_csv(representative / 'test' / 'history_subset_metrics.csv')
    history = history[history.name.isin(['sgd_C', 'sgd_D', selected])][['name', 'has_prior_kill', 'rows', 'macro_f1', 'accuracy']]
    errors = pd.read_csv(root / 'diagnostics' / str(seed) / 'error_profiles.csv')
    distances = errors[errors.category.eq('distance')][['group', 'rows', 'macro_f1', 'accuracy']]
    resource_rows = []
    for s in config['seeds']:
        r = read_json(root / 'full_models' / str(s) / 'run_metadata.json')
        for name, row in r['resources'].items():
            resource_rows.append({'seed': s, 'stage': name, 'seconds': row['wall_seconds'], 'peak_rss_mib': row['process_tree_peak_rss_mib']})
    resource_frame = pd.DataFrame(resource_rows)
    resource_frame.to_csv(out / 'full_resources.csv', index=False)
    worker = pd.read_csv(root / 'eda' / 'worker_comparison.csv')[['workers', 'rows', 'wall_seconds', 'process_tree_peak_rss_mib', 'rows_per_second']]
    sections = [
        '# 1. GIỚI THIỆU BÀI TOÁN',
        '## 1.1. Bài toán, mục tiêu và phạm vi',
        'Đồ án phân loại sáu nhóm vũ khí gây hạ gục trong PUBG từ ngữ cảnh của sự kiện và lịch sử đã quan sát trước đó. Mỗi dòng là một sự kiện. Nhãn được suy ra từ killed_by. Đầu vào gồm khoảng cách, thời điểm, bản đồ, vị trí và lịch sử hợp lệ trong cùng trận. Nghiên cứu giả định ngữ cảnh của sự kiện hiện tại đã được ghi nhận.',
        'Câu hỏi nghiên cứu gồm vai trò khoảng cách, phần đóng góp của thời gian và vị trí, giá trị của lịch sử, mức độ nhầm lẫn giữa các lớp và hiệu quả của các họ mô hình khi thay đổi quy mô dữ liệu.',
        '## 1.2. Nguồn dữ liệu và thiết kế tổng thể',
        'Nguồn là PUBG Match Deaths and Statistics trên Kaggle. Năm CSV sự kiện được đọc đầy đủ, gom theo mã trận, làm sạch và xây dựng đặc trưng. So sánh toàn dữ liệu dùng tất cả dòng train hợp lệ. So sánh TabM và TabICLv2 dùng cùng các trận và hàng với các mô hình đối chứng trong một bộ dữ liệu có giới hạn tài nguyên.',
        f'Mã nguồn được lưu tại {REPO}. Seed dữ liệu bằng 42. Ba seed mô hình là {config["seeds"]}. Tập đánh giá đã được sử dụng trong thực nghiệm trước và được giữ cố định để đánh giá lại theo giao thức đã định. Mọi lựa chọn tham số ở lượt này dựa trên validation.',
        '# 2. XÂY DỰNG TẬP DỮ LIỆU',
        '## 2.1. Kích thước, thuộc tính và chất lượng đầu vào',
        frame_table(pd.DataFrame(manifest['files']), 'Bảng 1. Kiểm kê năm tệp nguồn'),
        f'Tổng đầu vào có {manifest["source_rows"]:,} dòng và {manifest["available_matches"]:,} mã trận. Các trường tên người chơi và mã trận là định danh để ghép lịch sử, bản đồ và nguyên nhân tử vong là biến phân loại, tọa độ và thời điểm là biến số. Thứ hạng cuối trận được giữ cho kiểm tra dữ liệu và loại khỏi đặc trưng chính.',
        frame_table(raw_quality, 'Bảng 2. Chất lượng từng cột trước chuẩn hóa kiểu số'),
        'Missing là giá trị thiếu do CSV reader nhận diện, blank là chuỗi rỗng sau bỏ khoảng trắng. Invalid_number là giá trị không thể chuyển sang số trong cột số, còn infinite đếm vô cực. Bảng thống kê trên toàn bộ nguồn trước lọc.',
        '## 2.2. Nhãn, làm sạch và tập đánh giá',
        'Các nhóm AR, DMR, SR, SMG, Shotgun, LMG và Other được ánh xạ tường minh. Các nguyên nhân môi trường bị loại. Nhãn dưới 1% được xác định từ train và gộp vào Other. Kiểm tra tọa độ dựa trên miền [0, 800000] và thời gian (0, 2700] giây. Mỗi dòng được đếm ở bước đầu tiên loại nó.',
        frame_table(cleaning, 'Bảng 3. Kiểm kê làm sạch theo trình tự'),
        frame_table(stats, 'Bảng 4. Chia tập theo mã trận sau làm sạch'),
        'Tập train, validation và test có mã trận rời nhau. Các sự kiện cùng trận nằm chung một phân vùng kể cả khi đến từ nhiều CSV. Trọng số lớp và phép chuẩn hóa đều học từ train. Validation chọn mô hình và checkpoint. Test chỉ cung cấp kết quả đánh giá của các lựa chọn đã khóa.',
        '## 2.3. Phân phối và thống kê mô tả',
        '![Hình 1. Phân phối nhóm vũ khí trên toàn bộ train.](classes.png)',
        frame_table(numeric, 'Bảng 5. Thống kê số trên toàn bộ train'),
        frame_table(quantile, 'Bảng 6. Phân vị khoảng cách ước lượng từ histogram toàn train'),
        'Các phân vị trong Bảng 6 là cận dưới của ô histogram, có độ rộng 1 mét tới 1000 mét và 10 mét ở phần còn lại. Dữ liệu không cần nạp toàn bộ vào RAM để cộng histogram. Phân bố khoảng cách chồng lấn giữa các lớp là một nguyên nhân khiến thông tin khoảng cách riêng lẻ chưa đủ phân biệt vũ khí.',
        '## 2.4. Thiết kế đặc trưng theo thời gian',
        'Khoảng cách Euclid được tính từ hai tọa độ và chia 100 để đổi sang mét theo cấu hình. Mức A có khoảng cách. B thêm thời gian. C thêm log khoảng cách, giai đoạn, bản đồ và sáu biến vị trí để có 11 thuộc tính. D thêm sáu biến lịch sử để có 17 thuộc tính trước one-hot.',
        'Lịch sử dùng cặp match_id và killer_name và điều kiện t_trước < t_hiện_tại. Các sự kiện cùng thời điểm không nhìn nhãn của nhau. Sáu biến gồm có lịch sử hay chưa, số lần trước đó, nhóm phổ biến ở mốc trước gần nhất, khoảng cách trung bình ở mốc đó, thời gian từ mốc đó và nhóm phổ biến trong toàn bộ lịch sử trước. Khi chưa có lịch sử, nhóm nhận NONE, số lượng và khoảng cách nhận 0, thời gian từ lần trước nhận -1.',
        '# 3. CƠ SỞ LÝ THUYẾT VÀ THUẬT TOÁN',
        '## 3.1. Các phương pháp trong chương trình học',
        'Majority chọn lớp phổ biến nhất của train. Gaussian Naive Bayes dùng định lý Bayes và giả thiết các thuộc tính độc lập khi đã biết lớp, ước lượng trung bình và phương sai Gaussian theo lớp. Với khoảng cách và log khoảng cách cùng xuất hiện, giả thiết độc lập chỉ là xấp xỉ.',
        'Logistic Regression biến đổi điểm tuyến tính thành xác suất bằng sigmoid trong từng bộ phân loại một lớp so với phần còn lại. Hàm mất mát log-loss phạt dự đoán tự tin nhưng sai. SGD cập nhật hệ số theo gradient trên các lô, kết hợp điều chuẩn L2 để hạn chế hệ số quá lớn. Thống kê chuẩn hóa được cập nhật trước khi học mô hình.',
        'Cây quyết định tìm ngưỡng chia làm giảm độ hỗn tạp nhãn. Random Forest tổng hợp cây học trên mẫu bootstrap và lựa chọn thuộc tính ngẫu nhiên để giảm phương sai. HistGradientBoosting bổ sung cây tuần tự theo gradient của hàm mất mát và dùng histogram khi tìm ngưỡng chia.',
        '## 3.2. XGBoost Hist và bộ nhớ ngoài',
        'XGBoost tối ưu tổng hàm mất mát cộng điều chuẩn của cây. Ở từng vòng, mô hình dùng gradient và đạo hàm bậc hai để ước lượng lợi ích khi chia nút. Tham số chiều sâu quyết định độ phức tạp của tương tác. Learning rate giảm đóng góp mỗi cây. ExtMemQuantileDMatrix duyệt các lô Parquet và dùng trang đệm trên đĩa, giúp học trên toàn bộ train khi RAM hữu hạn [3].',
        '## 3.3. TabM và TabICLv2',
        'TabM (ICLR 2025) tổ hợp nhiều mạng MLP chia sẻ phần lớn tham số và giữ một số hệ số điều biến theo thành viên. Mỗi mẫu tạo nhiều bộ logits. Khi học, cross-entropy được tính cho từng thành viên. Khi dự đoán, các xác suất softmax được lấy trung bình. Cấu hình thực nghiệm dùng 16 thành viên, 3 khối, chiều rộng 256, dropout 0,1 và không dùng feature embeddings [1].',
        'TabICLv2 (ICML 2026) là mô hình dữ liệu bảng đã huấn luyện trước. Khi thích nghi, các dòng train có nhãn được dùng làm ngữ cảnh để mô hình dự đoán nhãn của truy vấn. Cơ chế này không thực hiện các epoch gradient trên PUBG. Lượt này dùng checkpoint classifier v2 ngày 12/02/2026, tổ hợp 4 biến đổi, xử lý lần lượt từng thành viên và lưu representation cache. Chi phí tiền huấn luyện bên ngoài PUBG không nằm trong thời gian fit đo tại đây [2].',
        'Bộ so sánh hiện đại dùng cùng phép mã hóa one-hot và cùng hàng cho mọi họ. Các mô hình trong bộ so sánh này dùng phân bố lớp tự nhiên và không dùng trọng số lớp. Lượt toàn train có cả XGBoost trọng số cân bằng và không trọng số để đối chiếu. Vì quy mô học khác nhau, hai bảng kết quả tương ứng có phạm vi kết luận riêng.',
        '# 4. THỰC NGHIỆM VÀ KẾT QUẢ',
        '## 4.1. Giao thức và chỉ số',
        'Mỗi mô hình được chạy từ đầu với ba seed mô hình, giữ cùng dữ liệu chia tập. Kết quả trung bình và độ lệch chuẩn mô tả độ nhạy với ngẫu nhiên huấn luyện. Độ lệch chuẩn ba seed không phải khoảng tin cậy. Macro-F1 là trung bình F1 theo lớp với trọng số bằng nhau, Accuracy đo tỷ lệ đúng, F1 trọng số dùng số mẫu của từng lớp và Top-2 đo nhãn thật có nằm trong hai ứng viên đầu hay không.',
        'Khoảng tin cậy 95% và chênh lệch giữa mô hình được tính bằng 1000 lần bootstrap theo trận. Các mô hình được so trong cùng một lượt lấy lại cùng nhóm trận. Quy trình giúp giữ phụ thuộc giữa các sự kiện trong trận. Các khoảng tin cậy theo từng seed và cặp mô hình có trong CSV đi kèm.',
        'Logistic SGD khảo sát ba alpha và theo dõi tối đa năm epoch, dừng khi Macro-F1 của mẫu validation cố định không cải thiện. XGBoost khảo sát depth 4 và 6 với trọng số cân bằng, cùng depth 6 không trọng số. Mỗi cấu hình học tối đa 120 vòng và early stopping dựa trên log-loss của mẫu validation. Lựa chọn cuối giữa các họ dựa trên toàn validation.',
        frame_table(cohort_table, 'Bảng 7. Cùng bộ dữ liệu cho các phương pháp hiện đại và đối chứng'),
        '## 4.2. So sánh trên toàn bộ train',
        frame_table(compact_full, 'Bảng 8. Trung bình và độ lệch chuẩn trên toàn bộ test'),
        f'Họ được chọn theo Macro-F1 validation trung bình là {winner}. Mỗi seed dùng cấu hình có điểm validation cao nhất trong họ. Các mô hình Majority và Gaussian NB có thể trùng kết quả giữa seed vì bản chất các phép tính ít hoặc không dùng ngẫu nhiên. Giá trị std bằng 0 trong trường hợp này là kết quả có thể xảy ra.',
        '![Hình 2. Đường học trên dữ liệu phát triển ở seed đầu.](learning_curves.png)',
        'Đường học cho biết thời điểm cải thiện ngừng lại và mức độ khác nhau giữa các cấu hình. Bộ SGD dùng cùng một mẫu train để đo log-loss qua epoch và cùng một mẫu validation cố định. Việc dừng SGD chọn bằng Macro-F1, còn XGBoost dừng theo log-loss. Các đường học của tất cả seed được lưu riêng để kiểm tra độ hội tụ.',
        '## 4.3. Hai phương pháp gần đây và các đối chứng',
        frame_table(compact_modern, 'Bảng 9. So sánh trên cùng các trận của bộ giới hạn kích thước'),
        f'Họ dẫn đầu theo Macro-F1 validation trung bình trong bộ so sánh là {modern_winner}. TabM khảo sát hai learning rate và chọn theo validation, còn TabICLv2 giữ cấu hình tổ hợp đã định. Bảng test mô tả kết quả sau các lựa chọn này.',
        '![Hình 3. So sánh phương pháp hiện đại qua ba seed mô hình.](modern_comparison.png)',
        'Thời gian fit trong Bảng 9 là thời gian của cấu hình được chọn. Chi phí toàn bộ tìm kiếm và bộ nhớ từng bước nằm trong resources.json. TabICLv2 phải tạo ngữ cảnh/caches và trả lời truy vấn. Thời gian dự đoán của nó cần được đọc cùng thời gian fit để hiểu tổng chi phí thích nghi.',
        '## 4.4. Đóng góp của nhóm đặc trưng',
        frame_table(abl_summary, 'Bảng 10. Ablation Logistic SGD qua ba seed'),
        'Cả bốn mức dùng cùng alpha 0,0001 để tránh quy mức thay đổi cho việc đổi tham số. Mỗi mức chọn epoch tốt nhất từ cùng validation theo cùng quy tắc. Việc thêm khối theo thứ tự A, B, C, D chỉ đo lợi ích trong thứ tự đó. Chênh lệch và bootstrap của từng seed được ghi riêng.',
        '## 4.5. Lịch sử, lớp khó và lỗi theo cự ly',
        frame_table(history, f'Bảng 11. Có và chưa có lịch sử, minh họa seed {seed}'),
        frame_table(per_class, f'Bảng 12. Kết quả từng lớp của mô hình được chọn ở seed {seed}'),
        '![Hình 4. Ma trận nhầm lẫn chuẩn hóa theo lớp thật ở seed đầu.](confusion.png)',
        frame_table(distances, f'Bảng 13. Phân tích theo cự ly trên toàn test của seed {seed}'),
        'Từng nhóm cự ly có phân bố lớp và số mẫu khác nhau. Chênh lệch Macro-F1 giữa các nhóm không trực tiếp chứng minh quan hệ nhân quả với khoảng cách. Nhóm Other có nhiều nguyên nhân khác nhau, nên các số đo theo cause trong error_profiles.csv cần được dùng để hiểu tính không đồng nhất của nhãn.',
        '![Hình 5. Permutation importance sau khi khóa lựa chọn.](importance.png)',
        'Permutation importance xáo trộn từng thuộc tính trong mẫu test 10000 dòng đã chọn cố định và lặp ba lần. Giá trị âm hoặc nhỏ có thể do nhiễu và tương quan giữa các biến. Phép phân tích này phục vụ diễn giải và không được dùng để đổi tập đặc trưng rồi chọn lại trên cùng test.',
        '# 5. THẢO LUẬN, HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN',
        'Khả năng nhận diện lớp ít mẫu cần được đánh giá bằng Macro-F1, Recall và Precision từng lớp. Accuracy của Majority có thể cao khi AR chiếm phần lớn dữ liệu. Cân bằng trọng số có thể tăng ưu tiên các nhóm hiếm nhưng tạo thêm dự đoán dương sai. Đối chiếu XGBoost weight 0 và 1 cùng chiều sâu, seed và dữ liệu giúp khảo sát sự đánh đổi cụ thể.',
        'Lịch sử yêu cầu biết nhãn vũ khí của các sự kiện trước. Khi ứng dụng chỉ có nhãn dự đoán, cần đánh giá tích lũy sai số qua chuỗi. Các trận được chia rời nhau nhưng người chơi có thể xuất hiện ở nhiều trận, nên khả năng tổng quát hóa theo người chơi cần một thiết kế riêng.',
        'Các phân vị khoảng cách dùng histogram có sai số trong độ rộng bin đã công bố. Một cấu hình không cải thiện trong số epoch cho phép có thể cần cách tối ưu khác. Ba seed giúp quan sát độ nhạy nhưng chưa đủ đại diện mọi ngẫu nhiên. Thử nghiệm hiện đại trên bộ giới hạn không chứng minh thứ hạng tương tự trên hàng chục triệu dòng.',
        'Nguồn dữ liệu ghi lại một giai đoạn trò chơi và hai bản đồ hợp lệ. Thay đổi phiên bản, vật phẩm, chế độ chơi hoặc bản đồ có thể đổi phân phối. Nghiên cứu tiếp theo có thể đánh giá theo thời gian, tách LMG, điều chỉnh trọng số bằng dữ liệu phát triển mới, và đối chiếu thêm chiến lược thiếu lịch sử.',
        '# 6. KIẾN TRÚC, TÀI NGUYÊN VÀ TÁI LẬP',
        frame_table(worker, 'Bảng 14. So sánh đọc và mã hóa cùng phân vùng với một và hai worker'),
        'Bảng 14 dùng cùng bốn phân vùng train và kiểm tra hash đầu ra để xác nhận hai cách cho kết quả giống nhau. Thứ tự chạy là một worker rồi hai worker, nên cache của hệ điều hành có thể ảnh hưởng. Đây là phép đo đọc/mã hóa và không được diễn giải thành tốc độ của toàn bộ bước làm sạch hoặc tạo lịch sử.',
        frame_table(resource_frame, 'Bảng 15. Thời gian thực và RAM theo công đoạn ở ba seed'),
        'RAM đo tổng RSS của tiến trình chính và con với chu kỳ 0,25 giây. Tổng này có thể đếm lặp trang nhớ chung. VRAM cấp phát qua PyTorch được ghi riêng trong resources.json. Các phép đo CPU, GPU và đĩa có phạm vi khác nhau. Nút thắt cần xác định từ thời gian từng công đoạn thay vì suy ra từ số luồng được cấu hình.',
        'Notebook công khai tải đúng commit của mã nguồn, kiểm tra hai mô hình mới trước khi quét CSV rồi lần lượt chạy xử lý, thống kê, ba seed toàn dữ liệu, ba seed bộ so sánh và xuất báo cáo. Một lỗi ở phương pháp bắt buộc làm status.json nhận failed. Chỉ khi toàn bộ bước và bộ sản phẩm hoàn tất, trạng thái completed mới được ghi.',
        'Gắn bộ dữ liệu skihikingkevin/pubg-match-deaths vào notebook, bật Internet và GPU T4 hoặc P100. Cài requirements.txt và requirements-modern.txt. Chạy python -m pubg_weapon.study với danh sách năm CSV và thư mục output mới. Gói đầu ra giữ cấu hình, commit, môi trường, nguồn chia tập, bộ so sánh, model và checksum các sản phẩm.',
        '# 7. KẾT LUẬN',
        f'Thực nghiệm xử lý đầy đủ {manifest["source_rows"]:,} dòng và học trên {manifest["splits"]["train"]["rows"]:,} sự kiện train hợp lệ. Ba seed dùng chung dữ liệu để đo biến thiên do mô hình. Kết quả toàn dữ liệu và bộ so sánh hai phương pháp gần đây được báo cáo theo cùng nguyên tắc chọn trên validation. Họ được chọn lần lượt là {winner} và {modern_winner}.',
        'Các kết quả chỉ có ý nghĩa trong phạm vi dữ liệu, nhãn, đặc trưng và tài nguyên đã công bố. Phân tích theo lịch sử, từng lớp và cự ly chỉ ra trường hợp khó và giúp định hướng cải thiện. Toàn bộ nguồn, notebook, báo cáo, slide và dữ liệu đối chiếu đi kèm cho phép kiểm tra lại quy trình.',
        '# TÀI LIỆU THAM KHẢO',
        '[1] Y. Gorishniy, A. Kotelnikov và A. Babenko. TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. ICLR, 2025. https://github.com/yandex-research/tabm',
        '[2] J. Qu, D. Holzmüller, G. Varoquaux và M. Le Morvan. TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model. ICML, 2026. https://proceedings.mlr.press/v306/qu26a.html',
        '[3] T. Chen và C. Guestrin. XGBoost: A Scalable Tree Boosting System. KDD, 2016. https://doi.org/10.1145/2939672.2939785',
        '[4] skihikingkevin. PUBG Match Deaths and Statistics. https://www.kaggle.com/datasets/skihikingkevin/pubg-match-deaths',
        '[5] F. Pedregosa và cộng sự. Scikit-learn: Machine Learning in Python. JMLR 12, 2011. https://jmlr.org/papers/v12/pedregosa11a.html',
    ]
    markdown = '\n\n'.join(sections)+'\n'
    (out / 'bao_cao_khai_pha_du_lieu.md').write_text(markdown, encoding='utf-8')
    latex_source(markdown, out)
    render_pdf(markdown, out / 'bao_cao_khai_pha_du_lieu.pdf', out)
    build_slides(out, manifest, compact_full, compact_modern, abl_summary, winner, modern_winner)
    write_json(out / 'requirements_check.json', {
        'large_real_dataset': manifest['source_rows'] > 1_000_000, 'modern_models_completed': all(name in set(modern.name) for name in ['tabm', 'tabiclv2']),
        'three_seeds_full': set(full.seed) == set(config['seeds']), 'three_seeds_modern': set(modern.seed) == set(config['seeds']),
        'report_pdf': (out / 'bao_cao_khai_pha_du_lieu.pdf').is_file(), 'latex_source': (out / 'main.tex').is_file(),
        'presentation': (out / 'Thuyet_trinh_PUBG.pptx').is_file(), 'font_points': FONT_SIZE,
        'oral_defense': 'Requires the group to explain the actual results. Automated checks cannot assign a grade.'})


def build_slides(out, manifest, full, modern, ablation, winner, modern_winner):
    from pptx import Presentation
    from pptx.util import Inches, Pt
    from pptx.dml.color import RGBColor
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    def slide(title, paragraphs, image=None):
        page = deck.slides.add_slide(deck.slide_layouts[6])
        box = page.shapes.add_textbox(Inches(.6), Inches(.35), Inches(12.1), Inches(.7))
        p = box.text_frame.paragraphs[0]
        p.text, p.font.size, p.font.bold = title, Pt(27), True
        p.font.color.rgb = RGBColor.from_string('173D58')
        text = page.shapes.add_textbox(Inches(.7), Inches(1.3), Inches(5.4 if image else 11.8), Inches(5.5)).text_frame
        text.word_wrap = True
        for i, paragraph in enumerate(paragraphs):
            p = text.paragraphs[0] if i == 0 else text.add_paragraph()
            p.text, p.font.size = paragraph, Pt(20)
            p.space_after = Pt(17)
        if image:
            page.shapes.add_picture(str(out / image), Inches(6.35), Inches(1.5), width=Inches(6.3))
        notes = page.notes_slide.notes_text_frame
        notes.text = 'Đối chiếu số liệu chi tiết và giới hạn của kết luận với phần tương ứng trong báo cáo. Các bảng điểm dùng lựa chọn từ validation.'
    slide('Phân loại nhóm vũ khí trong PUBG', ['Đồ án Khai phá dữ liệu - Nhóm 3', 'Ngữ cảnh sự kiện và lịch sử trước thời điểm hiện tại', REPO])
    slide('Bài toán và câu hỏi nghiên cứu', ['Dự đoán một trong sáu nhóm vũ khí từ dữ liệu sự kiện.', 'Đo đóng góp của khoảng cách, ngữ cảnh và lịch sử.', 'So sánh mô hình cổ điển, tổ hợp cây và hai phương pháp gần đây.'])
    slide('Dữ liệu thực tế', [f'{manifest["source_rows"]:,} dòng nguồn từ năm CSV.', f'{manifest["available_matches"]:,} mã trận ở đầu vào.', f'{manifest["splits"]["train"]["rows"]:,} sự kiện train hợp lệ.'], 'classes.png')
    slide('Làm sạch và ngăn rò rỉ dữ liệu', ['Kiểm kê thiếu, giá trị lỗi và miền tọa độ/thời gian.', 'Mỗi trận thuộc đúng một tập.', 'Nhãn hiện tại và thứ hạng cuối trận được loại khỏi đầu vào.', 'Mọi thống kê học chỉ từ train.'])
    slide('Đặc trưng và lịch sử', ['A dùng khoảng cách, B thêm thời gian.', 'C có 11 thuộc tính ngữ cảnh.', 'D có 17 thuộc tính sau khi thêm lịch sử.', 'Sự kiện đồng thời không nhìn nhãn của nhau.'])
    slide('Các mô hình đối chứng', ['Majority và Gaussian Naive Bayes.', 'Logistic SGD học qua từng lô.', 'Random Forest và HistGradientBoosting trong bộ so sánh chung.', 'XGBoost Hist dùng bộ nhớ ngoài trên toàn train.'])
    slide('Hai phương pháp gần đây', ['TabM (ICLR 2025) tổ hợp MLP có chia sẻ tham số.', 'TabICLv2 (ICML 2026) suy luận từ ngữ cảnh có nhãn.', 'Cùng hàng, nhãn và tập đánh giá với các đối chứng.', 'Chi phí tiền huấn luyện bên ngoài PUBG được công bố riêng.'])
    slide('Thiết kế thực nghiệm', ['Ba seed mô hình, một cách chia dữ liệu cố định.', 'Chọn cấu hình và checkpoint bằng validation.', 'Macro-F1 là chỉ số chính.', 'Bootstrap theo trận và chênh lệch ghép cặp.'])
    slide('Kết quả trên toàn dữ liệu', [f'Họ được chọn theo validation trung bình - {winner}.',
          *[f'{r.family}: Macro-F1 {r.macro_f1_mean:.4f}, SD {r.macro_f1_std:.4f}' for r in full.itertuples()]])
    slide('Kết quả phương pháp hiện đại', [f'Họ được chọn theo validation trung bình - {modern_winner}.',
          'Các thanh sai số là độ lệch chuẩn qua seed.', 'Đọc kết quả cùng thời gian và bộ nhớ.'], 'modern_comparison.png')
    slide('Đường học và chọn checkpoint', ['Theo dõi validation qua epoch và vòng boosting.', 'Các đường được lưu cho cả ba seed.', 'Chọn checkpoint trước khi đánh giá test.'], 'learning_curves.png')
    slide('Đóng góp của các khối đặc trưng', [f'{r.ablation}: Macro-F1 {r.macro_f1_mean:.4f}, SD {r.macro_f1_std:.4f}' for r in ablation.itertuples()])
    slide('Phân tích lỗi', ['Kết quả theo lớp, nhóm lịch sử và khoảng cách.', 'Ma trận nhầm lẫn minh họa seed đầu.', 'Các lớp có ngữ cảnh chồng lấn cần thông tin bổ sung.'], 'confusion.png')
    slide('Giải thích và sử dụng tài nguyên', ['Phân vùng theo trận và đọc Parquet theo lô.', 'Đo thời gian, RAM, VRAM và dung lượng đĩa.', 'So sánh một và hai worker trên cùng công việc.', 'Permutation importance phục vụ diễn giải sau khóa lựa chọn.'], 'importance.png')
    slide('Kết luận và hướng phát triển', ['Lưu đầy đủ nguồn, notebook và bằng chứng tái lập.', 'Giới hạn do nhãn tổng hợp, dữ liệu lịch sử và thông tin chưa quan sát.', 'Hướng phát triển gồm đánh giá theo thời gian và lịch sử dùng nhãn dự đoán.', REPO])
    slide('Tài liệu tham khảo', ['TabM - ICLR 2025, Yandex Research.', 'TabICLv2 - ICML 2026, Qu và cộng sự.', 'XGBoost - KDD 2016, Chen và Guestrin.', 'PUBG Match Deaths and Statistics - Kaggle.'])
    deck.save(out / 'Thuyet_trinh_PUBG.pptx')
