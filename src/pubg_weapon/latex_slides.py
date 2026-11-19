"""Create a timed Beamer deck from completed PUBG results."""
import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


NAMES = {
    "majority": "Majority", "naive_bayes": "Gaussian NB", "logistic": "Logistic",
    "random_forest": "Random Forest", "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost", "tabm": "TabM", "tabiclv2": "TabICLv2",
}
TIMES = [20, 70, 55, 35, 30, 45, 40, 45, 45, 35, 60, 60, 45, 40, 35, 75]
PREAMBLE = r"""
\documentclass[aspectratio=169,10pt]{beamer}
\usepackage{fontspec,amsmath,booktabs,array,graphicx,hyperref}
\IfFontExistsTF{Arial}{\setsansfont{Arial}}{\setsansfont{Liberation Sans}}
\setmonofont{Latin Modern Mono}[Scale=.85]
\definecolor{CourseBlue}{HTML}{1458B5}
\definecolor{HeaderGray}{HTML}{F0F0F0}
\definecolor{FooterGray}{HTML}{E8EDF4}
\definecolor{Muted}{HTML}{5E6772}
\setbeamercolor{normal text}{fg=black,bg=white}
\setbeamercolor{frametitle}{fg=CourseBlue,bg=HeaderGray}
\setbeamerfont{frametitle}{size=\fontsize{14}{16}\selectfont}
\setbeamercolor{structure}{fg=CourseBlue}
\setbeamercolor{item}{fg=CourseBlue}
\setbeamertemplate{itemize item}{\raise.2ex\hbox{\tiny$\bullet$}}
\setbeamertemplate{navigation symbols}{}
\setbeamersize{text margin left=7mm,text margin right=7mm}
\setbeamerfont{footline}{size=\fontsize{7}{8}\selectfont}
\setbeamercolor{footer left}{fg=white,bg=CourseBlue}
\setbeamercolor{footer center}{fg=CourseBlue,bg=FooterGray}
\setbeamercolor{footer right}{fg=CourseBlue,bg=FooterGray}
\setbeamertemplate{footline}{
 \leavevmode\hbox{
 \begin{beamercolorbox}[wd=.12\paperwidth,ht=2.5ex,dp=.7ex,center]{footer left}Nhóm 3\end{beamercolorbox}
 \begin{beamercolorbox}[wd=.76\paperwidth,ht=2.5ex,dp=.7ex,leftskip=3mm]{footer center}Khai phá dữ liệu\end{beamercolorbox}
 \begin{beamercolorbox}[wd=.12\paperwidth,ht=2.5ex,dp=.7ex,center]{footer right}\insertframenumber{} / 16\end{beamercolorbox}}}
\setbeamerfont{note page}{size=\small}
\setbeameroption{hide notes}
\newcommand{\nguon}[1]{\vfill{\fontsize{6.5}{8}\selectfont\color{Muted}#1}}
\newcommand{\nhanmanh}[1]{\textcolor{CourseBlue}{\textbf{#1}}}
\renewcommand{\arraystretch}{1.15}
\begin{document}
"""


def tex(value):
    changes = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
               "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(changes.get(c, c) for c in str(value))


def decimal(value):
    return f"{float(value):.4f}".replace(".", ",")


def number(value):
    return f"{int(value):,}".replace(",", ".")


def result_table(frame, name_column):
    rows = [r"\begin{tabular}{lrrr}", r"\toprule",
            r"Phương pháp & Macro-F1 & SD & Accuracy\\", r"\midrule"]
    for _, row in frame.iterrows():
        name = NAMES.get(row[name_column], row[name_column])
        rows.append(f"{tex(name)} & {decimal(row['macro_f1_mean'])} & "
                    f"{decimal(row['macro_f1_std'])} & {decimal(row['accuracy_mean'])}" + r"\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def bullets(*items):
    return "\n".join([r"\begin{itemize}", *(r"\item " + item for item in items), r"\end{itemize}"])


def build(root, out):
    out.mkdir(parents=True, exist_ok=True)
    saved = root / "deliverables"
    manifest = json.loads((root / "prepared" / "dataset_manifest.json").read_text(encoding="utf-8"))
    full = pd.read_csv(saved / "full_summary.csv")
    modern = pd.read_csv(saved / "modern_summary.csv")
    config = json.loads((root / "study_config.json").read_text(encoding="utf-8"))
    first_seed = config["seeds"][0]
    ablation = pd.concat([pd.read_csv(root / "full_models" / str(seed) / "test" / "ablation.csv")
                          for seed in config["seeds"]])
    ablation = ablation.groupby("ablation").macro_f1.agg(["mean", "std"]).reset_index()
    models = pd.read_csv(root / "full_test_selected_by_family.csv")
    winner = models.groupby("family").validation_macro_f1.mean().idxmax()
    split_rows = []
    for split, values in manifest["splits"].items():
        split_rows.append(f"{tex(split)} & {number(values['rows'])} & {number(values['matches'])}" + r"\\")
    clean = pd.read_csv(root / "prepared" / "cleaning_counts.csv")
    removed = int(clean.removed.sum())
    fraction = 100 * removed / manifest["source_rows"]
    for image in ("classes.png", "confusion.png", "importance.png", "modern_comparison.png", "learning_curves.png"):
        shutil.copy2(saved / image, out / image)
    slides, notes = [], []

    def add(title, content, note, source=""):
        slides.append(r"\begin{frame}{" + tex(title) + "}\n" + content +
                      (r"\nguon{" + tex(source) + "}" if source else "") +
                      "\n" + r"\note{" + tex(note) + "}\n" + r"\end{frame}" + "\n")
        notes.append((title, note))

    title = (
        r"{\fontsize{8}{10}\selectfont ĐẠI HỌC PHENIKAA\\TRƯỜNG CÔNG NGHỆ THÔNG TIN PHENIKAA}\par"
        r"\vspace{3mm}{\fontsize{18}{22}\selectfont\bfseries Phân loại nhóm vũ khí trong PUBG\\"
        r"từ ngữ cảnh và lịch sử sự kiện}\par\vspace{3mm}"
        r"{\fontsize{9}{11}\selectfont Giảng viên hướng dẫn - ThS. Phạm Kim Thành\\"
        r"Nhóm 3\\Kiều Thị Thu Trang \quad Trần Minh Sang \quad Ngô Quang Thiện}\par\vspace{3mm}"
        r"{\fontsize{8}{10}\selectfont\url{https://github.com/kttt294/PUBG-Weapon-Groups}}"
    )
    add("Bài tập lớn môn Khai phá dữ liệu", title,
        "Giới thiệu đề tài, nhóm và mục tiêu khai thác nhật ký PUBG. Chuyển ngay sang câu hỏi nghiên cứu, tránh đọc lại trang bìa.")
    add("1.1. Ngữ cảnh sự kiện cho biết gì về nhóm vũ khí?",
        r"\textbf{Bài toán}\par Dự đoán một trong sáu nhóm vũ khí khi đã quan sát được ngữ cảnh sự kiện.\par\vspace{2mm}"
        + bullets("Đo phần đóng góp của khoảng cách, thời gian, vị trí và lịch sử.",
                  "So sánh mô hình cổ điển, tổ hợp cây và hai phương pháp gần đây.",
                  "Đánh giá chất lượng cùng chi phí xử lý dữ liệu lớn.")
        + r"\vspace{2mm}\nhanmanh{Nhãn của sự kiện hiện tại được giữ ngoài đầu vào mô hình.}",
        "Nêu một ví dụ giao tranh gần có thể dùng Shotgun, SMG hoặc AR. Vì vậy khoảng cách có tín hiệu nhưng chưa đủ. Phân biệt dự đoán nhãn sự kiện đã ghi nhận với dự báo trước giao tranh.")
    add("2.1. Dữ liệu lớn và mất cân bằng giữa các nhóm",
        r"\begin{columns}[T,onlytextwidth]\column{.45\textwidth}"
        + bullets(f"{number(manifest['source_rows'])} dòng từ 5 CSV.",
                  f"{number(manifest['available_matches'])} mã trận đầu vào.",
                  "Định danh, bản đồ, thời gian, tọa độ và nguyên nhân tử vong.")
        + r"\column{.52\textwidth}\includegraphics[width=\linewidth]{classes.png}\end{columns}",
        "Giải thích một dòng là một sự kiện. Chỉ ra lớp AR phổ biến và tác động của mất cân bằng lên Accuracy. Nêu tập nguyên nhân hợp lệ và lý do cần nhóm Other.",
        "Nguồn dữ liệu - PUBG Match Deaths and Statistics, Kaggle")
    add("2.2. Kiểm kê chất lượng trước khi huấn luyện",
        bullets("Đếm dữ liệu thiếu, chuỗi rỗng, giá trị số lỗi và vô cực theo từng cột.",
                "Kiểm tra định danh, tọa độ, thời điểm và sự kiện trùng.",
                "Ánh xạ nhãn tường minh, gộp lớp dưới 1% dựa trên train.")
        + f"\n\\vspace{{3mm}}\\nhanmanh{{Loại {number(removed)} dòng, tương đương {fraction:.2f}\\% đầu vào.}}",
        "Trình bày quy tắc lọc theo thứ tự, mỗi dòng bị tính vào bước đầu tiên loại nó. Giải thích Down and Out không xác định vũ khí ban đầu. Bảng đếm chi tiết có trong báo cáo.")
    add("2.3. Chia theo trận để giữ phép đánh giá nhất quán",
        r"\begin{columns}[T,onlytextwidth]\column{.49\textwidth}"
        + "\n".join([r"\begin{tabular}{lrr}\toprule", r"Tập & Sự kiện & Trận\\\midrule", *split_rows, r"\bottomrule\end{tabular}"])
        + r"\column{.47\textwidth}"
        + bullets("Cùng trận nằm trong cùng tập.", "Train học thống kê và trọng số.",
                  "Validation chọn cấu hình, test báo cáo kết quả.")
        + r"\end{columns}\vspace{3mm}{\small Tập holdout được giữ cố định từ lượt trước và đánh giá lại theo giao thức đã định.}",
        "Nêu vì sao chia ngẫu nhiên từng dòng có thể đưa cùng trận sang nhiều tập. Công khai việc holdout đã được dùng ở lượt trước và mọi lựa chọn ở lượt này vẫn dựa trên validation.")
    add("3.1. Từ khoảng cách đến lịch sử đúng thứ tự",
        r"\begin{tabular}{ll}\toprule Mức & Thông tin\\\midrule"
        r"A & Khoảng cách\\ B & A và thời điểm\\"
        r"C & B và log khoảng cách, bản đồ, giai đoạn, vị trí\\"
        r"D & C và sáu thuộc tính lịch sử\\\bottomrule\end{tabular}"
        + r"\vspace{3mm}\[d=\frac{\sqrt{(x_k-x_v)^2+(y_k-y_v)^2}}{100},\qquad t_{\mathrm{trước}}<t_{\mathrm{hiện\ tại}}.\]"
        + r"\nhanmanh{Các sự kiện đồng thời không nhìn thấy nhãn của nhau.}",
        "Giải thích điều kiện thời gian nghiêm ngặt. Sáu biến lịch sử thể hiện có lịch sử, số lần trước, nhóm gần nhất, khoảng cách gần nhất, thời gian từ lần trước và nhóm phổ biến trước đó. Lịch sử giả định nhãn của sự kiện trước đã quan sát.")
    add("3.2. Các mô hình đối chứng và XGBoost",
        bullets("Majority làm mốc phân bố nhãn. Gaussian NB dùng xác suất theo lớp.",
                "Logistic SGD cập nhật theo lô và điều chuẩn L2.",
                "XGBoost bổ sung cây để giảm hàm mất mát, dùng histogram tìm ngưỡng chia.")
        + r"\vspace{2mm}\nhanmanh{Bộ nhớ ngoài cho phép XGBoost học từ mọi dòng train hợp lệ.}",
        "Giải thích sự khác nhau giữa mô hình tuyến tính và cây phi tuyến. XGBoost dùng gradient và đạo hàm bậc hai để tìm cập nhật. Đĩa chứa các trang đệm để tránh nạp cả ma trận vào RAM.",
        "Chen và Guestrin, KDD 2016")
    add("3.3. TabM tổ hợp nhiều MLP với chia sẻ tham số",
        bullets("16 thành viên cùng học trên mỗi lô dữ liệu.",
                "Các thành viên chia sẻ phần lớn trọng số và có hệ số điều biến riêng.",
                "Chọn checkpoint bằng Macro-F1 trên validation.")
        + r"\[\widehat{p}(y=c\mid x)=\frac{1}{K}\sum_{k=1}^{K}\operatorname{softmax}(f_k(x))_c.\]"
        + r"\nhanmanh{Lấy trung bình xác suất của các thành viên khi suy luận.}",
        "Mô tả trực quan nhiều bộ dự đoán cùng khai thác một phần biểu diễn chung. Phân biệt tổ hợp chia sẻ tham số với huấn luyện 16 mạng độc lập. Cấu hình trong nghiên cứu dùng 3 khối, rộng 256 và không có feature embeddings.",
        "Gorishniy, Kotelnikov và Babenko, TabM, ICLR 2025")
    add("3.4. TabICLv2 suy luận từ ngữ cảnh có nhãn",
        bullets("Mô hình đã huấn luyện trước nhận bảng train làm ngữ cảnh.",
                "Dự đoán nhãn của truy vấn qua một lượt suy luận.",
                "4 biến đổi tổ hợp, representation cache và CPU offloading.")
        + r"\vspace{3mm}\nhanmanh{Thời gian fit trên PUBG không gồm chi phí tiền huấn luyện.}",
        "Giải thích in-context learning bằng ví dụ mô hình nhận các hàng có nhãn và một hàng truy vấn. TabICLv2 không học các epoch gradient từ PUBG như TabM. Nêu nhu cầu bộ nhớ ngữ cảnh và lý do so sánh trên cùng các hàng có giới hạn kích thước.",
        "Qu và cộng sự, TabICLv2, ICML 2026")
    add("3.5. Ba seed và hai quy mô thí nghiệm",
        bullets("Toàn train: mọi dòng hợp lệ, cùng một cách chia dữ liệu.",
                "Bộ so sánh phương pháp mới: tất cả họ dùng cùng hàng và phân bố lớp.",
                "Seed mô hình 42, 2026, 7. Tham số và epoch chọn bằng validation.",
                "Macro-F1 là chỉ số chính, kèm Accuracy, từng lớp và chi phí.")
        + r"\vspace{2mm}{\small SD ba seed mô tả biến thiên mô hình. Khoảng tin cậy tính riêng bằng bootstrap theo trận.}",
        "Làm rõ lý do không đặt điểm của hai quy mô vào một bảng xếp hạng chung. Trong từng bảng, cách chia và dữ liệu của các mô hình tương ứng là chung. Bootstrap lấy lại nguyên trận để giữ phụ thuộc giữa các sự kiện.")
    add("4.1. Kết quả trên toàn bộ dữ liệu huấn luyện",
        r"\begin{center}" + result_table(full, "family") + r"\end{center}\vspace{3mm}"
        + r"\nhanmanh{Họ được chọn theo validation trung bình - " + tex(winner) + "}",
        "Đọc hai chênh lệch quan trọng trong bảng dựa trên số liệu thật, sau đó giải thích sự đánh đổi giữa Macro-F1 và Accuracy. Không dùng Accuracy riêng để kết luận khả năng nhận diện lớp ít mẫu.")
    add("4.2. TabM và TabICLv2 so với các đối chứng",
        r"\begin{center}{\small " + result_table(modern, "name") + r"}\end{center}",
        "Nêu thứ hạng và độ lệch chuẩn của hai phương pháp mới, rồi đối chiếu với XGBoost và Random Forest trên cùng dữ liệu. Chỉ khẳng định chênh lệch rõ khi bảng bootstrap trong báo cáo hỗ trợ. Giải thích nếu phương pháp mới chưa tốt hơn đối chứng.",
        "Cùng các trận và hàng trong comparison_cohorts")
    add("4.3. Nhóm đặc trưng nào cải thiện dự đoán?",
        r"\begin{center}\begin{tabular}{lrr}\toprule Mức & Macro-F1 trung bình & SD\\\midrule"
        + "\n".join(f"{r.ablation} & {decimal(r.mean)} & {decimal(r.std)}" + r"\\" for r in ablation.itertuples())
        + r"\bottomrule\end{tabular}\end{center}\vspace{3mm}"
        + r"\nhanmanh{Đọc tác động của lịch sử cùng kết quả nhóm chưa có lịch sử.}",
        "Nêu mức tăng từ C lên D bằng kết quả thật. Nhắc rằng thứ tự thêm đặc trưng ảnh hưởng chênh lệch. Lịch sử chỉ có tín hiệu khi tồn tại, nên báo cáo chia riêng nhóm đã có và chưa có sự kiện trước.")
    add("4.4. Ma trận nhầm lẫn chỉ ra các lớp khó",
        r"\begin{columns}[T,onlytextwidth]\column{.38\textwidth}"
        + bullets("Xem Precision và Recall từng lớp.", "Đối chiếu cặp nhầm lẫn lớn.",
                  "Phân tích thêm theo cự ly và tình trạng lịch sử.")
        + r"\column{.59\textwidth}\includegraphics[width=\linewidth,height=.66\textheight,keepaspectratio]{confusion.png}\end{columns}",
        "Chọn hai ô có số đếm hoặc tỷ lệ lớn trong kết quả thật. Phân biệt Recall cao với Precision cao. Ma trận thể hiện mẫu nhầm lẫn, còn diễn giải nguyên nhân cần xem bối cảnh và mức thông tin quan sát.")
    add("4.5. Tài nguyên và khả năng tái lập",
        bullets("Đọc CSV theo lô, phân vùng theo mã trận và dùng Parquet.",
                "Đo thời gian, tổng RSS tiến trình, VRAM PyTorch và đĩa.",
                "Đối chiếu một và hai worker bằng hash đầu ra.",
                "Lưu commit, cấu hình, môi trường và dữ liệu so sánh.")
        + r"\vspace{2mm}{\small So sánh worker chịu ảnh hưởng cache và chỉ đo bước đọc, mã hóa đã chọn.}",
        "Đọc thời gian và RAM từ bảng cuối báo cáo. Chỉ rõ công đoạn chậm nhất và công đoạn có đỉnh RAM cao nhất. Tránh suy ra hiệu suất CPU chỉ từ số luồng được đặt.")
    add("5. Kết luận và hướng phát triển",
        bullets("Đã xử lý nguồn lớn, tổ chức đặc trưng theo trận và thời gian.",
                "Đối chiếu phương pháp cổ điển với TabM và TabICLv2 qua ba seed.",
                "Giới hạn từ lớp Other, lịch sử chưa có và dữ liệu trò chơi theo giai đoạn.",
                "Hướng phát triển: đánh giá theo thời gian, người chơi và chuỗi dùng nhãn dự đoán.")
        + r"\vspace{3mm}\nhanmanh{Nguồn và kết quả đi kèm để kiểm chứng từng kết luận.}",
        "Tóm tắt một phát hiện về dữ liệu, một phát hiện về mô hình và một giới hạn quan trọng. Kết thúc trong thời gian đã định, mời thầy đặt câu hỏi. Phân công thành viên trả lời theo phần thực hiện.")
    appendix = [
        ("Dự phòng. Vì sao Accuracy của Majority có thể cao?",
         bullets("AR chiếm tỷ trọng lớn. Luôn chọn AR có thể đúng nhiều dòng.",
                 "Recall của năm lớp khác bằng 0.", "Macro-F1 cho trọng số bằng nhau giữa các lớp."),
         "Viết công thức Macro-F1 và minh họa một lớp bị dự đoán bằng 0 nếu thầy yêu cầu."),
        ("Dự phòng. Vì sao nhóm vũ khí và vì sao Other?",
         bullets("Bài toán được xác định ở cấp nhóm, với ánh xạ rõ ràng.",
                 "Các vũ khí trong cùng nhóm có bối cảnh chồng lấn.",
                 "Ngưỡng gộp lớp hiếm dựa trên train. Other có tính không đồng nhất."),
         "Giải thích đây là lựa chọn định nghĩa nhãn, cần công bố khi so sánh. Tách LMG hoặc phân loại tên vũ khí là thí nghiệm mở rộng có tập nhãn khác."),
        ("Dự phòng. Khi nào lịch sử có thể làm rò rỉ nhãn?",
         bullets("Nếu dùng vũ khí của chính sự kiện hiện tại.",
                 "Nếu dùng sự kiện ở thời gian sau hoặc cùng dấu thời gian.",
                 "Nếu dùng thứ hạng cuối trận trước khi biết kết quả trận."),
         "Trỏ đến điều kiện thời gian nghiêm ngặt, hàm build_history và danh sách cột đầu vào trong mã nguồn."),
        ("Dự phòng. Vì sao mô hình mới có thể chưa tốt hơn?",
         bullets("Ưu thế trung bình trên benchmark khác không đảm bảo ưu thế trên PUBG.",
                 "Đặc trưng, nhãn, quy mô và mất cân bằng ảnh hưởng thứ hạng.",
                 "Báo cáo cấu hình thực chạy, độ bất định và phạm vi kết luận."),
         "Trả lời dựa vào bảng thật, không viện dẫn danh tiếng phương pháp để phủ định số đo. Nêu các hướng tinh chỉnh trên validation có thể kiểm tra tiếp."),
        ("Dự phòng. Tái lập từ notebook",
         r"\texttt{python -m pubg\_weapon.study}\par\vspace{3mm}"
         + bullets("Một notebook, nguồn cố định và năm CSV.", "Ba seed mô hình và cùng các trận đánh giá.",
                   "Bảng gốc, dự đoán, bootstrap và phiên bản môi trường được lưu."),
         "Mở phần notebook tương ứng nếu được hỏi. API key chỉ dùng gửi lệnh Kaggle từ máy local, không nằm trong notebook công khai."),
    ]
    main_frames = len(slides)
    slides.append(r"\appendix\setbeamertemplate{footline}{\hfill{\tiny\color{CourseBlue}Nhóm 3 -- Khai phá dữ liệu -- Phụ lục dự phòng}\hspace{5mm}\vspace{2mm}}" + "\n")
    for title, content, note in appendix:
        add(title, content, note)
    (out / "thuyet_trinh.tex").write_text(PREAMBLE + "\n".join(slides) + r"\end{document}" + "\n", encoding="utf-8")
    elapsed, script = 0, ["# Kịch bản thuyết trình PUBG", "",
                           "Mục tiêu 10-15 phút. Các slide dự phòng chỉ mở khi có câu hỏi.", ""]
    for i, (title, note) in enumerate(notes):
        seconds = TIMES[i] if i < main_frames else 0
        script += [f"## Slide {i+1}. {title}", ""]
        if seconds:
            script += [f"Từ {elapsed//60}:{elapsed%60:02d} đến {(elapsed+seconds)//60}:{(elapsed+seconds)%60:02d}.", ""]
            elapsed += seconds
        else:
            script += ["Dự phòng, ngoài thời gian trình bày chính.", ""]
        script += [note, ""]
    (out / "kich_ban_thuyet_trinh.md").write_text("\n".join(script), encoding="utf-8")
    (out / "README.md").write_text(
        "# Slide thuyết trình bằng LaTeX\n\n"
        "Mở thuyet_trinh.tex và chọn XeLaTeX. Biên dịch hai lần. Phông Arial, hoặc Liberation Sans nếu thiếu Arial.\n\n"
        "Bố cục tham khảo slide Xử lý ảnh do nhóm cung cấp, với nền trắng, tiêu đề xanh, thanh đầu xám nhạt và chân trang môn học.\n\n"
        f"Có {main_frames} slide chính, {len(appendix)} slide dự phòng. Thời lượng dự kiến {elapsed//60} phút {elapsed%60} giây.\n\n"
        "Kịch bản nằm trong kich_ban_thuyet_trinh.md. Nguồn Beamer có speaker notes ẩn khi biên dịch mặc định.\n",
        encoding="utf-8")
    print(f"Created {main_frames} main frames, {len(appendix)} backup frames, {elapsed}s speaking plan.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.study_root, args.output)
