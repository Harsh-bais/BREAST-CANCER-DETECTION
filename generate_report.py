from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Advanced Breast Cancer Prediction Model - Project Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, body)
        self.ln()

def create_report():
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Introduction
    pdf.chapter_title("1. Introduction")
    pdf.chapter_body(
        "This project aims to detect breast cancer using advanced machine learning techniques "
        "applied to a large, real-world dataset (Wisconsin Diagnostic Breast Cancer). "
        "By analyzing 30 distinct features of cell nuclei derived from digitized images of a fine needle aspirate (FNA), "
        "the model distinguishes between Malignant (cancerous) and Benign (non-cancerous) tumors "
        "with high precision."
    )

    # 2. Concepts & Algorithms
    pdf.chapter_title("2. Concepts & Algorithms")
    pdf.chapter_body(
        "We implemented two state-of-the-art algorithms:\n\n"
        "A) HistGradientBoostingClassifier:\n"
        "An advanced implementation of Gradient Boosting Trees, similar to LightGBM. "
        "It bins continuous features into integers (histograms), making it extremely fast and "
        "scalable to very large datasets (millions of samples). It builds trees sequentially, "
        "where each tree corrects the errors of the previous ones.\n\n"
        "B) Neural Network (MLPClassifier):\n"
        "A Multi-Layer Perceptron (Deep Learning) model. It consists of layers of nodes (neurons): "
        "an input layer (30 features), hidden layers (learning complex non-linear patterns), and "
        "an output layer (probability of malignancy). We standardized the inputs to ensure stable convergence."
    )

    # 3. Tools Used
    pdf.chapter_title("3. Tools Used")
    pdf.chapter_body(
        "- Python 3.11: The core programming language.\n"
        "- Scikit-Learn: For dataset loading, model training (HistGradientBoosting, MLP), and evaluation metrics.\n"
        "- Pandas & NumPy: For high-performance data manipulation.\n"
        "- Flask: A lightweight web framework to serve the model via a REST API.\n"
        "- HTML/CSS (Glassmorphism): For creating a premium, responsive, and interactive user interface."
    )

    # 4. Real-World Impact
    pdf.chapter_title("4. Real-World Impact")
    pdf.chapter_body(
        "In a real-world medical setting, this tool serves as a powerful 'AI Assistant' for "
        "oncologists and radiologists:\n\n"
        "1. Early Detection: It can identify subtle patterns in cell data that the human eye might miss, "
        "potentially catching cancer at an earlier, more treatable stage.\n"
        "2. Reducing False Negatives: By providing a second independent opinion, it helps reduce the risk "
        "of dismissing a malignant tumor as benign.\n"
        "3. Speed & Efficiency: It processes patient data instantly, allowing doctors to focus more time "
        "on patient care and treatment planning.\n\n"
        "This project demonstrates how data science directly saves lives by bridging the gap between "
        "raw data and actionable medical insights."
    )

    pdf.output("Project_Report.pdf")
    print("PDF generated: Project_Report.pdf")

if __name__ == "__main__":
    create_report()
