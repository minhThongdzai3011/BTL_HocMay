import re
import io
import pandas as pd
import matplotlib.pyplot as plt

# 1. Hàm làm sạch LaTeX và math expressions
def clean_latex(text: str) -> str:
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.DOTALL)
    text = re.sub(r'\$.*?\$',    '', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-zA-Z]+',          '', text)
    return text

# 2. Đọc file CSV và làm sạch
file_path = r'E:\Hoc May\BTL_HocMay\arxiv_data.csv'
with open(file_path, 'r', encoding='utf-8') as f:
    raw = f.read().lstrip('\ufeff')
cleaned = clean_latex(raw)
df_all = pd.read_csv(io.StringIO(cleaned),
                     engine='python',
                     encoding='utf-8',
                     on_bad_lines='skip')

# 3. Chia dataset và gán nhãn
mid = len(df_all) // 2
df = df_all.iloc[:mid].reset_index(drop=True)

def assign_label(text: str) -> str:
    keywords = {
        'robot': 'robotics',
        'navigation': 'robotics',
        'depth': 'vision',
        'stereo': 'vision',
        'image': 'vision',
        'text': 'NLP',
        'classification': 'NLP',
        'language': 'NLP'
    }
    s = str(text).lower()
    for kw, lbl in keywords.items():
        if kw in s:
            return lbl
    return None

df['label'] = df['summaries'].apply(assign_label)

# 4. Tính count và percent
counts   = df['label'].value_counts()
percents = df['label'].value_counts(normalize=True).mul(100).round(2)

stats = pd.DataFrame({'count': counts, 'percent': percents})

# 5. Vẽ biểu đồ

# Bar chart: count
plt.figure(figsize=(8, 5))
stats['count'].plot(kind='bar',
                    color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'],
                    edgecolor='black')
plt.title('Số lượng bài báo theo nhãn', fontsize=14)
plt.ylabel('Số bài báo', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Bar chart: percent
plt.figure(figsize=(8, 5))
stats['percent'].plot(kind='bar',
                      color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'],
                      edgecolor='black')
plt.title('Tỷ lệ phần trăm bài báo theo nhãn', fontsize=14)
plt.ylabel('Phần trăm (%)', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Pie chart: percent
plt.figure(figsize=(6, 6))
stats['percent'].plot(kind='pie',
                      colors=['#4c72b0', '#55a868', '#c44e52', '#8172b2'],
                      autopct='%1.1f%%',
                      startangle=90,
                      counterclock=False)
plt.title('Cơ cấu nhãn bài báo', fontsize=14)
plt.ylabel('')
plt.tight_layout()
plt.show()