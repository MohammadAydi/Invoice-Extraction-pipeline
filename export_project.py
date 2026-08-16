import os

TARGET_EXTENSIONS = {'.py'}
OUTPUT_FILE = 'all_code.txt'

EXCLUDED_DIRS = {'__pycache__', '.venv', '.idea' ,'.git', "keywords", }

def should_include(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower() in TARGET_EXTENSIONS

with open(OUTPUT_FILE, 'w', encoding='utf-8') as output:
    for root, dirs, files in os.walk('.'):

        # استبعاد المجلدات المطلوبة
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

        for file in files:
            full_path = os.path.join(root, file)

            if should_include(full_path):
                output.write('=' * 60 + '\n')
                output.write(f'FILE: {os.path.abspath(full_path)}\n')
                output.write('=' * 60 + '\n')

                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        output.write(f.read())
                except Exception as e:
                    output.write(f'[ERROR READING FILE: {e}]')

                output.write('\n\n')

print(f'✔ Done. All code written to {OUTPUT_FILE}')
