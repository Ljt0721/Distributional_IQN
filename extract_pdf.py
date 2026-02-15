
import pypdf
import os

def extract_text(pdf_path, output_file):
    print(f"--- Extracting text from {pdf_path} ---")
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            output_file.write(f"\n--- Page {i+1} ---\n")
            text = page.extract_text()
            if text:
                output_file.write(text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

if __name__ == "__main__":
    folder = r"c:\Users\csqlj\Desktop\Distributional_RL_Navigation-master\参考文章"
    files = ["IQN参考文章.pdf", "RAL收录范文.pdf"]
    output_path = r"c:\Users\csqlj\Desktop\Distributional_RL_Navigation-master\pdf_content.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for file in files:
            f.write(f"\n\n{'='*50}\nFILENAME: {file}\n{'='*50}\n\n")
            extract_text(os.path.join(folder, file), f)
            
    print("Extraction complete. Written to pdf_content.txt")
