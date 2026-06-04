import sys
import os

def summarize_file_content(filename: str):
    """
    Reads the content of a specified file and uses advanced reasoning to summarize 
    the main named NPCs, places, and important items mentioned within.

    Args:
        filename: The full path to the text file (e.g., a story chapter).
    """
    if not os.path.exists(filename):
        print(f"Error: File not found at '{filename}'")
        return

    print(f"--- Analyzing file: {filename} ---")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # In a real Goose recipe environment, this function call would typically 
    # be replaced by an internal, specialized LLM call tailored for structured extraction.
    # For demonstration, we structure the output to guide the LLM.
    
    print("\n--- Summarization Request Sent to LLM ---")
    print("Please analyze the following text and populate the three categories:")
    print("1. Main Named NPCs: List key characters and a brief summary of their role.")
    print("2. Key Places: List significant locations and their context.")
    print("3. Important Items: List crucial objects and their narrative importance.")
    print("-" * 30)
    print(content)
    print("-" * 30)
    print("\nAnalysis complete. Please provide the structured summary.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 summarize_file.py <path_to_file>")
        print("Example: python3 summarize_file.py 'chapter_one.txt'")
    else:
        file_path = sys.argv[1]
        summarize_file_content(file_path)