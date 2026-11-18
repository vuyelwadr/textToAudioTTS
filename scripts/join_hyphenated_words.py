import re
import sys
import os

def join_hyphenated_words(input_file, output_file=None):
    """
    Process a text file to join words that are hyphenated across line breaks.
    Example: "encour-\nages" becomes "encourages"
    
    Args:
        input_file (str): Path to the input text file
        output_file (str, optional): Path to the output file. If None, will modify the input file.
    """
    # If no output file specified, create a temporary file
    if output_file is None:
        output_file = input_file + '.tmp'
        overwrite = True
    else:
        overwrite = False
    
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace hyphenated words across line breaks
        # The pattern looks for a word ending with hyphen followed by a newline and another word
        processed_content = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', content)
        
        # Write the processed content to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        # If we're overwriting the original, replace it
        if overwrite:
            os.replace(output_file, input_file)
            print(f"Successfully processed {input_file}")
        else:
            print(f"Successfully wrote output to {output_file}")
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        join_hyphenated_words(input_file, output_file)
    else:
        print("Usage: python join_hyphenated_words.py input_file [output_file]")
        print("If output_file is not provided, the input file will be modified.")
