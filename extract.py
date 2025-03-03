import os
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import logging
import time

# Set up logging to capture errors without flooding the console
logging.basicConfig(filename='pdf_extraction_errors.log', level=logging.ERROR,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Helper function to sort text blocks by reading order
def sort_blocks(blocks):
    """Sort blocks by vertical position first, then by horizontal position for blocks in the same line"""
    # Group blocks by lines (similar y-position)
    line_threshold = 10  # pixels within which blocks are considered on the same line
    lines = {}
    
    for block in blocks:
        y_pos = block[1]  # y-coordinate of the block
        
        # Find a line this block belongs to
        line_found = False
        for line_y in lines.keys():
            if abs(line_y - y_pos) < line_threshold:
                lines[line_y].append(block)
                line_found = True
                break
        
        # If no matching line found, create a new one
        if not line_found:
            lines[y_pos] = [block]
    
    # Sort blocks within each line by x-position (left to right)
    result = []
    for y_pos in sorted(lines.keys()):
        line_blocks = sorted(lines[y_pos], key=lambda b: b[0])  # Sort by x-coordinate
        result.extend(line_blocks)
        
    return result

# Function to extract text using PyMuPDF with batched processing
def extract_with_pymupdf_batch(pdf_path, page_range, extract_mode="text"):
    """Process a batch of pages at once for better performance"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        results = []
        
        for page_num in page_range:
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                
                if extract_mode == "blocks":
                    # Extract as blocks for better reading order
                    blocks = page.get_text("blocks")
                    
                    # Extract coordinates and text from each block
                    text_blocks = [(block[0], block[1], block[4]) for block in blocks]
                    
                    # Sort blocks by position (top to bottom, left to right)
                    sorted_blocks = sort_blocks(text_blocks)
                    
                    # Join the sorted text blocks
                    text = "\n".join([block[2] for block in sorted_blocks])
                    
                    if text and len(text.strip()) > 0:
                        results.append((page_num, text))
                else:
                    # Standard text extraction
                    text = page.get_text()
                    if text and len(text.strip()) > 0:
                        results.append((page_num, text))
                    
        doc.close()
        return results
    except Exception as e:
        logging.error(f"PyMuPDF batch error on pages {page_range}: {str(e)}")
        return []

# Function to extract text from a single page with robust error handling
def extract_with_pymupdf(pdf_path, page_num=None, extract_mode="text"):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        
        if page_num is not None:
            # Extract specific page
            if 0 <= page_num < len(doc):
                page = doc[page_num]
                
                if extract_mode == "blocks":
                    # Extract as blocks for better reading order
                    blocks = page.get_text("blocks")
                    text_blocks = [(block[0], block[1], block[4]) for block in blocks]
                    sorted_blocks = sort_blocks(text_blocks)
                    text = "\n".join([block[2] for block in sorted_blocks])
                else:
                    text = page.get_text()
                    
                if text and len(text.strip()) > 0:
                    return page_num, text
        else:
            # Extract all pages and return as list
            text_list = []
            for i, page in enumerate(doc):
                if extract_mode == "blocks":
                    blocks = page.get_text("blocks")
                    text_blocks = [(block[0], block[1], block[4]) for block in blocks]
                    sorted_blocks = sort_blocks(text_blocks)
                    text = "\n".join([block[2] for block in sorted_blocks])
                else:
                    text = page.get_text()
                    
                if text and len(text.strip()) > 0:
                    text_list.append((i, text))
            return text_list
            
        doc.close()
    except Exception as e:
        logging.error(f"PyMuPDF error on page {page_num if page_num is not None else 'all'}: {str(e)}")
        return page_num, None if page_num is not None else []
    return page_num, None if page_num is not None else []

# Function to extract text from a single page with pypdf (as fallback)
def extract_page_text(page_data):
    page_num, page = page_data
    try:
        text = page.extract_text()
        if text and len(text.strip()) > 0:
            return page_num, text
    except Exception as e:
        # Log the error but don't crash
        logging.error(f"Error extracting page {page_num}: {str(e)}")
    return page_num, None

# Function to perform OCR on a single image with error handling
def process_image_ocr(image_data):
    try:
        import pytesseract
        page_num, image = image_data
        text = pytesseract.image_to_string(image)
        return page_num, text
    except Exception as e:
        logging.error(f"OCR error on page {image_data[0]}: {str(e)}")
        return image_data[0], None

# Function to convert PDF to images using PyMuPDF (faster than pdf2image)
def convert_pdf_to_images(pdf_path, dpi=300):
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Set the matrix for higher resolution (dpi)
            zoom = dpi / 72  # Default is 72 dpi
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image format (compatible with pytesseract)
            from PIL import Image
            import io
            imgdata = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(imgdata))
            images.append(img)
        
        return images
    except Exception as e:
        logging.error(f"Error converting PDF to images with PyMuPDF: {str(e)}")
        return None

# Helper function to create batches
def create_batches(total, batch_size=10):
    """Split work into batches for more efficient processing"""
    return [range(i, min(i + batch_size, total)) for i in range(0, total, batch_size)]

# Process a batch of pages with pypdf (must be defined at module level for ProcessPoolExecutor)
def process_batch_pypdf(pdf_path, batch_range):
    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path, strict=False)
        batch_results = {}
        for i in batch_range:
            try:
                if i < len(reader.pages):
                    text = reader.pages[i].extract_text()
                    if text and len(text.strip()) > 0:
                        batch_results[i] = text
            except Exception as e:
                logging.error(f"Error extracting page {i}: {str(e)}")
        return batch_results
    except Exception as e:
        logging.error(f"Error in batch {batch_range}: {str(e)}")
        return {}

def extract_text_from_pdf(pdf_path, txt_path, max_workers=None, batch_size=10, timeout=300, extract_mode="blocks"):
    # If max_workers is not specified, use CPU count
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    
    print(f"Using {max_workers} worker processes for extraction with batch size {batch_size}")
    
    # Step 1: Try PyMuPDF first (fastest method) with batched processing
    try:
        import fitz  # Check if PyMuPDF is available
        print(f"Using PyMuPDF as primary extraction method (mode: {extract_mode})")
        
        start_time = time.time()
        # Open the document just to get the page count
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()  # Close it immediately to free resources
        
        print(f"Processing {total_pages} pages with PyMuPDF in batches...")
        
        # Create batches of pages for more efficient processing
        batches = create_batches(total_pages, batch_size)
        extracted_text = []
        
        # Use ProcessPoolExecutor for true parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch jobs
            future_to_batch = {
                executor.submit(extract_with_pymupdf_batch, pdf_path, batch, extract_mode): batch 
                for batch in batches
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                             total=len(future_to_batch), desc="PyMuPDF Extraction"):
                batch = future_to_batch[future]
                try:
                    # Each future returns a list of (page_num, text) tuples
                    results = future.result(timeout=timeout)
                    extracted_text.extend(results)
                except concurrent.futures.TimeoutError:
                    print(f"Batch {batch} processing timed out after {timeout} seconds")
                    logging.error(f"Timeout processing batch {batch}")
                except Exception as e:
                    print(f"Error processing batch {batch}: {str(e)}")
                    logging.error(f"Error in PyMuPDF batch {batch}: {str(e)}")
        
        elapsed = time.time() - start_time
        print(f"PyMuPDF extraction completed in {elapsed:.2f} seconds")
        
        # If PyMuPDF extracted a reasonable amount of text, write it to file
        if extracted_text and len(extracted_text) > total_pages * 0.1:  # At least 10% of pages have text
            pages_with_text = len(extracted_text)
            print(f"Text extracted using PyMuPDF: {pages_with_text}/{total_pages} pages contained text.")
            
            # Write results in correct page order
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                for page_num, text in sorted(extracted_text):
                    txt_file.write(text)
                    txt_file.write('\n')  # Add a newline after each page
            return
            
    except ImportError:
        print("PyMuPDF not available. Install with: pip install pymupdf")
    except Exception as e:
        print(f"Error during PyMuPDF extraction: {str(e)}")
        logging.error(f"Error during PyMuPDF extraction: {str(e)}")
    
    # Step 2: Try pypdf (as fallback) with batch processing
    all_results = {}
    try:
        print("PyMuPDF extraction insufficient. Falling back to pypdf with batch processing...")
        from pypdf import PdfReader
        
        start_time = time.time()
        reader = PdfReader(pdf_path, strict=False)
        total_pages = len(reader.pages)
        
        # Create batches of indices
        batch_indices = create_batches(total_pages, batch_size)
        
        # Process each batch in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches - using the global function instead of a local one
            future_to_batch = {
                executor.submit(process_batch_pypdf, pdf_path, batch): batch 
                for batch in batch_indices
            }
            
            # Collect results
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                             total=len(future_to_batch), desc="Standard Extraction"):
                try:
                    batch_results = future.result(timeout=timeout)
                    all_results.update(batch_results)
                except concurrent.futures.TimeoutError:
                    batch = future_to_batch[future]
                    print(f"Batch {batch} processing timed out")
                except Exception as e:
                    batch = future_to_batch[future]
                    print(f"Error processing batch {batch}: {str(e)}")
                    
        elapsed = time.time() - start_time
        print(f"Standard extraction completed in {elapsed:.2f} seconds")
        
        # Check if we got a reasonable amount of text
        extracted_text = [(i, all_results[i]) for i in range(total_pages) 
                         if i in all_results and all_results[i] is not None]
        
        if extracted_text and len(extracted_text) > total_pages * 0.1:
            pages_with_text = len(extracted_text)
            print(f"Text extracted using standard extraction: {pages_with_text}/{total_pages} pages contained text.")
            
            # Write results in correct page order
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                for page_num, text in sorted(extracted_text):
                    txt_file.write(text)
                    txt_file.write('\n')  # Add a newline after each page
            return
        
    except Exception as e:
        print(f"Error during pypdf extraction: {str(e)}")
        logging.error(f"Error during pypdf extraction: {str(e)}")
    
    # Step 3: If we reach here, use OCR with batched processing
    try:
        print("Standard extraction insufficient. Using OCR extraction...")
        
        # Use PyMuPDF to convert PDF to images (much faster than pdf2image)
        print("Converting PDF to images using PyMuPDF (this might be faster)...")
        images = convert_pdf_to_images(pdf_path)
        
        if images is None or len(images) == 0:
            # Fallback to pdf2image if PyMuPDF conversion fails
            print("PyMuPDF image conversion failed. Falling back to pdf2image...")
            from pdf2image import convert_from_path
            
            # Try with higher DPI for better quality if the PDF has fewer pages
            dpi = 150 if total_pages < 100 else 100
            images = convert_from_path(pdf_path, dpi=dpi)
            
        total_images = len(images)
        print(f"Processing {total_images} pages with OCR using {max_workers} workers...")
        
        # Prepare image data for parallel processing
        image_data = [(i, images[i]) for i in range(total_images)]
        ocr_results = {}
        
        # Use ProcessPoolExecutor for CPU bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_image = {executor.submit(process_image_ocr, img_data): img_data[0] 
                             for img_data in image_data}
            
            # Use tqdm to track progress
            for future in tqdm(concurrent.futures.as_completed(future_to_image), 
                             total=len(future_to_image), desc="OCR Processing"):
                try:
                    page_num, text = future.result()
                    if text:  # Only store if text was actually extracted
                        ocr_results[page_num] = text
                except Exception as e:
                    logging.error(f"Error in OCR future: {str(e)}")
        
        # Write results in correct page order
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            for i in range(total_images):
                if i in ocr_results and ocr_results[i]:
                    txt_file.write(ocr_results[i])
                txt_file.write('\n\n')  # Add double newline after each page
                
        print(f"OCR extraction completed. Extracted text from {len(ocr_results)}/{total_images} pages.")
        
    except ImportError:
        print("OCR extraction requires additional libraries. Install with:")
        print("pip install pytesseract pdf2image pymupdf tqdm")
        print("You'll also need to install Tesseract OCR on your system:")
        print("- Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("- macOS: brew install tesseract")
        print("- Linux: apt-get install tesseract-ocr")

if __name__ == "__main__":
    pdf_path = 'atlantis/atlantis.pdf'  # Replace with your PDF file path
    txt_path = 'atlantis/atlantis.txt'   # Output text file path
    
    # Using batched processing for better performance and the blocks mode for better text order
    extract_text_from_pdf(pdf_path, txt_path, batch_size=25, extract_mode="blocks")
