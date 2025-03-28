import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm

def pdf_to_images(pdf_path, output_folder='./test_images', page_indices=None, max_resolution=(2000, 2000), jpeg_quality=85, aspect_ratio_threshold=1.1):
    """
    Converts selected pages of a PDF into images using PyMuPDF, with a resolution cap.
    If an image is significantly wider than tall, it is split into two halves.

    Parameters:
    - pdf_path: Path to input PDF.
    - output_folder: Directory to save images.
    - page_indices: List of page indices to convert (e.g., range(5) for first 5 pages).
    - max_resolution: Maximum allowed resolution (width, height).
    - jpeg_quality: JPEG compression quality (1-100, higher = better quality).
    - aspect_ratio_threshold: If width/height exceeds this, the image is split in half.
    """
    os.makedirs(output_folder, exist_ok=True)
    pdf_name = os.path.basename(pdf_path).replace('.pdf', '')

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)  # Get total number of pages in PDF

        # If no specific pages provided, default to converting all pages
        if page_indices is None:
            page_indices = range(total_pages)

        saved_images = []

        for i in tqdm(page_indices, desc="Converting PDF to Images", unit="page"):
            if i >= total_pages:  # Prevent index errors
                print(f"⚠️ Skipping page {i} (out of range)")
                continue

            page = doc.load_page(i)  # Load page
            matrix = fitz.Matrix(2.0, 2.0)  # Scale factor for high-res rendering
            pix = page.get_pixmap(matrix=matrix)  # Render page with scaling

            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Resize if larger than max_resolution
            img.thumbnail(max_resolution, Image.LANCZOS)

            width, height = img.size

            # If width is significantly larger than height, split the image in half
            if width / height > aspect_ratio_threshold:
                left_half = img.crop((0, 0, width // 2, height))
                right_half = img.crop((width // 2, 0, width, height))

                left_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}_left.jpg")
                right_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}_right.jpg")

                left_half.save(left_path, format="JPEG", quality=jpeg_quality)
                right_half.save(right_path, format="JPEG", quality=jpeg_quality)

                saved_images.extend([left_path, right_path])
            else:
                image_path = os.path.join(output_folder, f"{pdf_name}_page_{i+1}.jpg")
                img.save(image_path, format="JPEG", quality=jpeg_quality)
                saved_images.append(image_path)

        print(f"✅ Conversion complete! Images saved in {output_folder}")
        return saved_images

    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []




def preprocess_image(image):
    """Prepares image by applying adaptive thresholding, noise removal, and inversion."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Thresholding
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 25)

    # Morphological Operations for noise removal
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove small noise using connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50:  # Threshold for noise removal
            processed[labels == i] = 0

    # Invert image
    processed = cv2.bitwise_not(processed)

    # Add a border
    bordered = cv2.copyMakeBorder(processed, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return bordered

def remove_borders(image):
    """Removes borders from scanned documents using edge and line detection."""
    if len(image.shape) == 3:  # Check if image has color channels
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    
    # Detect horizontal and vertical lines
    kernel_h = np.ones((1, 50), np.uint8)
    kernel_v = np.ones((50, 1), np.uint8)
    
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    
    # Combine detected lines
    lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
    
    # Detect strong lines using Hough Transform
    edges = cv2.Canny(lines, 50, 150, apertureSize=3)
    lines_detected = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=5)

    mask = np.ones_like(binary) * 255  # White background
    if lines_detected is not None:
        for line in lines_detected:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 0, 3)  # Draw detected lines in black

    # Dilate and subtract the mask from the original image
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def enhance_sharpness(image):
    """Enhances the sharpness of the image using PIL."""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Sharpness(pil_image)
    enhanced_image = enhancer.enhance(2)  # Increase sharpness factor
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)




def test_preprocessing(img_path):
    image = cv2.imread(img_path)
    preprocessed = preprocess_image(image)
    borderless = remove_borders(preprocessed)
    sharpened_img = enhance_sharpness(borderless)
    cv2.imwrite(img_path, sharpened_img)


def line_splitter(image_path ):
    print(f"Processing: {image_path}")  # Print for debugging
    # Run surya_detect with the image path
    os.system(f'surya_detect "{image_path}" --images')



def extract_bboxes(image_path, json_path,output_dir, threshold=10):
    # Load image
    image = cv2.imread(image_path)
    
    # Load JSON data from file
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    
    # Extract bounding box coordinates
    bboxes = json_data[(os.path.basename(image_path)).split(".")[0]][0]["bboxes"]

    if bboxes == []:
        print('No bbox found . Skipping this image')
        return None 

    # Step 1: Sort by y-coordinate first (top to bottom)
    bboxes.sort(key=lambda b: b["bbox"][1])

    # Step 2: Group bounding boxes by similar y-level (threshold = 10 pixels)
    grouped_bboxes = []
    current_group = [bboxes[0]]
    
    for i in range(1, len(bboxes)):
        if abs(bboxes[i]["bbox"][1] - current_group[-1]["bbox"][1]) <= threshold:
            current_group.append(bboxes[i])
        else:
            grouped_bboxes.append(sorted(current_group, key=lambda b: b["bbox"][0]))  # Sort by x within group
            current_group = [bboxes[i]]
    
    # Append last group
    if current_group:
        grouped_bboxes.append(sorted(current_group, key=lambda b: b["bbox"][0]))

    # Flatten the sorted list
    sorted_bboxes = [bbox for group in grouped_bboxes for bbox in group]

    cropped_images = []
    os.makedirs(output_dir, exist_ok=True)
    for idx, bbox in enumerate(sorted_bboxes):
        x1, y1, x2, y2 = bbox["bbox"]
    
        # Crop the region from the image
        cropped = image[y1:y2, x1:x2]
        
        # Convert to PIL format for easy display (optional)
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        cropped_images.append(cropped_pil)
        
        # Save cropped image
        cropped_pil.save(os.path.join(output_dir,f"cropped_{idx}.png"))
    
    return sorted_bboxes



