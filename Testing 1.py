import os
from PIL import Image
from transformers import pipeline

# Set the model name
model_name = "Norm/nougat-latex-base"

# Create the pipeline for image-to-text conversion.
# (This pipeline is expected to handle converting images to LaTeX code.)
latex_pipeline = pipeline("image-to-text", model=model_name)

def process_images_and_combine():
    """
    Process all images in the "images" folder (named 1.png, 2.png, etc.),
    convert each image to LaTeX using the model pipeline, and save all outputs in a single file.
    """
    image_folder = "images"
    combined_output_file = "combined_output.txt"
    
    # List image files with extensions .png, .jpg, .jpeg, sorted numerically by filename
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    combined_output = ""
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening {image_file}: {e}")
            continue
        
        # Use the pipeline to get output from the image.
        # The pipeline returns a list of dictionaries.
        results = latex_pipeline(image)
        if results and isinstance(results, list) and "generated_text" in results[0]:
            latex_code = results[0]["generated_text"]
        else:
            latex_code = ""
        
        page_number = os.path.splitext(image_file)[0]
        combined_output += f"--- Output from page {page_number} ---\n"
        combined_output += latex_code + "\n\n"
        print(f"Processed {image_file}")
    
    # Write all outputs to a single file
    with open(combined_output_file, "w", encoding="utf-8") as f:
        f.write(combined_output)
    
    print(f"Combined output written to {combined_output_file}")

if __name__ == "__main__":
    process_images_and_combine()
