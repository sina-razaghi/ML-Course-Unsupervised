import cv2
import glob

# Define the desired size
width = 92
height = 112

# Loop over all JPG images in the folder
for filename in glob.glob("*.jpg"):
    # Load the JPG image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # Resize the image to the desired size
    resized_img = cv2.resize(img, (width, height))

    # Save the resized image in PGM format
    pgm_filename = filename[:-4] + ".pgm"  # Replace the extension with ".pgm"
    cv2.imwrite(pgm_filename, resized_img)

    # Print a message to show the progress
    print(f"Converted {filename} to {pgm_filename}")
