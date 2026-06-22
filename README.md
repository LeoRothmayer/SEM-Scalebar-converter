# SEM-Scalebar-converter
Simple GUI Script which reads .tif SEM (Scanning Electron Microscope) Images and creates a nicer scalebar. File are then saved as .png into a selected destination folder.

## .tif images
Pixel size in SI units is saved in the metadata and used for scalebar creation.

## Scalebar rendering
The rendered scalebar (line + label) is drawn on top of a semi-transparent dark background rectangle for better readability against bright images. The rectangle is clamped to the image bounds so it can never bleed past the edges.

The scalebar length is picked from a dense "1-2-5" step ladder (50 nm up to 50 mm), choosing the largest size that still fits comfortably within the image width — this gives much finer length choices at low magnifications (large pixel sizes) than a handful of fixed thresholds would. The bar is anchored to a fixed left margin rather than a fixed center, so long bars are never cropped off the left edge.

## Mirror folder structure
When selecting source folders, the tool recursively finds all `.tif` files in the folder and its subfolders. By default, all rendered `.png` files are saved flat into the output folder.

Checking **"Mirror folder structure"** instead recreates the original subfolder layout: for each selected source folder, a copy named `{folder name}_png` is created inside the output folder, and every rendered image is placed at the same relative subpath it had in the source folder.

If the selected output folder's name already ends with `_png` (e.g. you picked a previously rendered output folder to re-render in place), the mirrored subfolder structure is recreated directly inside that folder instead of nesting another `{folder name}_png` layer, and existing files are overwritten.

## Disclaimer
Base functions and method portion was written by myself. The GUI implementation was generated using Claude code.
