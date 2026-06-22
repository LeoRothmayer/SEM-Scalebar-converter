# SEM-Scalebar-converter
Simple GUI Script which reads .tif SEM (Scanning Electron Microscope) Images and creates a nicer scalebar. File are then saved as .png into a selected destination folder.

## .tif images
Pixel size in SI units is saved in the metadata and used for scalebar creation.

## Scalebar rendering
The rendered scalebar (line + label) is drawn on top of a semi-transparent dark background rectangle for better readability against bright images.

## Mirror folder structure
When selecting source folders, the tool recursively finds all `.tif` files in the folder and its subfolders. By default, all rendered `.png` files are saved flat into the output folder.

Checking **"Mirror folder structure"** instead recreates the original subfolder layout: for each selected source folder, a copy named `{folder name}_png` is created inside the output folder, and every rendered image is placed at the same relative subpath it had in the source folder.

## Disclaimer
Base functions and method portion was written by myself. The GUI implementation was generated using Claude code.
