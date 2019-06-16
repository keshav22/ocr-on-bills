# OCR ON BILLS

## Installation & Setup
1. You have to run "main.py".
2. For that first we need to setup your PC. So requirements & steps to achieve it are listed below.
3. Check your python version it must be 3.6
4. Make sure you have following packages installed.
	 - Tesseract  you will find it in https://github.com/tesseract-ocr/tesseract/wiki & add it as Environmental variable(System Variable).
	 - Matplotlib  
	 - Pillow      
	 - Scipy 	   
	 - Opencv	   
	 - Pytesseract 
	 - numpy 	   
5. To Install the above dependencies:
> `pip install -r requirements.txt`
 
## Process & Results
1. Now heading into the process. Steps We used are pretty simple.
2. Some images are not perfectly straight so we used a skewing algorithm(skew.py). You can find the skewed images in "input_crop" folder.
3. Then we are detecting the text area in the Images & croping them out. You can find it in "input_dpi" folder .
4. Now to improve & as per research increasing the quality of image is better in image processing so we increased the DPI from 96 to 300. You can find the images in "input_otsu" folder.
5. Now applying otsu algorithm to binarise & increasing the threshold to get good output in tesseract-OCR. Leveling DPI again. You can the images in "finalimages" folder.
6. Using pytesseract to get the text output. You can find the text files in "final_text_files" folder.
7. Parsed the receipts text output to take out Amount spent & plotted it category wise in bar format using Matplotlib. 
8. You can find the Graph & Analysis report under Report directory.