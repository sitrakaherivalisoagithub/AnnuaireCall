# import module
from pdf2image import convert_from_path


class PDFConverter:

	def __init__(self, filename: str, path_of_poopler: str):

		self.path_of_poopler = path_of_poopler
		self.filename = filename

	def convert(self):
		# Store Pdf with convert_from_path function
		images = convert_from_path(self.filename,poppler_path= self.path_of_poopler)
		for i in range(len(images)):
			# Save pages as images in the pdf
			images[i].save('page' + str(i) + '.jpg', 'JPEG')


if __name__ == '__main__':
	path = "C:\\Users\\s.herivalisoa\\poppler-24.02.0\\Library\\bin"
	converter = PDFConverter("20240503091023635.pdf", path_of_poopler=path)
	converter.convert()
