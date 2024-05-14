from utils.image_splitter import ImageSplitter
import time

if __name__ == '__main__':
    image_splitter = ImageSplitter("notebooks/three_column.jpg")
    t1 = time.time()
    _, _, _ = image_splitter.split_image()
    t2 = time.time()
    print("execution time:", t2 - t1)
