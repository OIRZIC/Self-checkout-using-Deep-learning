import numpy as np
import torch
import cv2
import os

from torchvision.tv_tensors import Video
from ultralytics import YOLO
import src.detect_bytetrack as Objectdetect
# from deep_sort.deep_sort import nn_matching
# from deep_sort.deep_sort.tracker import Tracker

#Database initialize

# database.create_tables()
# database.add_product('Pencil', 0.50, 100)
# database.add_product('Notebook', 1.20, 50)
# products = database.get_all_products()
# print("Products:", products)
#
# database.update_stock(1, 10)  # Reduce stock for product with ID 1 by 10
# print("Updated products:", database.get_all_products())
#
# database.delete_product(2)  # Delete product with ID 2
# print("Final products:", database.get_all_products())
#
# # Close connection at the end
# database.close_connection()









def main():

    print ("concu")
if __name__ == "__main__":
    main()