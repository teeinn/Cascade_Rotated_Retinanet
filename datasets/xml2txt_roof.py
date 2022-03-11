import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, dump, ElementTree
import shutil
import numpy as np
import csv
import cv2


def loadXMLFiles(filename):

    line_total_ls = ['imagesource:GoogleEarth', 'gsd:0.2']

    xml_tree = ET.parse(filename)
    for object in xml_tree.iter("object"):
        if object.find('name').text == 'flat':
            object.find('name').text = 'flatroof'
            print('*************', filename)
        line = ' '.join([object.find('robndbox/x0').text, object.find('robndbox/y0').text, object.find('robndbox/x1').text, object.find('robndbox/y1').text,
                         object.find('robndbox/x2').text, object.find('robndbox/y2').text, object.find('robndbox/x3').text, object.find('robndbox/y3').text,
                         object.find('name').text, object.find('difficult').text])
        line_total_ls.append(line)

    return line_total_ls



def write_annoataion(save_path, line_ls):

    with open(save_path, 'w') as f:
        f.write('\n'.join(line_ls))
    print(1)

if __name__ == "__main__":
    txt_path = '/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/augmentation/labelTxt'
    xml_path = '/media/qisens/2tb1/python_projects/inference_pr/tfrecord-viewer/new_dataset_overfitting_test13_riverarea_area123/train/augmentation/annotations_new'

    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    for path, dirs, files in os.walk(xml_path):
        for file in files:
            file_path = os.path.join(path, file)
            line_ls = loadXMLFiles(file_path)

            txt_file_path = os.path.join(txt_path, file).replace('xml', 'txt')
            write_annoataion(txt_file_path, line_ls)