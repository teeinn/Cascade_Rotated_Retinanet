import argparse
import os
import numpy as np
import cv2
from inference import evaluate, dota_evaluate
from utils.utils import hyp_parse
import time

def drawbox(img_path,object_coors,label_ls, save_flag=True,save_path=None):
    print(img_path)
    img_name = os.path.split(img_path)[1]
    new_save_path = os.path.join(save_path,img_name)
    img=cv2.imread(img_path)

    for idx, coor in enumerate(object_coors):
        img = cv2.polylines(img,[coor],True,(0,0,255),2)
        # cv2.rectangle(img, (coor[0][0], coor[0][1]), (coor[0][0]+10, coor[0][1]+10), (0,255,0), 3)  # filled
        cv2.putText(img, label_ls[idx], (coor[0][0], coor[0][1]), cv2.FONT_HERSHEY_COMPLEX, 0.45, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    if save_flag:
        cv2.imwrite(new_save_path, img)
    else:
        cv2.imshow(img_path,img)
        cv2.moveWindow(img_path,100,100)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_DOTA_points(results, img_path_, save_path):

    object_coors=[]
    label_ls = []
    for object in results:
        coors = object.split(' ')
        if float(coors[-1]) >= 0.4:
            label_ls.append(coors[8])
            coors = [int(eval(x)) for x in coors[:8]]
            x0 = coors[0]; y0 = coors[1]; x1 = coors[2]; y1 = coors[3]
            x2 = coors[4]; y2 = coors[5]; x3 = coors[6]; y3 = coors[7]
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))

    drawbox(img_path, object_coors, label_ls, save_flag=True, save_path=save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', dest='backbone', default='res101', type=str)
    parser.add_argument('--weight', type=str,
                        default='/media/qisens/2tb1/python_projects/training_pr/Cascade-Rotated-RetinaNet/weights/best.pth')
    parser.add_argument('--target_size', dest='target_size', default=[416], type=int)
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--dataset', nargs='?', type=str, default='DOTA')
    parser.add_argument('--test_path', nargs='?', type=str, default='DOTA/test')
    parser.add_argument('--conf', nargs='?', type=float, default=0.05)


    arg = parser.parse_args()
    img_path = os.path.join(arg.test_path, 'images_area123')
    save_path = os.path.join(arg.test_path, 'augmentation_area123_anchor_existed')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    hyps = hyp_parse(arg.hyp)
    model = evaluate(arg.target_size,
                     arg.test_path,
                     arg.dataset,
                     arg.backbone,
                     arg.weight,
                     hyps=hyps)

    for path, dirs, files in os.walk(img_path):
        for file in files:
            img_save_path = os.path.join(save_path, path.split('/images/')[-1])
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)

            img_path = os.path.join(path, file)
            start = time.time()
            results = dota_evaluate(model, arg.target_size, img_path, arg.conf)
            print("inference time: {}".format(time.time() - start))
            get_DOTA_points(results, img_path, img_save_path)