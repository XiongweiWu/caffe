import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as  plt
import sys
import argparse
import pickle

# Convert detection result file to wider face format

parser = argparse.ArgumentParser(description = "")
parser.add_argument('--detfile', dest='detfile', type=str, help="dets file here")

if __name__ == '__main__':    
    args = parser.parse_args()
    detfile = args.detfile
    print detfile
    save_root = 'Results'
    #/opt/StorageArray2/dataset/WIDERFACE/FACE2017/WIDER_val/images/29--Students_Schoolkids/29_Students_Schoolkids_Students_Schoolkids_29_74.jpg 1 0.999818 269 279 469 477
    
    map_ = {}
    with open(detfile, 'r') as f:
        recs = [x.strip() for x in f.readlines()]
        images = [x.split(' ')[0] for x in recs]
        cont = 0
        for rec in recs:
            print "{}/{}".format(cont+1, len(recs))
            cont += 1
            img_path, cls, score, xmin, ymin, xmax, ymax = rec.split(' ')
            img_idx = os.path.basename(img_path)[:-4]
            sub_folder = os.path.basename(os.path.dirname(img_path))
            save_folder = os.path.join(save_root, sub_folder)
            
            width = int(xmax) - int(xmin)
            height = int(ymax) - int(ymin)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_file = os.path.join(save_folder, img_idx+'.txt')
            
            detresult = "{} {} {} {} {}".format(xmin, ymin, width, height, score)
            
            if not map_.has_key(save_file):
                face_num = images.count(img_path)
                print face_num
                map_[save_file] = []
                map_[save_file].append(img_idx)
                map_[save_file].append(face_num)
                map_[save_file].append(detresult)
            else:
                map_[save_file].append(detresult)
    
    save_det = 'Results/wider_face_det.pkl'
    f = open(save_det, 'w')
    pickle.dump(map_, f)

    for save_ in map_.keys():
        with open(save_, 'w') as f1:
            for x in map_[save_]:
                f1.writelines(str(x)+'\n')



