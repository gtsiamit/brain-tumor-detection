import cv2
import numpy as np
from pathlib import Path
import argparse
import json

FILEDIR = Path(__file__).parent
BASE_DIR = FILEDIR.parent

ANNOTATIONS_MAPPING = {
    'TRAIN': 'annotations_train.json', 
    'VAL': 'annotations_val.json', 
    'TEST': 'annotations_test.json'
}


def handle_shape(img_path, shape_attributes):

    image_loaded = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image_shape = image_loaded.shape
    mask = np.zeros(shape=image_shape)

    shape_type_name = shape_attributes['name']

    if shape_type_name=='polygon':
        x_points = shape_attributes['all_points_x']
        y_points = shape_attributes['all_points_y']
        concat_points = list( zip(x_points, y_points) )
        
        concat_points = list( map(list, concat_points) )
        concat_points = np.array(concat_points)

        mask = cv2.fillPoly(img=mask, pts=[concat_points], color=(255))
    
    elif shape_type_name=='ellipse':
        center = ( shape_attributes['cx'], shape_attributes['cy'] )
        axes = ( np.round(shape_attributes['rx']).astype(int), np.round(shape_attributes['ry']).astype(int) )
        angle = shape_attributes['theta']

        mask = cv2.ellipse(
            img=mask, center=center, axes=axes, 
            angle=angle, startAngle=0, endAngle=360, 
            color=(255), thickness=-2
        )
    
    elif shape_type_name=='circle':
        center = ( shape_attributes['cx'], shape_attributes['cy'] )
        radius = np.round(shape_attributes['r']).astype(int)

        mask = cv2.circle(img=mask, center=center, radius=radius, color=(255), thickness=-2)
    
    return mask


def create_masks():

    for key in ANNOTATIONS_MAPPING:
        set_path = DATASET_PATH.joinpath(f'archive/Br35H-Mask-RCNN/{key}')
        annotation_path = set_path.joinpath(ANNOTATIONS_MAPPING[key])

        with open(str(annotation_path)) as f:
            annot_data = json.load(f)
        
        dict_keys_list = list( annot_data.keys() )
        num_samples = len(dict_keys_list)

        for sample_idx in range(num_samples):
            sample_annotation = annot_data[ dict_keys_list[sample_idx] ]
            
            fname = sample_annotation['filename']
            img_filepath = set_path.joinpath(fname)

            annot_shape_attributes = sample_annotation['regions'][0]['shape_attributes']

            mask = handle_shape(img_path=str(img_filepath), shape_attributes=annot_shape_attributes)
            
            mask_fname = fname.split('.')[0] + '_mask.jpg'
            mask_path = set_path.joinpath(mask_fname)
            cv2.imwrite(filename=str(mask_path), img=mask)
        
        print(f'Masks for {key} set generated')
    
    print(f'-- Masks creation finished --')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=False, help='Path to dataset')
    args = parser.parse_args()

    global DATASET_PATH
    DATASET_PATH = args.dataset_path
    if not DATASET_PATH:
        DATASET_PATH = BASE_DIR.joinpath('dataset')
    else:
        DATASET_PATH = Path(DATASET_PATH)
    
    create_masks()


if __name__=='__main__':
    main()


