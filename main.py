from PIL import Image
import skimage as ski
from skimage import color, filters, transform
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import heatmap
import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.exposure import rescale_intensity
import os
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_dilation, disk
import yaml
import argparse

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config = {
        "rgb_path": cfg["paths"]["rgb"].replace("/*.png", ""),
        "nrg_path": cfg["paths"]["nrg"].replace("/*.png", ""),
        "mask_path": cfg["paths"]["masks"].replace("/*.png", ""),
        "h_min": float(cfg["thresholds"]["LOWER_PINK"][0]),
        "h_max": float(cfg["thresholds"]["UPPER_PINK"][0]),
        "size": tuple(cfg["general"]["TARGET_SIZE"]),
        "blue_min": np.array(cfg['thresholds']['LOWER_BLUE']),
        "blue_max": np.array(cfg['thresholds']['UPPER_BLUE'])
    }
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Dead Tree Segmentation Pipeline")
    
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to config.yaml")

    parser.add_argument("-h_min", "--hue-min", type=float, help="Override Lower Hue threshold")
    parser.add_argument("-h_max", "--hue-max", type=float, help="Override Upper Hue threshold")
    
    return parser.parse_args()

def paths(cfg):

    nrg_list = sorted(os.listdir((cfg['nrg_path'])))
    rgb_list = sorted(os.listdir((cfg['rgb_path'])))
    masks_list = sorted(os.listdir((cfg['mask_path'])))

    return nrg_list, rgb_list, masks_list

def keggle_masks(masks_list, cfg):
    list_kegle_mask = []

    for i in masks_list:

        kegle_mask_array = np.array(Image.open(os.path.join(cfg['mask_path'], i)))
        
        kegle_mask_resized = ski.transform.resize(kegle_mask_array, cfg['size'], order=0)

        kegle_masks_uint8 = (kegle_mask_resized.astype(np.uint8)) * 255
        list_kegle_mask.append(kegle_masks_uint8)

    return list_kegle_mask

def rgb_masks(rgb_list, cfg):

    list_rgb_masks = []
    rgb_images_list = []
     
    for i in rgb_list:

        rgb_array = np.array(Image.open(os.path.join(cfg["rgb_path"], i)))

        rgb_images_list.append(rgb_array)
        
        rgb_to_hsv = ski.color.rgb2hsv(rgb_array)
        rgb_treshold = (rgb_to_hsv[:,:,0] >= cfg["h_min"]) & (rgb_to_hsv[:,:,0] <= cfg["h_max"])

        rgb_resized_bool = ski.transform.resize(rgb_treshold, cfg["size"], order=0, preserve_range=True)

        rgb_filling = binary_fill_holes(rgb_resized_bool)

        rgb_uint8 = (rgb_filling.astype(np.uint8)) * 255
        
        list_rgb_masks.append(rgb_uint8)


    return list_rgb_masks, rgb_images_list

def nrg_masks(nrg_list, cfg):

    nir_masks_list = []
    nir_images_list = []

    for i in nrg_list:

        nir_array = np.array(Image.open(os.path.join(cfg['nrg_path'], i)))

        nir_images_list.append(nir_array)

        nir_resize = ski.transform.resize(nir_array, cfg['size'], order=0, preserve_range=True).astype(float)

        nir = nir_resize[:, :, 0]
        red = nir_resize[:, :, 1]

        ndvi = (nir - red) / (nir + red + 1e-10)

        nir_mask = (ndvi > 0.25) & (ndvi < 0.6)

        nir_filling = binary_fill_holes(nir_mask)

        mask_uint8 = (nir_filling.astype(np.uint8)) * 255

        nir_masks_list.append(mask_uint8)

    return nir_masks_list, nir_images_list

def nrg_treshold_method(nrg_list, cfg):

    nir_treshold_list = []

    nir_hsv_list = []
    
    for i in nrg_list:
    
        nir_array = np.array(Image.open(os.path.join(cfg['nrg_path'], i)))

        nir_resize = ski.transform.resize(nir_array, cfg['size'], order=0, preserve_range=True).astype(float)

        nir_hsv = rgb2hsv(nir_resize)
        nir_hsv_list.append(nir_hsv)

        nir_treshold = np.all((nir_hsv >= cfg['blue_min']) & (nir_hsv <= cfg['blue_max']), axis=-1)

        nir_filling = binary_fill_holes(nir_treshold)

        rgb_uint8 = (nir_filling.astype(np.uint8)) * 255
        
        nir_treshold_list.append(rgb_uint8)

    return nir_treshold_list, nir_hsv_list


def combined_masks(list_rgb_masks, nir_masks_list, nir_treshold_list):

    list_combined_masks = []
    list_combined_masks_tresh = []

    for i, j, k in zip(list_rgb_masks, nir_masks_list, nir_treshold_list):

        rgb_bool = i > 0
        nir_bool = j > 0
        nir_tresh_bool = k > 0

        combined = (rgb_bool & nir_bool)
        combined_2 = (rgb_bool & nir_tresh_bool)

        combined_filling_holes = binary_fill_holes(combined)
        combined_filling_holes_2 = binary_fill_holes(combined_2)

        combined_dilation = binary_dilation(combined_filling_holes, disk(5))
        combined_dilation_2 = binary_dilation(combined_filling_holes_2, disk(5))

        list_combined_masks.append(combined_dilation)
        list_combined_masks_tresh.append(combined_dilation_2)

    return list_combined_masks, list_combined_masks_tresh

def IoU(list_combined_masks, list_kegle_mask):
    list_for_IoU = []

    for i, j in zip(list_combined_masks, list_kegle_mask):

        gt_bool = j > 0
        compered_masks = i > 0

        intersection = compered_masks & gt_bool
        union = compered_masks | gt_bool

        intersection_count = np.count_nonzero(intersection)
        union_count = np.count_nonzero(union)

        if union_count == 0:
            iou = 0
        else:
            iou = intersection_count/union_count

        list_for_IoU.append(iou)

    avrage_IoU = (sum(list_for_IoU)/len(list_for_IoU))*100

    return avrage_IoU, list_for_IoU

def tresh_IoU(list_kegle_mask, list_combined_masks_tresh):

    tresh_method_IoU_list = []

    for i, j in zip(list_kegle_mask, list_combined_masks_tresh):

        gt_bool = i > 0
        compered_masks = j > 0

        intersection = compered_masks & gt_bool
        union = compered_masks | gt_bool

        intersection_count = np.count_nonzero(intersection)
        union_count = np.count_nonzero(union)

        if union_count == 0:
            iou = 0
        else:
            iou = intersection_count/union_count

        tresh_method_IoU_list.append(iou)

    avrage_IoU_tresh = (sum(tresh_method_IoU_list)/len(tresh_method_IoU_list))*100

    return avrage_IoU_tresh, tresh_method_IoU_list

def function_confusion_matrix(list_combined_masks, list_kegle_mask):

    confusion_matrix_list = []

    for i, j in zip(list_combined_masks, list_kegle_mask):

        combined_masks = i > 0
        kegle_masks = j > 0

        pred = combined_masks.astype(int).flatten()
        ground_truth = kegle_masks.astype(int).flatten()

        cm = confusion_matrix(ground_truth, pred, labels = [0, 1])
        confusion_matrix_list.append(cm)
    
    return confusion_matrix_list

def confusion_matrix_metrics(confusion_matrix_list):

    cm_matrix_array_sum = np.sum(confusion_matrix_list, axis = 0)

    TN = cm_matrix_array_sum[0][0]
    FN = cm_matrix_array_sum[1][0]
    TP = cm_matrix_array_sum[1][1]
    FP = cm_matrix_array_sum[0][1]

    accurucy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    error = (FP + FN) / (TP + TN + FP+ FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    recall_score = TP / (TP + FN + 1e-10)
    f1_score = 2 * (precision * recall_score) / (precision + recall_score + 1e-10)

    return accurucy, error, precision, recall_score, f1_score

#ploting

def IOU_bar_chart(avrage_IoU, list_for_IoU):

    evaluated_IoU_values = []

    for i in list_for_IoU:

        i = i*100
        evaluated_IoU_values.append(i)

    x = range(len(list_for_IoU))
    
    plt.bar(x, evaluated_IoU_values)
    plt.title('porównanie wartości IoU')
    plt.ylabel('wartość IoU')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.axhline(avrage_IoU, label=f'Mean: {avrage_IoU:.2f}%')
    plt.legend()
    plt.show()

    return None

def tresh_IOU_bar_chart(avrage_IoU_tresh, tresh_method_IoU_list):

    evaluated_nir_IoU_values = []

    for i in tresh_method_IoU_list:

        i = i*100
        evaluated_nir_IoU_values.append(i)

    x = range(len(tresh_method_IoU_list))
    
    plt.bar(x, evaluated_nir_IoU_values)
    plt.title('compare combined tresholds mask IoU')
    plt.ylabel('wartość IoU')
    plt.grid(True)
    plt.ylim(0, 100)
    plt.axhline(avrage_IoU_tresh, label=f'Mean: {avrage_IoU_tresh:.2f}%')
    plt.legend()
    plt.show()

    return None

def confusion_matrix_heat_map(confusion_matrix_list):

    sum_confusion_matrix = np.sum(confusion_matrix_list, axis = 0)

    sns.heatmap(sum_confusion_matrix, vmin=None, vmax=None,
        cmap='Reds', center=None, robust=False,
        annot = True, fmt='d', annot_kws=None, linewidth = 0,
        linecolor='white', cbar=True, cbar_kws=None, cbar_ax=None,
        square=False, xticklabels='auto', yticklabels='auto',
        mask=None, ax=None)
    
    plt.show()
    
    return None
    
def plot_confusion_metrics(accurucy, error, precision, recall_score, f1_score):

    x = ['Accuracy', 'Error', 'Precision', 'Recall', 'F1']
    y = [accurucy, error, precision, recall_score, f1_score]

    plt.figure(figsize=(15, 8))

    plt.bar(x, y)
    plt.title('Compare of confusion matrix metrics')
    plt.ylabel('Metric value')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    return None

def ploting_images(list_rgb_masks, nir_masks_list, list_combined_masks, list_kegle_mask, rgb_images_list, nrg_images_list):

    fig, axes = plt.subplots(1, 6, figsize=(16, 4))

    axes[0].imshow(list_rgb_masks[0], cmap='gray')
    axes[0].set_title("RGB mask")

    axes[1].imshow(nir_masks_list[0], cmap='gray')
    axes[1].set_title("NIR mask")

    axes[2].imshow(list_combined_masks[0], cmap='gray')
    axes[2].set_title("Combined mask")

    axes[3].imshow(list_kegle_mask[0], cmap='gray')
    axes[3].set_title("keggle mask")

    axes[4].imshow(rgb_images_list[0], cmap = 'gray')
    axes[4].set_title('rgb image')

    axes[5].imshow(nrg_images_list[0])
    axes[5].set_title('nir images')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_hsv_tresh(nir_hsv_list):
    
    plt.imshow(nir_hsv_list[0])
    plt.show()

    return None


def main():
    
    args = parse_args()

    config = load_config(args.config)

    if args.hue_min is not None:
        config["h_min"] = args.hue_min
    if args.hue_max is not None:
        config["h_max"] = args.hue_max
    
    nrg, rgb, mask = paths(config)
    kegle_m = keggle_masks(mask, config)
    rgb_m, rgb_images_list = rgb_masks(rgb, config)
    nrg_m, nrg_images_list = nrg_masks(nrg, config)
    nrg_tresh, nir_hsv_l = nrg_treshold_method(nrg, config)

    combined_m, combined_t = combined_masks(rgb_m, nrg_m, nrg_tresh)

    IoU_avr, IoU_l = IoU(combined_m, kegle_m)

    avrage_IoU_t, tresh_method_IoU_l = tresh_IoU(kegle_m, combined_t)

    cm_fun = function_confusion_matrix(combined_m, kegle_m)

    acc, err, prec, rec_sc, f1_sc = confusion_matrix_metrics(cm_fun)

    IOU_bar_chart(IoU_avr, IoU_l)

    tresh_IOU_bar_chart(avrage_IoU_t, tresh_method_IoU_l)

    confusion_matrix_heat_map(cm_fun)

    plot_confusion_metrics(acc, err, prec, rec_sc, f1_sc)

    ploting_images(rgb_m, nrg_m, combined_m, kegle_m, rgb_images_list, nrg_images_list)

    plot_hsv_tresh(nir_hsv_l)

if __name__ == '__main__':
    main()
