from tqdm import tqdm
from osgeo import gdal
import json
import numpy as np
import cv2
import os
from lib.utils.tools.logger import Logger as Log
from xml.dom.minidom import parse
import argparse
import  random

color_dict = {1:(0,255,0),2:(255,0,0),3:(0,0,255),4:(0,128,255),5:(128,0,255),6:(0,255,128)}

def str2bool(v):
    """ Usage:
    parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True,
                        dest='pretrained', help='Whether to use pretrained models.')
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
def readtransxml(transxml):
    path_transxml = transxml
    if not os.path.exists(path_transxml):
        Log('Error:{} is not existed.'.format(path_transxml))
    transBM = parse(path_transxml)
    root = transBM.documentElement
    all_codes = root.getElementsByTagName('BM')
    all_dict = {}
    num_class = 0
    for node in all_codes:
        class_geoid_name = node.attributes['key'].value
        class_id = node.attributes['val'].value
        all_dict[int(class_geoid_name)] = int(class_id)
        if int(class_id) > num_class:
            num_class = num_class + 1
    return num_class,all_dict
def readImagedata(img_dataset, xoff = 0, yoff = 0, winx = 512, winy = 512,band_lower = np.array([0,0,0]), band_higher = np.array([255,255,255])):
    max_x = xoff + winx
    max_y = yoff + winy

    col = img_dataset.RasterXSize
    row = img_dataset.RasterYSize

    # when the max x and y of win over the border
    """
 (x,y)   
     _________
    |  win_x  |
    |<-win_y  |
    |         |
     ---------`(max_x, max_y)
    """
    if max_x >= col:
        xoff = col - winx - 1

    if max_y >= row:
        yoff = row - winy - 1

    if xoff < 0:
        xoff = 0

    if yoff < 0:
        yoff = 0


    img_bands = img_dataset.RasterCount
    img_data = img_dataset.ReadAsArray(xoff, yoff, winx, winy).copy()
    m_dataType = gdal.GetDataTypeName(img_dataset.GetRasterBand(1).DataType)

    if img_bands == 1:
        img_data = img_dataset.GetRasterBand(1).ReadAsArray(xoff, yoff, winx, winy).copy()
        return  img_data
    img_data = img_data.transpose(1, 2, 0)[:,:,:3]

    if m_dataType == 'Byte' or m_dataType == 'Unknown':
        return  img_data
    # strech
    else:
        mask = (img_data[:, :, 0] == 0) & (img_data[:, :, 1] == 0) & (img_data[:, :, 2] == 0)
        for i in range(3):
            t = (img_data[:, :, i] - band_lower[i]) * 255 / (band_higher[i] - band_lower[i])
            t[t< 0] = 1
            t[t > 255] = 255
            t[mask] = 0
            img_data[:,:,i] = t
        return img_data.astype(np.uint8)
def readlistFromodgt( odgt, max_sample=-1, start_idx=-1, end_idx=-1):
    if isinstance(odgt, list):
        list_sample = odgt
    elif isinstance(odgt, str):
        list_sample = [ json.loads(x.rstrip())for x in open(odgt, 'r', encoding='utf-8') if x.strip()]
    if max_sample > 0:
        list_sample = list_sample[0:max_sample]
    if start_idx >= 0 and end_idx >= 0:     # divide file list
        list_sample = list_sample[start_idx:end_idx]
    return list_sample
def computelowhighval(path,winx = 5000,winy = 5000,low = 0.25,high = 99.75):
    img_dataset = gdal.Open(path,gdal.GA_ReadOnly)

    img_XSize = img_dataset.RasterXSize
    img_YSize = img_dataset.RasterYSize
    winx = min(img_XSize,winx)
    winy = min(img_YSize,winy)
    img_data = img_dataset.ReadAsArray(0,0,img_XSize,img_YSize,buf_xsize=winx,buf_ysize=winy)
    img_data = img_data.transpose(1,2,0)
    band_lower = np.zeros((3))
    band_higher = np.zeros((3))
    mask = (img_data[:, :, 0] == 0) & (img_data[:, :, 1] == 0) & (img_data[:, :, 2] == 0)
    mask = ~mask
    # # img_array = img_array[mask]
    for i in range(3):
        band_lower[i] = np.percentile(img_data[:, :, i][mask], low)
        band_higher[i] = np.percentile(img_data[:, :, i][mask], high)
    #     t = (img_data[:, :, i] - band_lower[i]) * 255 / (band_higher[i] - band_lower[i])
    #     t[t < 0] = 0
    #     t[t > 255] = 255
    #     img_data[:, :, i] = t
    del img_dataset
    return  band_lower, band_higher
def createImageLabellist(args):
    if not os.path.exists(args.odgt):
        Log.error("cannot find odgt file{}".format(args.odgt))
        return
    if not os.path.exists(args.transxml):
        Log.error("cannot find decode file{}".format(args.transxml))
        return
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if args.clip:
        if not os.path.exists(os.path.join(args.savedir,'train')):
            os.makedirs(os.path.join(args.savedir,'train'))
        if not os.path.exists(os.path.join(args.savedir,'val')):
            os.makedirs(os.path.join(args.savedir,'val'))
    list_sample = readlistFromodgt(args.odgt)
    num_class,dict = readtransxml(args.transxml_obj)
    driver = gdal.GetDriverByName("GTiff")
    gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'NO')
    band_lowhigh = {}
    savetraindt = os.path.join(args.savedir,"train.dt")
    f = open(savetraindt,"w+",encoding='utf-8')
    for i,smp in tqdm(enumerate(list_sample)):
        savefile = os.path.join(args.savedir,"{:08d}".format(i))


        img_file = smp["fpath_img"]
        lbl_file = smp["fpath_segm"]
        img_dataset = gdal.Open(img_file,gdal.GA_ReadOnly)
        lab_dataset = gdal.Open(lbl_file, gdal.GA_ReadOnly)
        winx = smp['width']
        winy = smp['height']
        xoff = smp['xoff']
        yoff = smp['yoff']
        rgb_flag = smp['rgb']
        m_dataType = gdal.GetDataTypeName(img_dataset.GetRasterBand(1).DataType)
        if img_file not in band_lowhigh.keys():
            band_lowhigh[img_file] = computelowhighval(img_file)

        if m_dataType == gdal.GDT_Byte:
            img_patch = readImagedata(img_dataset, xoff, yoff, winx=winx, winy=winy)
        else:
            band_low,band_high = band_lowhigh[img_file]
            img_patch = readImagedata(img_dataset, xoff, yoff, winx=winx, winy=winy,band_lower= np.array(band_low).reshape(3,1),band_higher=np.array(band_high).reshape(3,1))
        lab_patch = readImagedata(lab_dataset, xoff, yoff, winx=winx, winy=winy)
        lab_mask = np.zeros(lab_patch.shape,dtype=np.uint8)

        # lab_patch = np.array(
        #     [dict[str(lab_patch[y][x])] for y in range(lab_patch.shape[0]) for x in range(lab_patch.shape[1])]).reshape((winy, winx)).astype('uint8')
        lab_patch = np.vectorize(dict.get)(lab_patch)
        img = img_patch.copy()
        label_list = []
        for j , lab in enumerate(range(1,num_class+1)):
            lab_mask[lab_patch == lab] = 255
            non_zero = np.count_nonzero(lab_mask)
            if non_zero  == 0 :
                continue
            contours,hier = cv2.findContours(lab_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            num_c = len(contours)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x+w, y+h),color_dict[lab], 2)
            label_list.append([lab,x,y,w,h])
            lab_mask = 0 * lab_mask
        if len(label_list) == 0:
            continue
        else:
            cv2.imshow("lab_mask",img)
            cv2.waitKey(0)
            save_dict = {}
            save_dict["image"] = smp["fpath_img"]
            save_dict["label"] = smp["fpath_segm"]
            save_dict["x"] = smp['xoff']
            save_dict["y"] = smp['yoff']
            save_dict["w"] = smp['width']
            save_dict["h"] = smp['height']
            save_dict["obj"] = label_list
            f.write(json.dumps(save_dict,ensure_ascii=True))
            f.write("\n")
        if args.clip:
            r = random.uniform(0,1)
            dir = 'train' if r < args.split else 'val'
            saveimg = os.path.join(args.savedir,dir,"img_{:08d}".format(i)+ ".tif",)
            savelbl = os.path.join(args.savedir,dir,"lbl_{:08d}".format(i)  + ".tif",)
            height,width,im_bands = img_patch.shape
            outDS_Img = driver.Create(saveimg, winx, winy, img_patch.shape[2], gdal.GDT_Byte)
            outDS_Lbl = driver.Create(savelbl, winx, winy, 1, gdal.GDT_Byte)

            if im_bands == 1:
                outDS_Img.GetRasterBand(0).WriteArray(img_patch,0,0)
            else:
                for band in range(im_bands):
                    outDS_Img.GetRasterBand(band+1).WriteArray(img_patch[:,:,band],0,0)
            outDS_Lbl.GetRasterBand(1).WriteArray(lab_patch,0,0)
            outDS_Img = None
            outDS_Lbl = None

    f.close()
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--odgt', default="D:/dengkai/Code/dengkai_DL/dataset/train_gtoffset.odgt", type=str,
                        help='The sample file encode in odgt.')
    parser.add_argument('--transxml', default="D:/dengkai/Code/dengkai_DL/configs/trans.xml", type=str, nargs='+',
                        help='The configure file of rsdata.exp:guoqing and sandiao')
    parser.add_argument('--transxml_obj', default="D:/dengkai/Code/dengkai_DL/configs/trans_obj.xml", type=str,
                         help='id needs to convert yolo dataformat.')
    parser.add_argument('--savedir', default="D:/dengkai/data/dataset_0827", type=str,help='The output dir.')
    parser.add_argument('--clip', default=1, type=str2bool,help='True: output Small Image')
    parser.add_argument('--split', default=0.8, type=float,help='spilt samples into train  or val dir')
    args = parser.parse_args()

    createImageLabellist(args)
