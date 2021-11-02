import csv
from zipfile import ZipFile
import os
from Evaluator import *
# from utils import *
import copy
import argparse
import sys
import ntpath
#import cStringIO 
from io import BytesIO
import shutil
from datetime import datetime


def read_file(filename, bboxes, flag):
    '''
    Parses the input .csv file into map where key as page number and value as a list of bounding box objects
    corresponding to each math region in the file.
    :param filename: .csv file containing math regions
    :param bboxes: Map<page_num, List<bboxes>>
    :return:
    '''
    fh1 = open(filename, "r")
    prev_page = -1
    counter = 1
    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(",")
        idClass = float(splitLine[0])
        if prev_page == -1:
            prev_page = idClass
        else:
            if idClass != prev_page:
                counter = 1
                prev_page = idClass
        x = float(splitLine[1])
        y = float(splitLine[2])
        x2 = float(splitLine[3])
        y2 = float(splitLine[4])
        bb = BoundingBox(
            flag+"_"+str(counter),
            1,
            x,
            y,
            x2,
            y2,
            CoordinatesType.Absolute, (200, 200),
            BBType.GroundTruth,
            format=BBFormat.XYX2Y2)
        counter += 1
        #print(counter)
        if idClass not in bboxes:
            bboxes[idClass] = []
        bboxes[idClass].append(bb)

    fh1.close()


def extract_zipfile(zip_filename, target_dir):
    '''
    Extract zip file into the target directory
    :param zip_filename: full-file-path of the zip-file
    :param target_dir: target-dir to extract contents of zip-file
    :return:
    '''
    with ZipFile(zip_filename, 'r') as zip:
        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall(target_dir)
        print('Done!')


def create_doc_bboxes_map(dir_path,flag):
    '''
    Reads all files recursively in directory path and and returns a map containing bboxes for each page in each math
    file in directory.
    :param dir_path: full directory path containing math files
    :return: Map<PDF_name, Map<Page_number, List<BBoxes>>>
    '''
    pdf_bboxes_map = {}

    for filename in os.listdir(dir_path):
        full_filepath = os.path.join(dir_path, filename)
        filename_key = os.path.splitext(os.path.basename(full_filepath))[0]
        #print(full_filepath)
        if (full_filepath.startswith(".")) or (not (full_filepath.endswith(".csv") or full_filepath.endswith(".math"))):
            continue
        bboxes_map = {}

        if os.path.isdir(full_filepath):
            continue

        try:
            read_file(full_filepath, bboxes_map,flag)
        except Exception as e:
            print('exception occurred in reading file',full_filepath, str(e))
        
        #if len(bboxes_map)==0:
        #    raise ValueError("Empty ground truths file or not in valid format")
        pdf_bboxes_map[filename_key] = copy.deepcopy(bboxes_map)

    
    return pdf_bboxes_map

def unique_values(input_dict):
    #return ground truth boxes that have same det boxes
    pred_list=[]
    repair_keys=[]
    for value in input_dict.values():
        if value[1] in pred_list: #preds.append(value)
            gts=[k for k,v in input_dict.items() if v[1] == value[1]]
            #print('pair length',len(gts))
            repair_keys.append(gts)
        pred_list.append(value[1])
    
    return repair_keys
 
def generate_validpairs(pairs): 
    newpairs=[]
    for pair in pairs:
        if len(pair)>2:
            for i in range(len(pair)-1):
                newpair=(pair[i],pair[i+1])
                if newpair not in newpairs:newpairs.append(newpair)
                
        elif pair not in newpairs: newpairs.append(pair)
    return newpairs

def fix_preds(input_dict,keyPairs,thre):
    
    validPairs=generate_validpairs(keyPairs)
    
    for pair in validPairs:
        #check if both pair exists"
        if pair[0] not in list(input_dict.keys()) or pair[1] not in list(input_dict.keys()):
            continue
        val0=input_dict[pair[0]][0]
        val1=input_dict[pair[1]][0]
        if val0>=val1: #change prediction for second pair
            values = input_dict[pair[1]]
            if len(values[2]) > 1:
                newprob=values[2][1]
                if newprob<thre:
                    del input_dict[pair[1]]
                    continue
                #update dict
                input_dict[pair[1]]=newprob,values[3][1],values[2][1:],values[3][1:]
        else:
            del input_dict[pair[1]]
            
        if val1>val0: #change prediction for first pair
            values=input_dict[pair[0]]
            if len(values[2]) > 1:
                newprob=values[2][1]
                if newprob<thre:
                    del input_dict[pair[0]]
                    continue
                #update dict
                input_dict[pair[0]]=newprob,values[3][1],values[2][1:],values[3][1:]
        else:
            del input_dict[pair[0]]
    
    return input_dict      

def find_uni_pred(input_dict,thre):
    # check if it is unique
    pairs=unique_values(input_dict) 
    if pairs==[]:
        return input_dict
  
    while pairs:        
        output_dict=fix_preds(input_dict,pairs,thre)
        pairs=unique_values(output_dict) 
    
    return output_dict
              
        
def count_true_box(pred_dict,thre):
    #remove predictions below thre from dict
    for key in list(pred_dict.keys()):
        max_prob=pred_dict[key][0]
        if max_prob<thre:
            del pred_dict[key]
      
    #check for 101 mapping
    final_dict=find_uni_pred(pred_dict,thre)
    
    count=len(final_dict.keys())        
    return count,final_dict


def IoU_page_bboxes(gt_page_bboxes_map, det_page_bboxes_map, pdf_name, outdir=None):
    '''
    Takes two maps containing page level bounding boxes for ground truth and detections for same PDF filename and
    computes IoU for each BBox in a page in GT against all BBoxes in the same page in detections and returns them in
    decreasing value of IoU. In this way it computes IoU for all pages in map.

    :param gt_page_bboxes_map: Map<pageNum, List<bboxes>> for ground truth bboxes
    :param det_page_bboxes_map: Map<pageNum, List<bboxes>> for detection bboxes
    :return:
    '''
    evaluator = Evaluator() 
    
    correct_pred_1=0
    correct_pred_25=0
    correct_pred_50 = 0
    correct_pred_75 = 0
    correct_pred_100 = 0
    
    pdf_gt_boxes=0
    pdf_det_boxes=0

    coarse_keys = {}
    fine_keys = {}
    keys_1 = {}
    keys_25 = {}
    keys_50 = {}
    keys_75 = {}
    keys_100 = {}

    for page_num in gt_page_bboxes_map:
        if page_num not in det_page_bboxes_map:
            print('Detections not found for page', str(page_num + 1), ' in', pdf_name)
            continue
        gt_boxes = gt_page_bboxes_map[page_num]
        det_boxes = det_page_bboxes_map[page_num]
        
        pdf_gt_boxes+=len(gt_boxes)
        pdf_det_boxes+=len(det_boxes)

        pred_dict={}
        for gt_box in gt_boxes:
            ious = evaluator._getAllIOUs(gt_box, det_boxes) 
            preds=[]
            labels=[]
            for i in range(len(ious)):
                preds.append(round(ious[i][0],2))
                labels.append(ious[i][2].getImageName())
                
            pred_dict[gt_box.getImageName()]=preds[0],labels[0],preds,labels
        
        iou_1,iou_1_dict=count_true_box(copy.deepcopy(pred_dict),0.1)
        iou_25, iou_25_dict = count_true_box(copy.deepcopy(pred_dict), 0.25)
        iou_50, iou_50_dict = count_true_box(copy.deepcopy(pred_dict), 0.50)
        iou_75, iou_75_dict = count_true_box(copy.deepcopy(pred_dict), 0.75)
        iou_100, iou_100_dict = count_true_box(copy.deepcopy(pred_dict), 1.0)

        keys_1[page_num] = iou_1_dict.keys()
        keys_25[page_num] = iou_25_dict.keys()
        keys_50[page_num] = iou_50_dict.keys()
        keys_75[page_num] = iou_75_dict.keys()
        keys_100[page_num] = iou_100_dict.keys()

        # count correct preds for coarse 0.1, 0.25, 0.5, 0.75 and 1.0 in one page
        correct_pred_1= correct_pred_1+iou_1
        correct_pred_25= correct_pred_25+iou_25
        correct_pred_50 = correct_pred_50 + iou_50
        correct_pred_75 = correct_pred_75 + iou_75
        correct_pred_100 = correct_pred_100 + iou_100
        #write iou per page        
        # if outdir:
        #     out_file = open(os.path.join(outdir,pdf_name.split(".csv")[0]+"_"+str(page_num)+"_eval.txt"), "w")
        #     out_file.write('#page num '+str(page_num)+", gt_box:"+str(len(gt_boxes))+
        #                    ", pred_box:"+str(len(det_boxes))+"\n")
        #     out_file.write('\n')
        #     out_file.write('#COARSE DETECTION (iou>0.5):\n#number of correct prediction:'+ str(coarse)+ '\n#correctly detected:'+
        #                    str(list(coarse_dict.keys()))+'\n')
        #     out_file.write('\n')
        #     out_file.write('#FINE DETECTION (iou>0.75):\n#number of correct prediction:'+ str(fine)+ '\n#correctly detected:'+
        #                    str(list(fine_dict.keys()))+'\n')
        #     out_file.write('\n')
        #     out_file.write('#Sorted IOU scores for each GT box:\n')
        #     for gt_box in gt_boxes:
        #         ious = evaluator._getAllIOUs(gt_box, det_boxes)
        #         out_file.write(gt_box.getImageName()+",")
        #         for i in range(len(ious)-1):
        #             out_file.write("("+str(round(ious[i][0],2))+" "+ str(ious[i][2].getImageName())+"),")
        #         out_file.write( "("+str(round(ious[-1][0],2))+" "+ str(ious[-1][2].getImageName())+")\n" )
        #     out_file.close()

    return correct_pred_1, correct_pred_25, correct_pred_50, correct_pred_75, correct_pred_100,\
           pdf_gt_boxes, pdf_det_boxes, keys_1, keys_25, keys_50, keys_75, keys_100

def count_box(input_dict):
    count=0
    for pdf in input_dict.values():
        for page in pdf.values():
            count+=len(page)
            
    return count

# Zip every uploading files
def archive_iou_txt(username, task_id, sub_id,userpath):

    inputdir=os.path.join(userpath,'iouEval_stats')
    
    if not os.path.exists(inputdir):
        print('No txt file is generated for IOU evaluation')
        pass
    
    dest_uploader = 'IOU_stats_archive'
    dest_uploader = os.path.join(userpath, dest_uploader)

    if not os.path.exists(dest_uploader):
        os.makedirs(dest_uploader)

    zip_file_name = '/' + task_id + '_' + sub_id
    shutil.make_archive(dest_uploader + zip_file_name, 'zip', inputdir)

    # return '/media/' + dest_uploader

def  write_html(gtFile,resultsFile,info,scores,destFile):
    
		destFile.write('<html>')
		destFile.write('<head><link rel="stylesheet" href="//maxcdn.bootstrapcdn.com/font-awesome/4.3.0/css/font-awesome.min.css"><link href="/static/css/bootstrap.min.css" rel="stylesheet"></head>')
		destFile.write('<body>')
		#writeCSS(destFile)
		destFile.write ("<blockquote><b>CROHME 2019</b> <h1> Formula Detection Results ( TASK 3 )</h1><hr>")
		destFile.write("<b>Submitted Files</b><ul><li><b>Output:</b> "+ ntpath.basename(resultsFile) +"</li>")    		
		destFile.write ("<li><b>Ground-truth:</b> " + ntpath.basename(gtFile) + "</li></ul>")
		if info['allGTbox'] == 0:
			sys.stderr.write("Error : no sample in this GT list !\n")
			exit(-1) 
        #all detection and gt boxes 
		destFile.write ("<p><b> Number of ground truth bounding boxes: </b>" + str(info['allGTbox']) + "<br /><b> Number of detected bounding boxes: </b>" + str(info['allDet']))            
		destFile.write ("<hr>")
        #coarse results
		destFile.write ("<p><b> **** Coarse Detection Results (IOU>0.5) ****</b><br />")
		destFile.write ("<ul><li><b>"+str(scores['coarse_f']) + "</b> F-score</li>")
		destFile.write ("<li>"+str(scores['coarse_pre']) + " Precision</li>")
		destFile.write ("<li>"+str(scores['coarse_rec']) + " Recall</li></ul>")
		destFile.write ("<b>" + str(info['correctDet_c']) + "</b> Number of correctly detected bounding boxes</p>")
		destFile.write ("<hr>")
        #fine results
		destFile.write ("<p><b> **** Fine Detection Results (IOU>0.75) ****</b><br />")
		destFile.write ("<ul><li><b>"+str(scores['fine_f']) + "</b> F-score</li>")
		destFile.write ("<li>"+str(scores['fine_pre']) + " Precision</li>")
		destFile.write ("<li>"+str(scores['fine_rec']) + " Recall</li></ul>")
		destFile.write ("<b>" + str(info['correctDet_f']) + "</b> Number of correctly detected bounding boxes</p>")
		destFile.write ("<hr>")
		destFile.write('</body>')
		destFile.write('</html>')
                
def pre_rec_calculate(count):
    
    if count['allDet']==0:
        print ('No detection boxes found')
        scores={'fine_f':0,'coarse_f':0}
    else:
        pre_1 = count['correctDet_1']/float(count['allDet'])
        pre_25 = count['correctDet_25'] / float(count['allDet'])
        pre_50 = count['correctDet_50'] / float(count['allDet'])
        pre_75 = count['correctDet_75'] / float(count['allDet'])
        pre_100 = count['correctDet_100'] / float(count['allDet'])
        recall_1 = count['correctDet_1']/float(count['allGTbox'])
        recall_25 = count['correctDet_25'] / float(count['allGTbox'])
        recall_50 = count['correctDet_50'] / float(count['allGTbox'])
        recall_75 = count['correctDet_75'] / float(count['allGTbox'])
        recall_100 = count['correctDet_100'] / float(count['allGTbox'])
        if pre_1 == 0 and recall_1 == 0:
            f_1=0
        else:
            f_1=2*(pre_1*recall_1)/float(pre_1+recall_1)
        if pre_25 == 0 and recall_25 == 0:
            f_25=0
        else:
            f_25=2*(pre_25*recall_25)/float(pre_25+recall_25)
        if pre_50 == 0 and recall_50 == 0:
            f_50 = 0
        else:
            f_50 = 2*(pre_50*recall_50)/float(pre_50+recall_50)
        if pre_75 == 0 and recall_75 == 0:
            f_75 = 0
        else:
            f_75 = 2*(pre_75*recall_75)/float(pre_75+recall_75)
        if pre_100 == 0 and recall_100 == 0:
            f_100 = 0
        else:
            f_100 = 2*(pre_100*recall_100)/float(pre_100+recall_100)
    
        print('')
        print('**** result : threshold: 0.01 *****')
        print('  f =',f_1,'   precision =',pre_1,'   recall =',recall_1)
        print('')
        print('**** result : threshold: 0.25 *****')
        print('  f =',f_25,'   precision =',pre_25,'    recall =',recall_25)
        print('')
        print('**** result : threshold: 0.50 *****')
        print('  f =', f_50, '   precision =', pre_50, '    recall =', recall_50)
        print('')
        print('**** result : threshold: 0.75 *****')
        print('  f =', f_75, '   precision =', pre_75, '    recall =', recall_75)
        print('')
        print('**** result : threshold: 1.0 *****')
        print('  f =', f_100, '   precision =', pre_100, '    recall =', recall_100)
        
        scores={'f_1':round(f_1,4),'pre_1':round(pre_1,4),'rec_1':round(recall_1,4),
                'f_25': round(f_25, 4), 'pre_25': round(pre_25, 4), 'rec_25': round(recall_25, 4),
                'f_50': round(f_50, 4), 'pre_50': round(pre_50, 4), 'rec_50': round(recall_50, 4),
                'f_75': round(f_75, 4), 'pre_75': round(pre_75, 4), 'rec_75': round(recall_75, 4),
                'f_100': round(f_100, 4), 'pre_100': round(pre_100, 4), 'rec_100': round(recall_100, 4)}
    return scores
    

def IOUeval(ground_truth, detections, outdir=None):
    
    keys = ['allGTbox','correctDet_1','correctDet_25','correctDet_50','correctDet_75',
            'correctDet_100','allDet']
    info=dict.fromkeys(keys,0)
 
    gt_file_name = ground_truth
    det_file_name = detections

    #TODO : Mahshad  change it to user directory
    if outdir:
        #outdir='IOU_eval_stats'
        if os.path.exists(outdir):
            shutil.rmtree(outdir)
        os.makedirs(outdir)

    gt_pdfs_bboxes_map = create_doc_bboxes_map(gt_file_name,'gt')
    det_pdfs_bboxes_map = create_doc_bboxes_map(det_file_name,'det')
    #count boxes
    all_gtbox=count_box(gt_pdfs_bboxes_map)
    all_detbox=count_box(det_pdfs_bboxes_map)
    
    
    info['allGTbox']=all_gtbox
    info['allDet']=all_detbox
    
    pdf_gt_bbs = 0
    pdf_dt_bbs = 0
    pdf_info = {}
    pdf_calcs = {}

    detailed_detections = {}

    for pdf_name in gt_pdfs_bboxes_map:
        print(pdf_name)
        if pdf_name not in det_pdfs_bboxes_map:
            print('Detections not found for ',pdf_name)
            continue

        det_page_bboxes_map = det_pdfs_bboxes_map[pdf_name]
        gt_page_bboxes_map = gt_pdfs_bboxes_map[pdf_name]
        correct_pred_1, correct_pred_25, correct_pred_50, correct_pred_75, correct_pred_100, \
        pdf_gt_boxes, pdf_det_boxes, keys_1, keys_25, keys_50, keys_75, keys_100 =\
            IoU_page_bboxes(gt_page_bboxes_map, det_page_bboxes_map, pdf_name,outdir)
        info['correctDet_1']=info['correctDet_1']+correct_pred_1
        info['correctDet_25'] = info['correctDet_25'] + correct_pred_25
        info['correctDet_50'] = info['correctDet_50'] + correct_pred_50
        info['correctDet_75'] = info['correctDet_75'] + correct_pred_75
        info['correctDet_100'] = info['correctDet_100'] + correct_pred_100

        pdf_info['correctDet_1']=correct_pred_1
        pdf_info['correctDet_25'] = correct_pred_25
        pdf_info['correctDet_50'] = correct_pred_50
        pdf_info['correctDet_75'] = correct_pred_75
        pdf_info['correctDet_100'] = correct_pred_100
        pdf_info['allGTbox']=pdf_gt_boxes
        pdf_info['allDet']=pdf_det_boxes
        
        print('For pdf: ',  pdf_name)
        pdf_calcs[pdf_name]=pre_rec_calculate(pdf_info)
        detailed_detections[pdf_name] = [keys_1, keys_25, keys_50, keys_75, keys_100]
        #print('Pdf score:',pdf_name, " --> ", pre_rec_calculate(pdf_info))

    print('\n')    
    print(info)    
    scores=pre_rec_calculate(info)
    
    print('\n PDF Level \n')
    #print(pdf_calcs)
    
    #{'fine_f': 0.7843, 'fine_pre': 0.7774, 'fine_rec': 0.7914, 'coarse_f': 0.902, 'coarse_pre': 0.894, 'coarse_rec': 0.9101}
    for pdf_name in pdf_calcs:
        print(pdf_name,'\t', pdf_calcs[pdf_name]['f_1'],'\t',pdf_calcs[pdf_name]['f_25'],
              '\t',pdf_calcs[pdf_name]['f_50'], '\t',pdf_calcs[pdf_name]['f_75'],
              '\t',pdf_calcs[pdf_name]['f_100'])
        
    # TODO: Redundant after here...
    #return corase and fine F-scores
    return scores, detailed_detections
    

def write_final_csv(exp_nm, al_scores, save_dir='ssd/metrics'):
    header = [datetime.now(), exp_nm]
    fields = ['Conf', 'F_0.01', 'P_0.01', 'R_0.01', 'F_0.25', 'P_0.25', 'R_0.25',
              'F_0.5', 'P_0.5', 'R_0.5', 'F_0.75', 'P_0.75', 'R_0.75',
              'F_1.0', 'P_1.0', 'R_1.0']
    with open(os.path.join(save_dir,exp_nm+'_overall.csv'), 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerow(fields)
        for conf in sorted(al_scores.keys()):
            row_info = [conf, al_scores[conf]['f_1'], al_scores[conf]['pre_1'], al_scores[conf]['rec_1'],
                        al_scores[conf]['f_25'], al_scores[conf]['pre_25'], al_scores[conf]['rec_25'],
                        al_scores[conf]['f_50'], al_scores[conf]['pre_50'], al_scores[conf]['rec_50'],
                        al_scores[conf]['f_75'], al_scores[conf]['pre_75'], al_scores[conf]['rec_75'],
                        al_scores[conf]['f_100'], al_scores[conf]['pre_100'], al_scores[conf]['rec_100']]
            csvwriter.writerow(row_info)
    f.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, default=None, help="detections file path")
    parser.add_argument("--ground_truth", type=str, required=True, help="ground_truth file path")
    parser.add_argument("--det_folder", type=str, default=None,
                        help="In case of multiple confidences enter a folder")
    args = parser.parse_args()
    gt_file_name = args.ground_truth

    if args.det_folder is None:
        det_file_name = args.detections
        scores, det_dets = IOUeval(gt_file_name, det_file_name, outdir=None)
    else:
        all_scores = {}
        exp_name = args.det_folder.split('/')[-2]
        dir_names = [direc[0] for direc in os.walk(args.det_folder)][1:]
        for direc in dir_names:
            scores, det_dets = IOUeval(gt_file_name, direc, outdir=None)
            conf_lvl = direc.split('/')[-1].split('_')[-1]
            all_scores[conf_lvl] = scores
        write_final_csv(exp_name, all_scores)


