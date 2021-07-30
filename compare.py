from subtraction import backgroundSubtraction
from cfgreader import config as config
import time
import shutil
import os
import redis
import cv2
import base64
from difflib import SequenceMatcher
import numpy as np
from datetime import datetime
import hitGAPI as LPNapi
import oracleConn as oracleDB
import dashboard_LP_rec as sendAPI
from shapely.geometry import Point, Polygon
import proto.Inference_pb2 as inference
import logging as logger

#for bbox in bbox_list
    #if bbox(x1) or bbox(x2) lies of forground 
        #save the bbox on redis
        #keep bbox or apply tracker or this bbox 
    #else
        #discard bbox

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def compare(client):
    start = time.time()
    timestamp = client.blpop(config.ocr_in_channel.encode(), 0)[1]
    if timestamp is None:
        time.sleep(0.05)
        # continue
    track_data = client.hget(timestamp, b'InferResults') 
    if track_data is None:
        time.sleep(0.05)
        # continue
    track_result = inference.DetectorResults()
    track_result.ParseFromString(track_data)

    ###########  getting frame from redis  ###############

    frame_data = client.hget(timestamp, b'Frame')
    if frame_data is None:
        time.sleep(1)
        # continue
    mat_data = base64.b64decode(frame_data)
    nparr = np.fromstring(mat_data, np.uint8)
    processFrame = cv2.imdecode(nparr, 1)

    ########### GETTING FOREGROUND ###################

    FGmASK = backgroundSubtraction(processFrame)

#SHORTEN BBOX
    finalBBox = []
    for i, bound in enumerate(track_result.bounds):
        d = [int(bound.x)+int(int(bound.width)*0.25), int(bound.y)+int(int(bound.height)*0.25), int(int(bound.width)*0.5), int(int(bound.height)*0.5)]
        fgmask_crop = crop_center(FGmASK,int(int(bound.width)),int(int(bound.height)))
        fgmask_crop_mean = fgmask_crop.mean()


        if fgmask_crop_mean >= config.threshold_compare_fg:
            passed = [int(bound.x), int(bound.y), int(bound.width), int(bound.height)]
            finalBBox.append(passed)
    
    for r in range(0, len(finalBBox)):
        bounding_box = track_result.bounds.add()
        bounding_box.x = finalBBox[r][0]
        bounding_box.y = finalBBox[r][1]
        bounding_box.width = finalBBox[r][2]
        bounding_box.height = finalBBox[r][3]
        # bounding_box.label = "group"
        # bounding_box.classid = 10
        # bounding_box.classname = "group"
        # bounding_box.trackid = 0.0

    # track_result.count.inCount=total_counts
    out=track_result.SerializeToString()
    client.hset(timestamp,b'InferResults',out)
    client.pexpire(timestamp,config.redis_hash_timeout)
    client.rpush(config.lbc_out_channel,timestamp)
    client.ltrim(config.lbc_out_channel,-4,-1)
    fps = round(1 / (time.time() - start), 1)
    logger.info("Bag Group Counting")
    logger.info("FPS")
    logger.info(str(fps))