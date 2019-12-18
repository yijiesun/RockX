/****************************************************************************
*
*    Copyright (c) 2017 - 2019 by Rockchip Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Rockchip Corporation. This is proprietary information owned by
*    Rockchip Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Rockchip Corporation.
*
*****************************************************************************/
#include <stdio.h>
#include <memory.h>
#include <sys/time.h>
#include <stdlib.h>

#include "rockx.h"

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "../src/config.h"
#include "../src/v4l2/v4l2.h"  
#include "../src/screen/screen.h"
#include <stdint.h>
#include <fstream>
#include <atomic>
#include <queue>
#include<deque>
#include <thread>
#include <mutex>
#include <chrono>
#include <sys/stat.h>
#include <dirent.h>
#include <signal.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <sys/syscall.h>

using namespace std;
using namespace cv;

#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        LOGD("%s :%f ms ",#tag, ____##tag##_total_time/1000.0);

cv::VideoCapture capture;
VideoWriter outputVideo;
V4L2 v4l2_;
unsigned int * pfb;
SCREEN screen_;
int IMG_WID;
int IMG_HGT;
cv::Mat frame;

int main(int argc, char** argv) {
    std::string in_video_file;
    std::string out_video_file;
    get_param_mssd_video_knn(in_video_file,out_video_file);
    LOGD("save  video:   %s",out_video_file.c_str());

    Size sWH = Size( 2*IMG_WID,2*IMG_HGT);
    outputVideo.open(out_video_file.c_str(), cv::VideoWriter::fourcc ('M', 'P', '4', '2'), 25, sWH);
    frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    frame = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);

    screen_.init((char *)"/dev/fb0",640,480);
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);
    // v4l2_.init(dev_num.c_str(),640,480);

    capture.open(in_video_file.c_str());
    capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));

    rockx_ret_t ret;
    rockx_handle_t object_det_handle;
    rockx_handle_t object_track_handle;
    struct timeval tv;

    const char *img_path = argv[1];
__TIC__(INIT);
    // create a object detection handle
    ret = rockx_create(&object_det_handle, ROCKX_MODULE_OBJECT_DETECTION, nullptr, 0);
    if (ret != ROCKX_RET_SUCCESS) {
        printf("init rockx module ROCKX_MODULE_OBJECT_DETECTION error %d\n", ret);
    }

    // create a object track handle
    ret = rockx_create(&object_track_handle, ROCKX_MODULE_OBJECT_TRACK, nullptr, 0);
    if (ret != ROCKX_RET_SUCCESS) {
        printf("init rockx module ROCKX_MODULE_OBJECT_DETECTION error %d\n", ret);
    }
__TOC__(INIT);

    rockx_image_t input_image;

        while (1)
        {
            if(!capture.read(frame))
                break;
            // read image
            
            input_image.width = 640;
            input_image.height = 480;
            input_image.pixel_format = ROCKX_PIXEL_FORMAT_BGR888;
            input_image.data = frame.data;
            // rockx_image_read(imagepath, &input_image, 1);

            // create rockx_object_array_t for store result
            rockx_object_array_t object_array;
            memset(&object_array, 0, sizeof(rockx_object_array_t));
            __TIC__(mssd);
            // detect object
            ret = rockx_object_detect(object_det_handle, &input_image, &object_array, nullptr);
            if (ret != ROCKX_RET_SUCCESS) {
                printf("rockx_object_detect error %d\n", ret);
                return -1;
            }

            // object track
            int max_track_time = 10;
            rockx_object_array_t in_track_objects;
            rockx_object_array_t out_track_objects;

            ret = rockx_object_track(object_track_handle, input_image.width,  input_image.height, max_track_time,
                    &object_array, &out_track_objects);
            if (ret != ROCKX_RET_SUCCESS) {
                printf("rockx_object_track error %d\n", ret);
                return -1;
            }

            // process result
            for (int i = 0; i < out_track_objects.count; i++) {
                int left = out_track_objects.object[i].box.left;
                int top = out_track_objects.object[i].box.top;
                int right = out_track_objects.object[i].box.right;
                int bottom = out_track_objects.object[i].box.bottom;
                int cls_idx = out_track_objects.object[i].cls_idx;
                const char *cls_name = OBJECT_DETECTION_LABELS_91[cls_idx];
                float score = out_track_objects.object[i].score;
                int track_id = out_track_objects.object[i].id;
                printf("box=(%d %d %d %d) cls_name=%s, score=%f track_id=%d\n", left, top, right, bottom,
                        cls_name, score, track_id);
                char show_str[32];
                memset(show_str, 0, 32);
                snprintf(show_str, 32, "%d - %s", track_id, cls_name);
                // draw
                rockx_image_draw_rect(&input_image, {left, top}, {right, bottom}, {255, 0, 0}, 2);
                rockx_image_draw_text(&input_image, show_str, {left, top-8}, {255, 0, 0}, 2);
            }
            __TOC__(mssd);
            v4l2_. mat_to_argb(input_image.data,pfb,640,480,screen_.vinfo.xres_virtual,0,0);
            memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);
            sleep(0.01); 

            // save image
            // char output_name[128];
            // snprintf(output_name, 128, "./out/%s", imagepath);
            // rockx_image_write(output_name, &input_image);

            

            //process every image
            // printf("%s\n", output_name);
        }
    rockx_image_release(&input_image);

    // release handle
    rockx_destroy(object_det_handle);
    rockx_destroy(object_track_handle);
}
