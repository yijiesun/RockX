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

#define MAX_QUEUE_SIZE 20
#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        LOGD("%s :%f ms ",#tag, ____##tag##_total_time/1000.0);
typedef pair<int, Mat> imagePair;
class paircompbig {
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) return n1.first < n2.first;
        return n1.first < n2.first;
    }
};
class paircompless {
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const {
        if (n1.first == n2.first) return n1.first > n2.first;
        return n1.first > n2.first;
    }
};

cv::VideoCapture capture;
VideoWriter outputVideo;
V4L2 v4l2_;
unsigned int * pfb;
SCREEN screen_;
int IMG_WID;
int IMG_HGT;
cv::Mat frame;
bool use_camera_not_video,save_video,quit;
struct timeval show_img_time;
priority_queue<imagePair, vector<imagePair>, paircompless>queueFrameIn; 
priority_queue<imagePair, vector<imagePair>, paircompless> queueShow;
pthread_mutex_t  mutex_show;
pthread_mutex_t  mutex_frameIn;
pthread_mutex_t  mutex_quit;
rockx_ret_t ret;
rockx_handle_t object_det_handle;
rockx_handle_t object_track_handle;

inline int set_cpu(int i);
void my_handler(int s);
void *v4l2_thread(void *threadarg);
void *npu_thread(void *threadarg);
void *screen_thread(void *threadarg);

int main(int argc, char** argv) {
    IMG_HGT = 480;
    IMG_WID = 640;
    quit = false;
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    gettimeofday(&show_img_time, NULL);
    std::string in_video_file;
    std::string out_video_file;
    get_param_mssd_video_knn(in_video_file,out_video_file);
    LOGD("save  video:   %s",out_video_file.c_str());
    std::string dev_num;
    get_param_mms_V4L2(dev_num);
    LOGD("open  %s",dev_num.c_str());
    get_use_camera_or_video(use_camera_not_video);
    LOGD("use_camera_not_video  %d",use_camera_not_video);
    get_save_video(save_video);
    LOGD("save_video  %d",save_video);

    frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    frame = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);

    screen_.init((char *)"/dev/fb0",640,480);
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);

    if(use_camera_not_video)
    {
        capture.open(in_video_file.c_str());
        capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
    }
    else
    {
        v4l2_.init(dev_num.c_str(),640,480);
        v4l2_.open_device();
        v4l2_.init_device();
        v4l2_.start_capturing();
    }

    if(save_video)
    {
        Size sWH = Size( IMG_WID,IMG_HGT);
        outputVideo.open(out_video_file.c_str(), CV_FOURCC('M', 'J', 'P', 'G'), 25, sWH,true);
        if(!outputVideo.isOpened())
            LOGD("save video failed!");
    }

__TIC__(NPU_INIT);
    // create a object detection handle
    ret = rockx_create(&object_det_handle, ROCKX_MODULE_OBJECT_DETECTION, nullptr, 0);
    if (ret != ROCKX_RET_SUCCESS) {
        LOGD("init rockx module ROCKX_MODULE_OBJECT_DETECTION error %d\n", ret);
    }

    // create a object track handle
    ret = rockx_create(&object_track_handle, ROCKX_MODULE_OBJECT_TRACK, nullptr, 0);
    if (ret != ROCKX_RET_SUCCESS) {
        LOGD("init rockx module ROCKX_MODULE_OBJECT_DETECTION error %d\n", ret);
    }
__TOC__(NPU_INIT);

    pthread_t threads_npu;   
    pthread_t threads_v4l2;
    pthread_t threads_screen;
    pthread_create(&threads_npu, NULL, npu_thread, NULL);
    pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);
    pthread_create(&threads_screen, NULL,screen_thread, NULL);
    pthread_join(threads_npu,NULL);
    pthread_join(threads_v4l2,NULL);
    pthread_join(threads_screen,NULL);


    if(save_video)
        outputVideo.release();
            // release handle
    rockx_destroy(object_det_handle);
    rockx_destroy(object_track_handle);
    if(use_camera_not_video)
    {
        capture.release();
    }
    else
    {
        v4l2_.stop_capturing();
        v4l2_.uninit_device();
        v4l2_.close_device();
    }
    LOGD("exit success!");
    return 0;
}

inline int set_cpu(int i)  
{  
    cpu_set_t mask;  
    CPU_ZERO(&mask);  
  
    CPU_SET(i,&mask);  

    if(-1 == pthread_setaffinity_np(pthread_self() ,sizeof(mask),&mask))  
    {  
        fprintf(stderr, "pthread_setaffinity_np erro\n");  
        return -1;  
    }  
    return 0;  
} 

void *v4l2_thread(void *threadarg)
{
    set_cpu(3);
    while(1)
    {
        pthread_mutex_lock(&mutex_quit);
        if(quit)
        {
            pthread_mutex_unlock(&mutex_quit);
            break;
        }
        else
            pthread_mutex_unlock(&mutex_quit);

        __TIC__(CAMERA);
        if(use_camera_not_video)
        {
            if(!capture.read(frame))
            {
                pthread_mutex_lock(&mutex_quit);
                quit = true;
                pthread_mutex_unlock(&mutex_quit);
            }
        }
        else
        {
            v4l2_.read_frame(frame);
        }

        int time_ = getTimesInt();
        imagePair pframe(time_,frame);
        pthread_mutex_lock(&mutex_frameIn);
        if(queueFrameIn.size()<MAX_QUEUE_SIZE)
            queueFrameIn.push(pframe);
        pthread_mutex_unlock(&mutex_frameIn);
        usleep(10000); 
        __TOC__(CAMERA);
        
    }
    pthread_exit(NULL);
}

void *screen_thread(void *threadarg)
{
    set_cpu(4);

    while(1)
    {    
        pthread_mutex_lock(&mutex_quit);
        if(quit)
        {
            pthread_mutex_unlock(&mutex_quit);
            break;
        }
        else
            pthread_mutex_unlock(&mutex_quit);

        pthread_mutex_lock(&mutex_show);
         if(!queueShow.empty())
        {
            __TIC__(SHOW);
            pthread_mutex_unlock(&mutex_show);
            std::string fps_str;
            struct timeval t1;
            gettimeofday(&t1, NULL);
            float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (show_img_time.tv_sec * 1000000 + show_img_time.tv_usec)) / 1000.0;
            float fps = 1000.0/mytime;
            char buffer[32];
            std::sprintf(buffer, "%0.2f",fps);
            string fps_s = buffer;
            fps_str = "fps:"+fps_s;

            pthread_mutex_lock(&mutex_show);
            Mat show_img = queueShow.top().second;
            queueShow.pop();
            pthread_mutex_unlock(&mutex_show);

            Point siteNo;
            siteNo.x = 25;
            siteNo.y = 25;
            putText( show_img, fps_str, siteNo, 2,1,Scalar( 255, 0, 0 ), 4);
            v4l2_. mat_to_argb(show_img.data,pfb,IMG_WID,IMG_HGT,screen_.vinfo.xres_virtual,0,0);
            memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);

            if(save_video)
            {
                outputVideo.write(show_img);
            }
               
            show_img_time.tv_sec=0;
            show_img_time.tv_usec=0;
            gettimeofday(&show_img_time, NULL);
            __TOC__(SHOW);
        }
        else
        {
            pthread_mutex_unlock(&mutex_show);
        }
        usleep(10000); 
    }
    pthread_exit(NULL);
}

void *npu_thread(void *threadarg)
{
    set_cpu(5);

    rockx_image_t input_image;
    while(1)
    {
        pthread_mutex_lock(&mutex_quit);
        if(quit)
        {
            pthread_mutex_unlock(&mutex_quit);
            break;
        }
        else
            pthread_mutex_unlock(&mutex_quit);

        Mat img_roi;
        pthread_mutex_lock(&mutex_frameIn);
        if(!queueFrameIn.empty())
        {
            img_roi = queueFrameIn.top().second;
            queueFrameIn.pop();
        }
        else
        {
            pthread_mutex_unlock(&mutex_frameIn);
            continue;
        }
        pthread_mutex_unlock(&mutex_frameIn);

        __TIC__(NPU);
        input_image.width = IMG_WID;
        input_image.height = IMG_HGT;
        input_image.pixel_format = ROCKX_PIXEL_FORMAT_BGR888;
        input_image.data = img_roi.data;
        // rockx_image_read(imagepath, &input_image, 1);

        // create rockx_object_array_t for store result
        rockx_object_array_t object_array;
        memset(&object_array, 0, sizeof(rockx_object_array_t));

        // detect object
        ret = rockx_object_detect(object_det_handle, &input_image, &object_array, nullptr);
        if (ret != ROCKX_RET_SUCCESS) {
            LOGD("rockx_object_detect error %d\n", ret);
            pthread_mutex_lock(&mutex_quit);
            quit=true;
            pthread_mutex_unlock(&mutex_quit);
            break;
        }

        // object track
        int max_track_time = 10;
        rockx_object_array_t in_track_objects;
        rockx_object_array_t out_track_objects;

        ret = rockx_object_track(object_track_handle, input_image.width,  input_image.height, max_track_time,
        &object_array, &out_track_objects);
        if (ret != ROCKX_RET_SUCCESS) {
            LOGD("rockx_object_track error %d\n", ret);
            pthread_mutex_lock(&mutex_quit);
            quit=true;
            pthread_mutex_unlock(&mutex_quit);
            break;
        }

        // process result
        for (int i = 0; i < out_track_objects.count; i++) {
            int left = out_track_objects.object[i].box.left;
            int top = out_track_objects.object[i].box.top;
            int right = out_track_objects.object[i].box.right;
            int bottom = out_track_objects.object[i].box.bottom;
            int cls_idx = out_track_objects.object[i].cls_idx;
            if(cls_idx!=1)
                continue;
            const char *cls_name = OBJECT_DETECTION_LABELS_91[cls_idx];
            float score = out_track_objects.object[i].score;
            int track_id = out_track_objects.object[i].id;
            LOGD("box=(%d %d %d %d) cls_name=%s, score=%f track_id=%d\n", left, top, right, bottom,
                cls_name, score, track_id);
            char show_str[32];
            memset(show_str, 0, 32);
            snprintf(show_str, 32, "%0.2f", score);
            // draw
            rockx_image_draw_rect(&input_image, {left, top}, {right, bottom}, {0, 255, 0}, 2);
            rockx_image_draw_text(&input_image, show_str, {left, top-8}, {0, 255, 0}, 1);
        }


        Mat show_img(input_image.height,input_image.width,CV_8UC3);
        memcpy(show_img.data,input_image.data,input_image.height*input_image.width*3*sizeof(uchar));
        int time_ = getTimesInt();
        imagePair pframe(time_,show_img);
        pthread_mutex_lock(&mutex_show);
        queueShow.push(pframe);
        pthread_mutex_unlock(&mutex_show);

        __TOC__(NPU);

        // sleep(0.04); 
    }

    rockx_image_release(&input_image);
    pthread_exit(NULL);
}

void my_handler(int s)
{
            pthread_mutex_lock(&mutex_quit);
            quit=true;
            pthread_mutex_unlock(&mutex_quit);
            LOGD("Caught signal  %d  quit= %d",s,quit);
}