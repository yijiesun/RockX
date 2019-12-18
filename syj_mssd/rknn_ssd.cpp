#if 0 
#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include "rknn_api.h"
#include <sys/time.h>
#include <stdio.h>
#include "config.h"
#include "knn/knn.h"
#include "v4l2/v4l2.h"  
#include "screen/screen.h"
#include <stdint.h>
#include <stdlib.h>
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
// using namespace std::chrono;
#define INPUT_VIDEO 1
#define SAVE_VIDEO 1
#define DEF_PROTO "models/MobileNetSSD_deploy.prototxt"
// #define DEF_MODEL "models/MobileNetSSD_deploy.caffemodel"
#define DEF_MODEL "models/MobileNetSSD_deploy.tmfile"
#define CLIP(a,b,c) (  (a) = (a)>(c)?(c):((a)<(b)?(b):(a))  )
#define CPU_THREAD_CNT 3 //a53-012         a72-4          a72-5
#define GPU_THREAD_CNT 2 

#define QUEUE_SIZE 2
#define NUM_RESULTS         1917
#define NUM_CLASSES         91
#define Y_SCALE  10.0f
#define X_SCALE  10.0f
#define H_SCALE  5.0f
#define W_SCALE  5.0f

#define __TIC__(tag) timeval ____##tag##_start_time, ____##tag##_end_time;\
        gettimeofday(&____##tag##_start_time, 0);

#define __TOC__(tag) gettimeofday(&____##tag##_end_time, 0); \
        int ____##tag##_total_time=((int)____##tag##_end_time.tv_sec-(int)____##tag##_start_time.tv_sec)*1000000+((int)____##tag##_end_time.tv_usec-(int)____##tag##_start_time.tv_usec); \
        LOGD("%s :%f ms ",#tag, ____##tag##_total_time/1000.0);

const char *img_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/road.bmp";
const char *img_path1 = "/home/firefly/RKNN/rknn-api/Linux/tmp/dog.jpg";
const char *model_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/mobilenet_ssd.rknn";
const char *label_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/coco_labels_list.txt";
const char *box_priors_path = "/home/firefly/RKNN/rknn-api/Linux/tmp/box_priors.txt";

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
struct timeval show_img_time;
bool npu_calc[2]={false,false};
int frame_cnt;
cv::VideoCapture capture;
VideoWriter outputVideo;
V4L2 v4l2_;
unsigned int * pfb;
SCREEN screen_;
pthread_mutex_t  mutex_show;
pthread_mutex_t  mutex_frameIn;
vector<Box>	boxes; 
vector<BOX_COLOR>	boxes_all; 
cv::Mat frame;

int IMG_WID;
int IMG_HGT;
int img_h;
int img_w;
int img_size;
bool quit;
bool is_show_img;
bool is_show_knn_box;
rknn_context rknn_ctx;
float* input_data;
string labels[91];
priority_queue<imagePair, vector<imagePair>, paircompless>
        queueFrameIn; 
priority_queue<imagePair, vector<imagePair>, paircompless> queueShow;


Scalar colorArray[10] = {
        Scalar(139,   0,   0, 255),
        Scalar(139,   0, 139, 255),
        Scalar(  0,   0, 139, 255),
        Scalar(  0, 100,   0, 255),
        Scalar(139, 139,   0, 255),
        Scalar(209, 206,   0, 255),
        Scalar(  0, 127, 255, 255),
        Scalar(139,  61,  72, 255),
        Scalar(  0, 255,   0, 255),
        Scalar(255,   0,   0, 255),
};

float MIN_SCORE = 0.3f;

float NMS_THRESHOLD = 1.0f;

int multi_npu_process_initialized[2] = {0, 0};

void *v4l2_thread(void *threadarg);

void my_handler(int s);

inline pid_t gettid() 
{
  return syscall(__NR_gettid);
}

int loadLabelName(string locationFilename, string* labels) {
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    while(getline(fin, line))
    {
        labels[lineNum] = line;
        lineNum++;
    }
    return 0;
}

int loadCoderOptions(string locationFilename, float (*boxPriors)[NUM_RESULTS])
{
    const char *d = ", ";
    ifstream fin(locationFilename);
    string line;
    int lineNum = 0;
    while(getline(fin, line))
    {
        char *line_str = const_cast<char *>(line.c_str());
        char *p;
        p = strtok(line_str, d);
        int priorIndex = 0;
        while (p) {
            float number = static_cast<float>(atof(p));
            boxPriors[lineNum][priorIndex++] = number;
            p=strtok(nullptr, d);
        }
        if (priorIndex != NUM_RESULTS) {
            return -1;
        }
        lineNum++;
    }
    return 0;

}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = max(0.f, min(xmax0, xmax1) - max(xmin0, xmin1));
    float h = max(0.f, min(ymax0, ymax1) - max(ymin0, ymin1));
    float i = w * h;
    float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;
    return u <= 0.f ? 0.f : (i / u);
}

float expit(float x) {
    return (float) (1.0 / (1.0 + exp(-x)));
}

void decodeCenterSizeBoxes(float* predictions, float (*boxPriors)[NUM_RESULTS]) {

    for (int i = 0; i < NUM_RESULTS; ++i) {
        float ycenter = predictions[i*4+0] / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
        float xcenter = predictions[i*4+1] / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
        float h = (float) exp(predictions[i*4 + 2] / H_SCALE) * boxPriors[2][i];
        float w = (float) exp(predictions[i*4 + 3] / W_SCALE) * boxPriors[3][i];

        float ymin = ycenter - h / 2.0f;
        float xmin = xcenter - w / 2.0f;
        float ymax = ycenter + h / 2.0f;
        float xmax = xcenter + w / 2.0f;

        predictions[i*4 + 0] = ymin;
        predictions[i*4 + 1] = xmin;
        predictions[i*4 + 2] = ymax;
        predictions[i*4 + 3] = xmax;
    }
}

int scaleToInputSize(float * outputClasses, int (*output)[NUM_RESULTS], int numClasses)
{
    int validCount = 0;
    // Scale them back to the input size.
    for (int i = 0; i < NUM_RESULTS; ++i) {
        float topClassScore = static_cast<float>(-1000.0);
        int topClassScoreIndex = -1;

        // Skip the first catch-all class.
        for (int j = 1; j < numClasses; ++j) {
            float score = expit(outputClasses[i*numClasses+j]);
            if (score > topClassScore) {
                topClassScoreIndex = j;
                topClassScore = score;
            }
        }

        if (topClassScoreIndex==1&&topClassScore >= MIN_SCORE) {
            output[0][validCount] = i;
            // output[1][validCount] = topClassScoreIndex;
            output[1][validCount] = static_cast< int >(topClassScore*100.0);
            ++validCount;
        }
        
    }

    return validCount;
}

int nms(int validCount, float* outputLocations, int (*output)[NUM_RESULTS])
{
    for (int i=0; i < validCount; ++i) {
        if (output[0][i] == -1) {
            continue;
        }
        int n = output[0][i];
        for (int j=i + 1; j<validCount; ++j) {
            int m = output[0][j];
            if (m == -1) {
                continue;
            }
            float xmin0 = outputLocations[n*4 + 1];
            float ymin0 = outputLocations[n*4 + 0];
            float xmax0 = outputLocations[n*4 + 3];
            float ymax0 = outputLocations[n*4 + 2];

            float xmin1 = outputLocations[m*4 + 1];
            float ymin1 = outputLocations[m*4 + 0];
            float xmax1 = outputLocations[m*4 + 3];
            float ymax1 = outputLocations[m*4 + 2];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou >= NMS_THRESHOLD) {
                output[0][j] = -1;
            }
        }
    }

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

void draw_img(Mat &img,vector<BOX_COLOR> & boxa_)
{
    int line_width=300*0.002;
    
    for(int i=0;i<(int)boxa_.size();i++)
    {
        Box box=boxa_[i].box;
        char buffer[32];
        std::sprintf(buffer, "%0.2f",box.score);
        if(boxa_[i].npu_td_num==0)
        {
            cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(0, 0, 255),line_width);
            cv::putText(img, buffer, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        }
        else
        {
            cv::rectangle(img, cv::Rect(box.x0, box.y0,(box.x1-box.x0),(box.y1-box.y0)),cv::Scalar(0, 255, 0),line_width);
            cv::putText(img, buffer, cv::Point(box.x0, box.y0),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
        }

    }

}
float DecideOverlap(const Rect &r1, const Rect &r2, Rect &r3)
{
	//r3 = r1;
	int x1 = r1.x;
	int y1 = r1.y;
	int width1 = r1.width;
	int height1 = r1.height;

	int x2 = r2.x;
	int y2 = r2.y;
	int width2 = r2.width;
	int height2 = r2.height;

	int endx = max(x1 + width1, x2 + width2);
	int startx = min(x1, x2);
	int width = width1 + width2 - (endx - startx);

	int endy = max(y1 + height1, y2 + height2);
	int starty = min(y1, y2);
	int height = height1 + height2 - (endy - starty);

	float ratio = 0.0f;
	float Area, Area1, Area2;

	if (width <= 0 || height <= 0)
		return 0.0f;
	else
	{
		Area = width*height;
		Area1 = width1*height1;
		Area2 = width2*height2;
		ratio = max(Area / (float)Area1, Area / (float)Area2);
		r3.x = startx;
		r3.y = starty;
		r3.width = endx - startx;
		r3.height = endy - starty;
	}

	return ratio;
}

void togetherAllBox(int th_num,vector<Box> &b0,vector<BOX_COLOR> &b_all)
{
    for (int i = 0; i<b0.size(); i++) {
        BOX_COLOR bc;
        bc.box = b0[i];
        bc.npu_td_num = th_num;
        b_all.push_back(bc);
    }
}
void get_input_data_ssd(Mat& img, float* input_data, int img_h, int img_w)
{
    cv::resize(img, img, cv::Size(img_h, img_w));
    img.convertTo(img, CV_32FC3);
    float* img_data = ( float* )img.data;
    int hw = img_h * img_w;

    float mean[3] = {127.5, 127.5, 127.5};
    for(int h = 0; h < img_h; h++)
    {
        for(int w = 0; w < img_w; w++)
        {
            for(int c = 0; c < 3; c++)
            {
                input_data[c * hw + h * img_w + w] = 0.007843 * (*img_data - mean[c]);
                img_data++;
            }
        }
    }
}

void post_process_ssd(vector<Box> & box_,int thread_num,int raw_h,int raw_w, float* pred, int outp[2][NUM_RESULTS],int num)
{
    box_.clear();
    LOGD("[%d] detect ruesult num: %d",thread_num,num);
    for (int i=0;i<num;i++)
    {
        if(outp[0][i] != -1/*&&outp[1][i]==1*/)
        {
            int n = outp[0][i];
            Box box;
            // box.class_idx=outp[1][i];
            box.score=(float)outp[1][i]/100.0;
            box.x0=pred[n * 4 + 1]*raw_w;
            box.y0=pred[n * 4 + 0]*raw_h;
            box.x1=pred[n * 4 + 3]*raw_w;
            box.y1=pred[n * 4 + 2]*raw_h;
            CLIP(box.x0,0,IMG_WID-1);
            CLIP(box.y0,0,IMG_HGT-1);
            CLIP(box.x1,0,IMG_WID-1);
            CLIP(box.y1,0,IMG_HGT-1);
            box_.push_back(box);
            LOGD("[%d]  %.0f%%",thread_num, box.score * 100.0);
            LOGD("[%d]  BOX:( %g , %g ),( %g , %g )",thread_num,box.x0,box.y0,box.x1,box.y1);

        }
    }
}

void mssd_core(rknn_context &ctx, int thread_num )
{

    struct timeval t0, t1;
    float total_time = 0.f;
    gettimeofday(&t0, NULL);
    
    Mat	img_roi;
    Mat show_img;
    imagePair ip;


    pthread_mutex_lock(&mutex_frameIn);
    if(!queueFrameIn.empty())
    {
        img_roi = queueFrameIn.top().second;
        show_img = img_roi.clone();
        queueFrameIn.pop();
    }
    else
    {
        pthread_mutex_unlock(&mutex_frameIn);
        return;
    }
    pthread_mutex_unlock(&mutex_frameIn);

    
    int raw_h = img_roi.size().height;
    int raw_w = img_roi.size().width;
    if(raw_h==0|| raw_w==0)
        return;
        // Start Inference
    const int img_width = 300;
    const int img_height = 300;
    const int img_channels = 3;

    int ret =0;
    const int input_index = 0;
    rknn_input inputs[1];
    rknn_output outputs[2];
    rknn_tensor_attr outputs_attr[2];
    outputs_attr[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[0]), sizeof(outputs_attr[0]));
    if(ret < 0) {
        LOGD("rknn_query fail! ret=%d",ret);
        // goto Error;
    }

    outputs_attr[1].index = 1;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(outputs_attr[1]), sizeof(outputs_attr[1]));
    if(ret < 0) {
        LOGD("rknn_query fail! ret=%d",ret);
        // goto Error;
    }
    LOGD("img_roi.size %dX%d",img_roi.size().width,img_roi.size().height);
    cv::resize(img_roi, img_roi, cv::Size(img_width, img_height), (0, 0), (0, 0), cv::INTER_LINEAR);
    inputs[0].index = input_index;
    inputs[0].buf = img_roi.data;
    inputs[0].size = img_width * img_height * img_channels;
    inputs[0].pass_through = false;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    ret = rknn_inputs_set(ctx, 1, inputs);
    if(ret < 0) {
        LOGD("rknn_input_set fail! ret=%d",ret);
        // goto Error;
    }

    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        LOGD("rknn_run fail! fail! ret=%d",ret);
        // goto Error;
    }

    outputs[0].want_float = true;
    outputs[0].is_prealloc = false;
    outputs[1].want_float = true;
    outputs[1].is_prealloc = false;
    ret = rknn_outputs_get(ctx, 2, outputs, nullptr);
    if(ret < 0) {
        LOGD("rknn_outputs_get fail! ret=%d",ret);
        // goto Error;
    }

        // Process output
    if(outputs[0].size == outputs_attr[0].n_elems*sizeof(float) && outputs[1].size == outputs_attr[1].n_elems*sizeof(float))
    {
        float boxPriors[4][NUM_RESULTS];
        loadCoderOptions(box_priors_path, boxPriors);

        float* predictions = (float*)outputs[0].buf;
        float* outputClasses = (float*)outputs[1].buf;

        int output[2][NUM_RESULTS];

        /* transform */
        decodeCenterSizeBoxes(predictions, boxPriors);

        int validCount = scaleToInputSize(outputClasses, output, NUM_CLASSES);

            /* detect nest box */
        nms(validCount, predictions, output);
        post_process_ssd(boxes,thread_num,raw_h,raw_w,predictions,output,validCount);

        togetherAllBox(thread_num, boxes, boxes_all);
        draw_img(show_img,boxes_all);
        boxes_all.clear();
        pthread_mutex_lock(&mutex_show);
        int time_ = getTimesInt();
        imagePair pframe(time_,show_img);
        queueShow.push(pframe);
        pthread_mutex_unlock(&mutex_show);
    }
    rknn_outputs_release(ctx, 2, outputs);
    gettimeofday(&t1, NULL);
    float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (t0.tv_sec * 1000000 + t0.tv_usec)) / 1000;

    LOGD(" NPU times %f  ms",mytime);
}

void *npu_pthread(void *threadarg)
{
  set_cpu(5);
  while(1)
  {
    if(quit)
        pthread_exit(NULL);
    mssd_core(rknn_ctx, 0);
    usleep(100);
  }

}

int main(int argc, char* argv[])
{
    set_cpu(0);
    struct sigaction sigIntHandler;
   sigIntHandler.sa_handler = my_handler;
   sigemptyset(&sigIntHandler.sa_mask);
   sigIntHandler.sa_flags = 0;
   sigaction(SIGINT, &sigIntHandler, NULL);

    gettimeofday(&show_img_time, NULL);
    frame_cnt = 0;
    quit = false;
    int first_cnt =0;
    std::string in_video_file;
    std::string out_video_file;
    get_param_mssd_video_knn(in_video_file,out_video_file);
    LOGD("save  video:   %s",out_video_file.c_str());
    std::string dev_num;
    get_param_mms_V4L2(dev_num);
    LOGD("open  %s",dev_num.c_str());
    int knn_box_exist_cnt=2;
    // get_knn_box_exist_cnt(knn_box_exist_cnt);
    LOGD("knn_box_exist_cnt  %d",knn_box_exist_cnt);
    // double knn_thresh;
    // get_knn_thresh(knn_thresh);
    // std::cout<<"knn_thresh: "<<knn_thresh<<std::endl;
    get_show_img(is_show_img);
    LOGD("is_show_img  %d",is_show_img);
    get_show_knn_box(is_show_knn_box);
    LOGD("is_show_knn_box  %d",is_show_knn_box);

    IMG_WID = 640;
    IMG_HGT = 480;

#if SAVE_VIDEO
    Size sWH = Size( 2*IMG_WID,2*IMG_HGT);
    outputVideo.open(out_video_file.c_str(), cv::VideoWriter::fourcc ('M', 'P', '4', '2'), 25, sWH);
#endif
    frame.create(IMG_HGT,IMG_WID,CV_8UC3);
    frame = Mat::zeros(IMG_HGT,IMG_WID,CV_8UC3);

    screen_.init((char *)"/dev/fb0",640,480);
    pfb = (unsigned int *)malloc(screen_.finfo.smem_len);
    v4l2_.init(dev_num.c_str(),640,480);
#if INPUT_VIDEO
    capture.open(in_video_file.c_str());
    capture.set(CV_CAP_PROP_FOURCC, cv::VideoWriter::fourcc ('M', 'J', 'P', 'G'));
#else
    v4l2_.open_device();
    v4l2_.init_device();
    v4l2_.start_capturing();
#endif

    LOGD("V4L2 init bk done!");
    // Load model
    loadLabelName(label_path, labels);
    FILE *fp = fopen(model_path, "rb");
    if(fp == NULL) {
        LOGD("fopen %s fail!",model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    void *model = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        LOGD("fread %s fail!",model_path);
        free(model);
        return -1;
    }
    fclose(fp);
    int ret = 0;

    rknn_ctx = 0;
    ret = rknn_init(&rknn_ctx, model, model_len, RKNN_FLAG_PRIOR_MEDIUM);
    
    if(ret < 0) {
        LOGD("rknn_init fail! ret=%d",ret);
        // goto Error;
    }
    LOGD("init NPU  done!");
    

    pthread_t threads_v4l2;    
    pthread_t threads_npu;   
    pthread_create(&threads_v4l2, NULL, v4l2_thread, NULL);
    pthread_create(&threads_npu, NULL,npu_pthread, NULL);
    pthread_join(threads_v4l2,NULL);
    pthread_join(threads_npu,NULL);

    rknn_destroy(rknn_ctx);
    free(input_data);
    
    free(model);
    free(pfb);
    LOGD("mssd quit success!");
    return 0;
}


void *v4l2_thread(void *threadarg)
{
    set_cpu(1);
	while (1)
	{
        if(is_show_img)
        {
            if(is_show_knn_box)
            {
                pthread_mutex_lock(&mutex_frameIn);
                if(queueFrameIn.empty())
                {
                  pthread_mutex_unlock(&mutex_frameIn);
                  __TIC__(v4l2_thread);
                  frame_cnt++;
#if INPUT_VIDEO
                  if(!capture.read(frame))
                        quit = true;
#else
                  v4l2_.read_frame(frame);
#endif

                  int time_ = getTimesInt();
                  imagePair pframe(time_,frame);
                  pthread_mutex_lock(&mutex_frameIn);
                  queueFrameIn.push(pframe);
                  pthread_mutex_unlock(&mutex_frameIn);
                  __TOC__(v4l2_thread);
                }
                else
                {
                  pthread_mutex_unlock(&mutex_frameIn);
                }
                

                pthread_mutex_lock(&mutex_show);
                if(!queueShow.empty())
                {
                  __TIC__(v4l2_show);
                  Mat show_img = queueShow.top().second;
                  queueShow.pop();
                  pthread_mutex_unlock(&mutex_show);

                  
                  std::string fps_str;
                  struct timeval t1;
                  gettimeofday(&t1, NULL);
                  LOGD("t1.tv_sec%d",t1.tv_sec);
                  LOGD("t1.tv_usec%d",t1.tv_usec);
                  LOGD("show_img_time.tv_sec%d",show_img_time.tv_sec);
                  LOGD("show_img_time.tv_usec%d",show_img_time.tv_usec);
                  float mytime = ( float )((t1.tv_sec * 1000000 + t1.tv_usec) - (show_img_time.tv_sec * 1000000 + show_img_time.tv_usec)) / 1000.0;
                  float fps = 1000.0/mytime;
                  char buffer[32];
                  std::sprintf(buffer, "%0.2f",fps);
                  string fps_s = buffer;
                  fps_str = "fps:"+fps_s;

                  // string no=to_string(frame_cnt);
                  Point siteNo;
                  siteNo.x = 25;
                  siteNo.y = 25;
                  putText( show_img, fps_str, siteNo, 2,1,Scalar( 255, 0, 0 ), 4);
#if SAVE_VIDEO
                  outputVideo.write(show_img);
#endif
            
                  v4l2_. mat_to_argb(show_img.data,pfb,640,480,screen_.vinfo.xres_virtual,0,0);
                  memcpy(screen_.pfb,pfb,screen_.finfo.smem_len);

                  __TOC__(v4l2_show);
                  LOGD("%s",fps_str.c_str());
                show_img_time.tv_sec=0;
                show_img_time.tv_usec=0;
                gettimeofday(&show_img_time, NULL);
                pthread_mutex_lock(&mutex_frameIn);
                LOGD("queueFrameIn.size() %d",queueFrameIn.size());
                pthread_mutex_unlock(&mutex_frameIn);
                LOGD("queueShow.size() %d",queueShow.size());
                }
                else
                {
                  pthread_mutex_unlock(&mutex_show);
                }
                
            }
        }

        sleep(0.01); 
        if (quit)
        {
#if SAVE_VIDEO
            outputVideo.release();
#endif
#if INPUT_VIDEO
            capture.release();
#else
            v4l2_.stop_capturing();
            v4l2_.uninit_device();
            v4l2_.close_device();
#endif

            break;
        }
        
    }
}

void my_handler(int s)
{
            quit = true;
            LOGD("Caught signal  %d  quit= %d",s,quit);
}
#endif