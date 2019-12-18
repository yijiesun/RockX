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

#include "rockx.h"

int main(int argc, char** argv) {

    rockx_ret_t ret;
    rockx_handle_t face_det_handle;
    struct timeval tv;

    const char *img_path = argv[1];

    // create a face detection handle
    ret = rockx_create(&face_det_handle, ROCKX_MODULE_FACE_DETECTION, nullptr, 0);
    if (ret != ROCKX_RET_SUCCESS) {
        printf("init rockx module ROCKX_MODULE_FACE_DETECTION error %d\n", ret);
    }

    // read image
    rockx_image_t input_image;
    rockx_image_read(img_path, &input_image, 1);

    // create rockx_face_array_t for store result
    rockx_object_array_t face_array;
    memset(&face_array, 0, sizeof(rockx_object_array_t));

    // detect face
    ret = rockx_face_detect(face_det_handle, &input_image, &face_array, nullptr);
    if (ret != ROCKX_RET_SUCCESS) {
        printf("rockx_face_detect error %d\n", ret);
        return -1;
    }

    // process result
    for (int i = 0; i < face_array.count; i++) {
        int left = face_array.object[i].box.left;
        int top = face_array.object[i].box.top;
        int right = face_array.object[i].box.right;
        int bottom = face_array.object[i].box.bottom;
        float score = face_array.object[i].score;
        printf("box=(%d %d %d %d) score=%f\n", left, top, right, bottom, score);
        // draw
        char score_str[8];
        memset(score_str, 0, 8);
        snprintf(score_str, 8, "%.3f", score);
        rockx_image_draw_rect(&input_image, {left, top}, {right, bottom}, {255, 0, 0}, 3);
        rockx_image_draw_text(&input_image, score_str, {left, top-12}, {255, 0, 0}, 3);
    }

    // save image
    rockx_image_write("./out.jpg", &input_image);

    // release
    rockx_image_release(&input_image);
    rockx_destroy(face_det_handle);
}
