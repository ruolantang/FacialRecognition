#include <string>
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;


void cvText(IplImage* img,const char* text, int x, int y)
{
    CvFont font;

    double hscale = 0.5;
    double vscale = 0.5;
    int linewidth = 1;
    cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX |CV_FONT_ITALIC,hscale,vscale,0,linewidth);

    CvScalar textColor =cvScalar(0,255,255);
    CvPoint textPos =cvPoint(x, y);

    cvPutText(img, text, textPos, &font,textColor);
}


void mask(Mat& face, int eyes[])
{
    int dis = eyes[0] - eyes[2];
    int center[2];
    int width = dis * 2, height = dis * 2.5;
    int left, right, top, bottom;

    center[0] = (eyes[0] + eyes[2])/2;
    center[1] = (eyes[1] + eyes[3])/2;
    left = center[0] - width/2;
    right = center[0] + width/2;
    top = center[1] - 2*height/5;
    bottom = center[1] + 3*height/5;

    while(1)
    {
        if(left >= 0 && right < face.cols && top >= 0 && bottom < face.rows)
            break;
        left += 5;
        right -= 5;
        top += 7;
        bottom -=7;
    }

    Range rows, cols;
    rows.start = top, rows.end = bottom;
    cols.start = left, cols.end = right;
    face = Mat(face, rows, cols);
    resize(face, face, Size(100,125));
}


int main() {

    Mat trans,faces,testface;
    int eyes[4];

    string eigenfile = "eigenface";
    FileStorage fs(eigenfile, FileStorage::READ);
    fs["eigenface"]>>trans;
    fs["faces"]>>faces;

    Mat theface = imread("BioID_0005.pgm",CV_LOAD_IMAGE_GRAYSCALE);
    IplImage* ipl_face=cvLoadImage("BioID_0005.pgm");

    char eyeFile[30];
    sprintf(eyeFile, "BioID_0005.eye");

    fstream in;
    in.open(eyeFile, ios::in);

    string str;
    in>>str>>str>>str>>str>>eyes[0]>>eyes[1]>>eyes[2]>>eyes[3];
    mask(theface, eyes);
    in.close();

    int w = theface.rows;  //112
    int h = theface.cols;  //92
    int length = w*h;

    equalizeHist( theface, theface );
    theface.reshape(0,1).copyTo(testface);

    trans.convertTo(trans,CV_32F);
    theface.convertTo(theface,CV_32F);\
    faces.convertTo(faces,CV_32F);
    testface.convertTo(testface,CV_32F);


    Mat theface_zb;
    theface_zb = trans*testface.t();

//    cout<<faces.rows<<endl;
//    cout<<faces.cols<<endl;


        int index=0;
        float min=10;
        float jieguo[400];
        for(int i=0;i<faces.rows;i++){
            float chazhi=0;
            Mat facei = trans*faces.row(i).t();

            for(int j=0;j<theface_zb.rows;j++){

                chazhi+=(facei.at<float>(j,0)-theface_zb.at<float>(j,0))*(facei.at<float>(j,0)-theface_zb.at<float>(j,0));

            }
            if(chazhi<min){
                min = chazhi;
                index = i;
            }
            jieguo[i] = sqrt(chazhi);

           // cout<<chazhi<<endl;

        }

            Mat recface=faces.row(index);
            recface = recface.reshape(0, w);
            normalize(recface,recface, 0, 255, CV_MINMAX);
            recface.convertTo(recface, CV_8U);
            namedWindow("eigenface", 1);
            imshow("eigenface", recface);


        char indexstr[10];
        itoa(index,indexstr,16);
        string text = "the face number is ";
        //sprintf(indexstr, "%d", index);
        text.append(indexstr);
        const char *ch = text.c_str();



        cvText(ipl_face,ch,50,50);
//        Mat img2(&ipl_face,0);
        //namedWindow("theface", 2);
//        imshow("theface", img2);
        cvNamedWindow("2");
        cvShowImage("2",ipl_face);



    waitKey (0);
    return 0;
}
