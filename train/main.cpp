#include <string>
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;



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



int main(int argc, char* argv[]) {
    //Mat theface = imread("F:\\particle\\face\\att_faces\\s12\\1.pgm",CV_LOAD_IMAGE_GRAYSCALE);
    int w = 125;  //112
    int h = 100;  //92
    int length = w*h;
    int eyes[4];
    Mat faces(400, length, CV_32F);
    fstream in;

    //string dir="F:\\particle\\face\\att_faces\\s";
    Mat img;

    cout<<"loading"<<endl;
    for (int i = 0 ; i < 400; i++)
    {
        char filename[30], eyeFile[30];
        if(i < 10)
        {
            sprintf(filename, "face/BioID_000%d.pgm", i);
            sprintf(eyeFile, "face/BioID_000%d.eye", i);
        }
        else if(i < 100)
        {
            sprintf(filename, "face/BioID_00%d.pgm", i);
            sprintf(eyeFile, "face/BioID_00%d.eye", i);
        }
        else
        {
            sprintf(filename, "face/BioID_0%d.pgm", i);
            sprintf(eyeFile, "face/BioID_0%d.eye", i);
        }
        img = imread(filename,CV_LOAD_IMAGE_GRAYSCALE);


        in.open(eyeFile, ios::in);
        string str;
        in>>str>>str>>str>>str>>eyes[0]>>eyes[1]>>eyes[2]>>eyes[3];
        mask(img, eyes);
        in.close();

        //cout<<img.rows<<endl;

        Mat dstrow = faces.row(i);
        equalizeHist( img, img );
        img.convertTo(img, CV_32F);
        img.reshape(0,1).copyTo(dstrow);

    }
    cout<<"all masked"<<endl;

    Mat meanMat1;
    reduce(faces,meanMat1,0,CV_REDUCE_AVG);
    //Mat meanMat2=meanMat1.reshape(0,112);


    Mat subFaces(400, length, CV_32F);
    for(int i=0;i<400;i++){
        Mat dstrow = faces.row(i);
        Mat dstrow2 = subFaces.row(i);
        subtract(dstrow,meanMat1,dstrow2);
    }


    //Mat transFaces(length, 400,theface.type());
   // transFaces=subFaces.t();
    //cvTranspose(&subFaces, &transFaces);

    //CovarMatrix

    cout<<"calculating CovarMatrix"<<endl;
    Mat covMat;
    Mat meanMat3;
    calcCovarMatrix(subFaces, covMat, meanMat3, CV_COVAR_ROWS,CV_32F);


//    //eigen value
    cout<<"calculating eigens"<<endl;
    Mat eValuesMat,eVectorsMat,evects1,eVectorsMatT;
    eigen(covMat, true,eValuesMat, eVectorsMat);

    eVectorsMat.convertTo(eVectorsMat,CV_32F);
    subFaces.convertTo(subFaces,CV_32F);
    eValuesMat.convertTo(eValuesMat,CV_32F);

    Mat eigenVec(faces.rows, faces.cols, CV_32F);
    gemm( eVectorsMat, faces, 1, Mat(), 0, eigenVec);
    eVectorsMat = eigenVec;

    cout<<"calculating energy"<<endl;
    float energy = 0.9;
    float eigensum = 0.0;
    for(int i = 0; i < eValuesMat.rows; i++)
       eigensum += eValuesMat.at<float>(i,0);

    float tmpsum = 0.0;
    int index1;
    for( index1 = 0; index1 < eValuesMat.rows; index1++)
        {
            tmpsum += eValuesMat.at<float>(index1,0);
            if(tmpsum/eigensum >= energy)
                break;
        }
    int number = index1;

    //cout<<number<<endl;

    //trans里面记录的是前number个特征向量
    Mat trans,norm_trans;
    trans = Mat(number, eVectorsMat.cols, eVectorsMat.type());
    for(int i = 0; i < number; i++)
    {
        Mat dstRow = trans.row(i);
        eVectorsMat.row(i).copyTo(dstRow);
    }
    normalize( trans, norm_trans, 0, 255, NORM_MINMAX, CV_8U, Mat() );

    string eigenfile = "eigenface";
    FileStorage fs(eigenfile, FileStorage::WRITE);
    fs<<"eigenface"<<norm_trans;
    fs<<"faces"<<faces;
    fs.release();

    //叠加特征人脸
    Mat combineFace = Mat::zeros(1, length, CV_32F);
    for(int i = 0; i < 10; i++)
    {
        combineFace += eVectorsMat.row(i);
    }
    combineFace = combineFace.reshape(0, w);
    divide(combineFace, 10, combineFace);
    normalize(combineFace, combineFace, 0, 255, CV_MINMAX);
    combineFace.convertTo(combineFace, CV_8U);

    namedWindow("eigenface", WINDOW_NORMAL);
    imshow("eigenface", combineFace);


//    Mat aa=eVectorsMat.row(0);
//    aa = aa.reshape(0, w);
//    normalize(aa,aa, 0, 255, CV_MINMAX);
//    aa.convertTo(aa, CV_8U);
//    namedWindow("eigenface", 11);
//    imshow("eigenface", aa);


    //    namedWindow("2", 1);
    //    imshow("2",faces);
    waitKey (0);
    return 0;
}
