#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>
#include <ctype.h>

using namespace cv;

cv::CascadeClassifier face_cascade, eye_cascade;

void detectFaces(cv::Mat& _frame, std::vector<cv::Rect>& _faces){

    cv::Mat frame_gray;
    cvtColor(_frame, frame_gray, CV_BGR2GRAY);  // Convert to gray scale
    cv::equalizeHist(frame_gray, frame_gray);   	// Equalize histogram

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, _faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

    // Iterate over all of the faces
    for( size_t i = 0; i < _faces.size(); i++ ) {

        // Find center of faces
        cv::Point center(_faces[i].x + _faces[i].width/2, _faces[i].y + _faces[i].height/2);

        // Find center of face
        cv::Mat face = frame_gray(_faces[i]);
        std::vector<cv::Rect> eyes;

        // Try to detect eyes, inside each face
        eye_cascade.detectMultiScale(face, eyes, 1.1, 2, 0 |cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30) );

        // Check to see if eyes inside of face, if so, draw ellipse around face
        if(eyes.size() > 0){
            cv::ellipse(_frame, center, cv::Size(_faces[i].width/2, _faces[i].height/2),
                        0, 0, 360, cv::Scalar( 255, 0, 255 ), 4, 8, 0 );

            cv::Point eye;
            for (unsigned int j = 0; j < eyes.size(); j++){
                eye = cv::Point( _faces[i].x + eyes[j].x + eyes[j].width*0.5, _faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                cv::circle( _frame, eye, radius, cv::Scalar( 255, 0, 0 ), 4, 8, 0 );
            }
        }
    }
}


Mat image;

bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
int vmin = 10, vmax = 256, smin = 30;

string hot_keys =
    "\nHot keys: \n"
    "\tESC - quit the program\n"
    "\tb - switch to/from backprojection view\n"
    "\th - show/hide object histogram\n"
    "\tp - pause video\n"
    "To initialize tracking, select the object with mouse\n";

const char* keys =
{
    "{help h | | show help message}{@camera_number| 0 | camera number}"
};

int main( int argc, const char** argv )
{
    VideoCapture cap;
    Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;
    CommandLineParser parser(argc, argv, keys);

    face_cascade.load("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml");
    eye_cascade.load("/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml");


    int camNum = 0;
    cap.open(camNum);

    if( !cap.isOpened() ) {
        std::cout << "***Could not initialize capturing...***\n";
        std::cout << "Current parameter's value: \n";
        return -1;
    }


    std::cout << hot_keys;

    namedWindow( "Histogram", 0 );
    namedWindow( "CamShift Demo", 0 );

    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;

    Rect mainFace;

    while (true){

        cap >> frame;
        std::vector<Rect> faces;
        detectFaces(frame, faces);

        imshow("CamShift Demo", frame);

        int bigFace = 0;
        int bigArea = -INT_MAX;
        for (unsigned int i = 0; i < faces.size(); i++){
            const int area = faces[i].area();
            if (area > bigArea){
                bigArea = area;
                bigFace = i;
            }
        }

        mainFace = faces[bigFace];

        std::cout << "Hit any key if face is tracked correctly\n";

        if (waitKey(30)>=0){
            trackObject = -1;
            break;
        }
    }

    std::cerr << mainFace.width << " " << mainFace.height << std::endl;


    while (true){

        if( !paused ){

            cap >> frame;
            if( frame.empty() )
                break;
        }

        frame.copyTo(image);

        if( !paused ){

            cvtColor(image, hsv, COLOR_BGR2HSV);

            if( trackObject )
            {
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)),
                        Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if( trackObject < 0 ){

                    Mat roi(hue, mainFace), maskroi(mask, mainFace);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, NORM_MINMAX);

                    trackWindow = mainFace;
                    trackObject = 1;

                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, COLOR_HSV2BGR);

                    for( int i = 0; i < hsize; i++ ){

                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows),
                                   Point((i+1)*binW,histimg.rows - val),
                                   Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }
                }

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                    TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
                if( trackWindow.area() <= 1 ){

                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                if( backprojMode ){
                    cvtColor( backproj, image, COLOR_GRAY2BGR );
                }
                ellipse( image, trackBox, Scalar(0,0,255), 3 );
            }
        }
        else if( trackObject < 0 ){
            paused = false;
        }

        if( selectObject && mainFace.width > 0 && mainFace.height > 0 )
        {
            Mat roi(image, mainFace);
            bitwise_not(roi, roi);
        }

        imshow( "CamShift Demo", image );
        imshow( "Histogram", histimg );

        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch(c)
        {
        case 'b':
            backprojMode = !backprojMode;
            break;
        case 'h':
            showHist = !showHist;
            if( !showHist )
                destroyWindow( "Histogram" );
            else
                namedWindow( "Histogram", 1 );
            break;
        case 'p':
            paused = !paused;
            break;
        default:
            ;
        }
    }

    return 0;
}
