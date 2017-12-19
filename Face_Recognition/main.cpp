#include "mainwindow.h"
#include <QApplication>

#include <detectObject.h>
#include <preprocessFace.h>


int main(int argc, char *argv[])
{
    cout << "WebcamFaceRec, by Shervin Emami (www.shervinemami.info), June 2012." << endl;
    cout << "Realtime face detection + face recognition from a camera using Eigenfaces." << endl;
    cout << "Class call the names system, by Herbert Hu, December 2017" << endl;
	cout << "github: https://github.com/HerbertHu/Face-Recognition-Call-the-names-system" << endl;
    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

    QApplication a(argc, argv);
    MainWindow w;
    //设置主窗口的显示与名字
    w.setWindowTitle("课堂人脸识别点名系统");
    w.setFixedSize(QSize( 600, 300 ));
    w.show();

    return a.exec();
}

