#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <iostream>

#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlQuery>
#include <QMessageBox>
#include <QtSql/QSqlError>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>

#include <detectObject.h>
#include <preprocessFace.h>
#include <recognition.h>

#include <QTimer>
#include <QTime>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_actionOpen_camera_triggered();
    void on_actionClose_camera_triggered();
    void on_actionClose_system_triggered();
    void on_actionTrain_model_triggered();
    void on_actionReconition_triggered();
    void on_actionDelete_information_triggered();
    void on_actionDelete_student_triggered();
    void on_actionHelp_triggered();
    void readCamera_detect();
    void readCamera_collect();
    void readCamera_recognition();
    void on_pushButton_clicked();
    void on_pushButton_2_clicked();
    void initPersonNum();
    void printStudentInformation(string identity);

private:
    Ui::MainWindow *ui;
    QTimer *timer;
    QTimer *timer_recognition;
    cv::VideoCapture videoCapture;
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    bool opencamera = false;
    int m_numPersons = 0;
    int m_selectedPerson = 0;

    QSqlDatabase db;

    void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2);
    void initWebcam(VideoCapture &videoCapture, int cameraNumber);

    void detect(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2);
    void collect(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, vector<Mat> &preprocessedFaces, vector<int> &faceLabels);
    void recognition(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2);
    Ptr<BasicFaceRecognizer> training(vector<Mat> &preprocessedFaces, vector<int> &faceLabels);
    void sleep(unsigned int msec);
};

#endif // MAINWINDOW_H
