#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    //this->resize(QSize( 900, 600 ));

    // Load the face and 1 or 2 eye detection XML classifiers.
    initDetectors(faceCascade, eyeCascade1, eyeCascade2);
    initPersonNum();
    ui->lineEdit_3->hide();
    ui->label_6->setText("请点击菜单");
    ui->pushButton->hide();
    ui->pushButton_2->hide();
    ui->label_5->hide();
    ui->lineEdit->setEnabled(false);
    ui->lineEdit_2->setEnabled(false);
}

MainWindow::~MainWindow()
{
    delete ui;
}


const float UNKNOWN_PERSON_THRESHOLD = 0.65f;

const int BORDER = 8;  // Border between GUI elements to the edge of the image.

const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";

// Cascade Classifier file, used for Face Detection.
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
//const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.

// Try to set the camera resolution. Note that this only works for some cameras on
// some computers and only for some drivers, so don't rely on it to work!
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
const int faceWidth = 70;
const int faceHeight = faceWidth;

// Parameters controlling how often to keep new faces when collecting them. Otherwise, the training set could look to similar to each other!
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      // How much the facial image should change before collecting a new face photo for training.
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;       // How much time must pass before collecting a new face photo for training.

const bool preprocessLeftAndRightSeparately = true;
// Preprocess left & right sides of the face separately, in case there is stronger light on one side.

int m_selectedPerson = 0;
vector<int> m_latestFaces;

QImage Mat2QImage(const Mat &mat)
{
    //8-bits unsigned,NO.OFCHANNELS=1
    if(mat.type()==CV_8UC1)
    {
       //cout<<"1"<<endl;
        //Set the color table(used to translate colour indexes to qRgb values)
        QVector<QRgb>colorTable;
        for(int i=0;i<256;i++)
            colorTable.push_back(qRgb(i,i,i));
        //Copy input Mat
        const uchar*qImageBuffer=(const uchar*)mat.data;
        //Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer,mat.cols,mat.rows,mat.step,QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    //8-bits unsigned,NO.OFCHANNELS=3
    if(mat.type()==CV_8UC3)
    {
       //cout<<"3"<<endl;
        //Copy input Mat
        const uchar*qImageBuffer=(const uchar*)mat.data;
        //Create QImage with same dimensions as input Mat
        QImage img(qImageBuffer,mat.cols,mat.rows,mat.step,QImage::Format_RGB888);
        return img.rgbSwapped();

    }
    else
    {
        cout <<"ERROR:Mat could not be converted to QImage.";
        return QImage();
    }
}

void MainWindow::initPersonNum()
{
    //连接数据库
    //db = QSqlDatabase::database("qt_sql_default_connection");
    db = QSqlDatabase::addDatabase("QMYSQL");//使用MySQL数据库
    db.setHostName("localhost");
    db.setPort(3306);
    db.setDatabaseName("face_recognition");
    db.setUserName("root");
    db.setPassword("hzl1996626");
    db.open();
    QSqlQuery query;
    //判断数据库是否已存在表
    query.prepare("SELECT count(*) FROM students");

    query.exec();
    if(query.next()){
        m_numPersons = query.value(0).toInt();
        m_selectedPerson = m_numPersons;
    }

    cout << "connect database successful, the number of person is " << m_numPersons << endl;
}

// Load the face and 1 or 2 eye detection XML classifiers.
void MainWindow::initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // Load the Face Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\haarcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
        // Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
        //exit(1);
    }
    else
        cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}

// Get access to the webcam.
void MainWindow::initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    // Get access to the default camera.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        videoCapture.open(cameraNumber);
        opencamera = true;
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}

//检测人脸
void MainWindow::detect(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // Grab the next camera frame. Note that you can't modify camera frames.
    Mat cameraFrame;
    videoCapture >> cameraFrame;

    if( cameraFrame.empty() ) {
        cerr << "ERROR: Couldn't grab the next camera frame." << endl;
        //exit(1);
        timer->stop();
    }

    // Get a copy of the camera frame that we can draw onto.
    Mat displayedFrame;
    cameraFrame.copyTo(displayedFrame);

    // Find a face and preprocess it to have a standard size and contrast & brightness.
    Rect faceRect;  // Position of detected face.
    Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
    Point leftEye, rightEye;    // Position of the detected eyes.
    Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);
    //imshow("preprocessedFace", preprocessedFace );

    // Draw an anti-aliased rectangle around the detected face.
    if (faceRect.width > 0) {
        rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

        // Draw light-blue anti-aliased circles for the 2 eyes.
        Scalar eyeColor = CV_RGB(0,255,255);
        if (leftEye.x >= 0) {   // Check if the eye was detected
            circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
        }
        if (rightEye.x >= 0) {   // Check if the eye was detected
            circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
        }
    }

    //图像处理函数
    QImage img = Mat2QImage(displayedFrame);//将mat格式转换为Qimage格式
    ui->label->setPixmap(QPixmap::fromImage(img));//将结果在label上显示
    ui->label->setScaledContents(true);//使图像尺寸与label大小匹配
}

void MainWindow::collect(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2,  vector<Mat> &preprocessedFaces,vector<int> &faceLabels)
{
    Mat old_prepreprocessedFace;
    double old_time = 0;
    int i = 0;

    m_latestFaces.push_back(-1);

    while(i < 7){
        // Grab the next camera frame. Note that you can't modify camera frames.
        Mat cameraFrame;
        videoCapture >> cameraFrame;

        // Get a copy of the camera frame that we can draw onto.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

        // Find a face and preprocess it to have a standard size and contrast & brightness.
        Rect faceRect;  // Position of detected face.
        Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
        Point leftEye, rightEye;    // Position of the detected eyes.
        Mat preprocessedFace;
        preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

        // Draw an anti-aliased rectangle around the detected face.
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

            // Draw light-blue anti-aliased circles for the 2 eyes.
            Scalar eyeColor = CV_RGB(0,255,255);
            if (leftEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) {   // Check if the eye was detected
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
            }
        }

        //图像处理函数
        QImage img = Mat2QImage(displayedFrame);//将mat格式转换为Qimage格式
        ui->label->setPixmap(QPixmap::fromImage(img));//将结果在label上显示
        ui->label->setScaledContents(true);//使图像尺寸与label大小匹配

        bool gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;

        // Check if we have detected a face.
        if (gotFaceAndEyes) {

            // Check if this face looks somewhat different from the previously collected face.
            double imageDiff = 10000000000.0;

            if (old_prepreprocessedFace.data) {
                imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
            }

            // Also record when it happened.
            double current_time = (double)getTickCount();
            double timeDiff_seconds = (current_time - old_time)/getTickFrequency();

            // Only process the face if it is noticeably different from the previous frame and there has been noticeable time gap.
            if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
                // Also add the mirror image to the training set, so we have more training data, as well as to deal with faces looking to the left or right.
                Mat mirroredFace;
                flip(preprocessedFace, mirroredFace, 1);

                // Add the face images to the list of detected faces.
                preprocessedFaces.push_back(preprocessedFace);
                preprocessedFaces.push_back(mirroredFace);
                faceLabels.push_back(m_selectedPerson);
                faceLabels.push_back(m_selectedPerson);

                /*
                cout << m_selectedPerson << " m_selectedPerson" << endl;
                cout << preprocessedFaces.size() << " preprocessedFaces" << endl;
                cout << imageDiff << " imageDiff" << endl;
                */

                // Keep a reference to the latest face of each person.
                m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  // Point to the non-mirrored face.

                // Show the number of collected faces. But since we also store mirrored faces, just show how many the user thinks they stored.
                cout << "Saved face " << (preprocessedFaces.size()/2) << " for person " << m_selectedPerson << endl;
                //String message = "仍需采集 " + to_string(preprocessedFaces.size()/2) + " 张照片";
                //ui->label_6->setText(QString::fromStdString(message));
                i++;

                // Make a white flash on the face, so the user knows a photo has been taken.
                Mat displayedFaceRegion = displayedFrame(faceRect);
                displayedFaceRegion += CV_RGB(90,90,90);

                // Keep a copy of the processed face, to compare on next iteration.
                old_prepreprocessedFace = preprocessedFace;
                old_time = current_time;
            }
        }
    }
}

Ptr<BasicFaceRecognizer> MainWindow::training(vector<Mat> &preprocessedFaces, vector<int> &faceLabels)
{
    Ptr<BasicFaceRecognizer> model;

    // Start training from the collected faces using Eigenfaces or a similar algorithm.
    model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);
    string name = "person_" + to_string(m_selectedPerson) +".xml";
    cout << name << endl;

    model->save(name);
    cout << "person " << to_string(m_selectedPerson) << " train model finish " << endl;

    ui->lineEdit->setEnabled(true);
    ui->lineEdit_2->setEnabled(true);

    return model;
}


void MainWindow::recognition(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    Ptr<BasicFaceRecognizer> model = face::createEigenFaceRecognizer();

    // Grab the next camera frame. Note that you can't modify camera frames.
    Mat cameraFrame;
    videoCapture >> cameraFrame;

    if( cameraFrame.empty() ) {
        cerr << "ERROR: Couldn't grab the next camera frame." << endl;
        //exit(1);
        timer->stop();
    }

    // Get a copy of the camera frame that we can draw onto.
    Mat displayedFrame;
    cameraFrame.copyTo(displayedFrame);
    //imshow("displayedFrame", displayedFrame);

    // Run the face recognition system on the camera image. It will draw some things onto the given image, so make sure it is not read-only memory!
    int identity = -1;

    // Find a face and preprocess it to have a standard size and contrast & brightness.
    Rect faceRect;  // Position of detected face.
    Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
    Point leftEye, rightEye;    // Position of the detected eyes.
    Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

    bool gotFaceAndEyes = false;
    if (preprocessedFace.data)
        gotFaceAndEyes = true;

    //cout << m_numPersons << " m_numPersons" << endl;

    double similarity;
    string outputStr;

    for(int j = 0; j < m_numPersons; j++){
        string name = "person_" + to_string(j) + ".xml";
        model->load(name);
        cout << "load " << name << endl;

        if (gotFaceAndEyes) {

            // Generate a face approximation by back-projecting the eigenvectors & eigenvalues.
            Mat reconstructedFace;
            reconstructedFace = reconstructFace(model, preprocessedFace);

            // Verify whether the reconstructed face looks like the preprocessed face, otherwise it is probably an unknown person.
            similarity = getSimilarity(preprocessedFace, reconstructedFace);

            if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                // Identify who the person is in the preprocessed face image.
                identity = j;

                outputStr = to_string(identity);
                timer_recognition->stop();
                cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;
                printStudentInformation(outputStr);
                break;

            }
            else if(j+1 == m_numPersons){
                // Since the confidence is low, assume it is an unknown person.
                outputStr = "Unknown";
                cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;
                ui->label_6->setText("Unknown");
            }
        }
    }
    if(!gotFaceAndEyes){
        cout << "No face"<< endl;
        ui->label_6->setText("No face");
    }

}

void MainWindow::test()
{
    Ptr<BasicFaceRecognizer> model = face::createEigenFaceRecognizer();

    if (model.empty()) {
        cerr << "ERROR: The FaceRecognizer algorithm [" << facerecAlgorithm << "] is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer." << endl;
        exit(1);
    }

    model->load("E:/C++_Qt/build-Face_Recognition-Desktop_Qt_5_6_2_MinGW_32bit-Debug/person_0.xml");
    cout << "load successful " << endl;
}

void MainWindow::readCamera_detect()
{
    //保存当前帧的图像
    detect(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

}

void MainWindow::readCamera_collect()
{
    //保存当前帧的图像
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    collect(videoCapture, faceCascade, eyeCascade1, eyeCascade2, preprocessedFaces, faceLabels);
    training(preprocessedFaces, faceLabels);
}

void MainWindow::readCamera_recognition()
{
    //保存当前帧的图像
    recognition(videoCapture, faceCascade, eyeCascade1, eyeCascade2);
}

void MainWindow::printStudentInformation(string identity)
{
    ui->label_6->setText("识别结果");
    ui->pushButton_2->hide();

    QSqlQuery query;
    QString id;
    id = QString::fromStdString(identity);
    //判断数据库是否已存在表
    QString sql = "SELECT student_name, student_id FROM students WHERE id = '" + id;
    sql.append("'");
    query.prepare(sql);
    query.exec();

    if(query.next()){
        ui->lineEdit->setText(query.value(0).toString());
        ui->lineEdit_2->setText(query.value(1).toString());
        ui->lineEdit_3->setText("签到成功");

        ui->lineEdit->setEnabled(false);
        ui->lineEdit_2->setEnabled(false);
        ui->lineEdit_3->setEnabled(false);
    }
    QString sql_update = "UPDATE students SET attend_class = 'yes' WHERE id = '"+ id;
    sql_update.append("'");
    query.prepare(sql_update);
    query.exec();

    QMessageBox::about(NULL, "提示", "签到成功");
}

void MainWindow::on_actionOpen_camera_triggered()//打开摄像头
{
    // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraNumber = 0;   // Change this if you want to use a different camera device.

    // Get access to the webcam.
    initWebcam(videoCapture, cameraNumber);
    // Try to set the camera resolution. Note that this only works for some cameras on
    // some computers and only for some drivers, so don't rely on it to work!
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    timer=new QTimer(this);
    connect(timer,SIGNAL(timeout()),this,SLOT(readCamera_detect()));
    if(!videoCapture.isOpened())
    {
       cout<<"nocap"<<endl;
    }

    timer->start(100);
}

void MainWindow::on_actionClose_camera_triggered()//关闭摄像头
{
    if(opencamera){
        videoCapture.release();
        opencamera = false;
    }
}

void MainWindow::on_actionClose_system_triggered()
{
    if (!(QMessageBox::information(this,tr("提示"),tr("确定要退出系统?"),tr("是"),tr("否"))))
    {
        qApp->exit(0);
    }
}

void MainWindow::on_actionTrain_model_triggered()
{
    if(opencamera){
        ui->lineEdit->setEnabled(false);
        ui->lineEdit_2->setEnabled(false);
        ui->lineEdit_3->hide();
        ui->label_6->setText("请输入学生信息");
        ui->pushButton->show();
        ui->label_5->hide();

        QMessageBox::about(NULL, "提示", "摄像头录入图像并输入信息");

        readCamera_collect();
    }
}
void MainWindow::on_actionReconition_triggered()
{
    if(opencamera && m_numPersons > 0){
        ui->label_6->setText("识别结果");
        ui->pushButton->hide();
        ui->pushButton_2->show();
        ui->label_5->show();
        ui->lineEdit_3->show();
        ui->lineEdit_3->setEnabled(false);

        if(opencamera){
            timer_recognition=new QTimer(this);
            connect(timer_recognition,SIGNAL(timeout()),this,SLOT(readCamera_recognition()));
            timer_recognition->start(1000);
        }
    }
}

void MainWindow::on_actionDelete_information_triggered()
{
    QSqlQuery query;
    QString sql_update;
    QString id;
    if (!(QMessageBox::information(this,tr("提示"),tr("确定删除学生点名信息?"),tr("是"),tr("否"))))
    {
        //cout << m_numPersons << endl;
        for(int i = 0; i < m_numPersons; i++){
            id = QString::fromStdString(to_string(i));
            sql_update = "UPDATE students SET attend_class = 'unknown' WHERE id = '"+ id;
            sql_update.append("'");
            //cout << sql_update.toStdString() << endl;
            query.prepare(sql_update);
            query.exec();
        }
    }

}

void MainWindow::on_actionDelete_student_triggered()
{
    QSqlQuery query;
    if (!(QMessageBox::information(this,tr("提示"),tr("确定删除全部学生信息?"),tr("是"),tr("否"))))
    {
        QString sql_delete = "DELETE FROM students";
        query.prepare(sql_delete);

        if(query.exec()){
            m_numPersons = 0;
            m_selectedPerson = 0;
        }
    }
}

void MainWindow::on_pushButton_clicked()
{
    QSqlQuery query;
    QString attend = "yes";
    QString person = QString::fromStdString(to_string(m_selectedPerson));
    QString pose[] = {
        person,
        ui->lineEdit->text(),
        ui->lineEdit_2->text(),
        attend
         };

    //从表单中获取数据，执行插入数据语句
    query.prepare("INSERT INTO students VALUES(?,?,?,?)");

    QVariantList Person; //1
    Person << pose[0];
    query.addBindValue(Person);
    QVariantList student_name; //2
    student_name << pose[1];
    query.addBindValue(student_name);
    QVariantList student_id; //3
    student_id << pose[2];
    query.addBindValue(student_id);
    QVariantList attend_class; //4
    attend_class << pose[3];
    query.addBindValue(attend_class);

    if(!query.execBatch()){
        QMessageBox::critical(0,QObject::tr("Error"),query.lastError().text());
    }
    query.finish();//插入错误时报错
    QMessageBox::about(NULL, "提示", "插入成功");

    m_selectedPerson++;
    m_numPersons++;

    ui->lineEdit->setEnabled(false);
    ui->lineEdit_2->setEnabled(false);
    ui->pushButton->hide();
    ui->label_6->setText("请点击菜单");
}

void MainWindow::on_pushButton_2_clicked()
{
    timer_recognition->stop();
    ui->pushButton_2->hide();
    ui->label_6->setText("请点击菜单");
}

//说明文字
void MainWindow::on_actionHelp_triggered()
{
    QMessageBox::about(NULL, "说明","人脸识别点名系统\ngithub: \nhttps://github.com/HerbertHu/Face-Recognition-Call-the-names-system\n©Herbert_Hu");
}
