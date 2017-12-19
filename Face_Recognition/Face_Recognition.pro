#-------------------------------------------------
#
# Project created by QtCreator 2017-12-10T16:28:22
#
#-------------------------------------------------

QT       += core gui
QT       += sql

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = Face_Recognition
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    detectObject.cpp \
    preprocessFace.cpp \
    recognition.cpp

HEADERS  += mainwindow.h \
    detectObject.h \
    preprocessFace.h \
    recognition.h

FORMS    += mainwindow.ui

INCLUDEPATH += D:\OpenCV_lib\include
INCLUDEPATH += D:\OpenCV_lib\include\opencv
INCLUDEPATH += D:\OpenCV_lib\include\opencv2

LIBS += D:\OpenCV_lib\lib\libopencv_*.a\
