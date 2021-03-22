# GNU General Public License v3.0
# Originally Posted: github.com/yaashwardhan

from tkinter import *
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import cv2
import os
import time
import datetime
import csv
import pandas as pd
from PIL import Image
import numpy as np


#========================= B A C K E N D ==========================#

########################## Basic Methods ##########################


# This function will create the directory if it does not exist


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def tick():
    # fetches local computer time
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    # refreshes every 200 milliseconds
    clock.after(200, tick)

########################## Take Images, Create Dataset ##########################

# This function takes images of the student and stores them in a 'dataset/' directory


def capture_img():
    notifier.configure(text= 'CONSOLE: Capturing Images.. Creating A Dataset..')
    # Firstly make sure all the directories are present
    assure_path_exists("dataset/")
    assure_path_exists("attendance_sheets/")
    assure_path_exists("student_details/")
    assure_path_exists("trainer/")
    assure_path_exists("trainer/trainer.yml")

    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    serial = 0

    exists = os.path.isfile("student_details/student_details.csv")

    if exists:
        with open("student_details/student_details.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        csvFile1.close()
    else:
        with open("student_details/student_details.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()

    Id = (txtfield1.get())
    name = (txtfield2.get())

    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')

    count = 0

    while (True):

        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_rects:

            # Note: Unlike matplotlib.pyplot, cv2 takes images as BGR instead of RGB
            cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 206, 128), 2)

            count += 1

            cv2.imwrite("dataset/ " + name + "." + str(serial) + "." +
                        Id + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('Learning Your Face', frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        elif count >= 100:
            notifier.configure(text= 'CONSOLE: Images Have Been Captured Successfully..')
            break

    cap.release()

    cv2.destroyAllWindows()

    row = [serial, '', Id, '', name]
    with open('student_details/student_details.csv', 'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()

########################## Train Dataset, Generate trainer.yml file ##########################


def train_dataset():
    notifier.configure(text= 'CONSOLE: Training and Learning Face..')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

    def getImagesAndLabels(path):

        path_of_img = [os.path.join(path, f) for f in os.listdir(path)]

        faceSamples = []

        Ids = []

        for current_img in path_of_img:

            converted_img = Image.open(current_img).convert('L')

            imageNp = np.array(converted_img, 'uint8')

            ID = int(os.path.split(current_img)[-1].split(".")[1])

            faces = detector.detectMultiScale(imageNp)

            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(ID)
        return faceSamples, Ids

    faces, ID = getImagesAndLabels('dataset')

    s = recognizer.train(faces, np.array(ID))

    recognizer.write('trainer/trainer.yml')

    notifier.configure(text= 'CONSOLE: Training Successful.. User\'s Face Learnt..')


########################## Clock Pre-Trained Users In, And save attendance time in a csv ##########################

def clock_in():
    notifier.configure(text= 'CONSOLE: Analysing and Rendering Facial Features..')
    start = time.time()
    period = 15

    face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')

    i = 0
    j = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']

    df = pd.read_csv("student_details/student_details.csv")

    while True:
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 7)

        for (x, y, w, h) in faces:

            # roi is the region of interest, it slices the gray array. It selected row starting with y till y+h and column starting with x till x+w
            roi_gray = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            serial, confidence = recognizer.predict(roi_gray)

            if (confidence < 40):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date),
                              '', str(timeStamp)]

            else:
                Id = '\nUnable To Recognize This Entity!\n'
                bb = str(Id)

            cv2.putText(frame, "Name : " + str(bb) + " Confidence (Lower the Better):  " +
                        str(int(confidence)), (x, y - 10), font, 1, (120, 255, 120), 4)
        cv2.imshow('frame', frame)

        if time.time() > start + period:
            break

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    exists = os.path.isfile(
        "attendance_sheets/attendance_sheets_" + date + ".csv")

    if exists:
        with open("attendance_sheets/attendance_sheets_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open("attendance_sheets/attendance_sheets_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()

    with open("attendance_sheets/attendance_sheets_" + date + ".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '

    csvFile1.close()
    cap.release()
    cv2.destroyAllWindows()
    notifier.configure(text= 'CONSOLE: Thank You! Please Check the Attendance Sheet..')
    csv_updater()

#======================== F R O N T E N D ==========================#

######################### Root Window GUI ##########################

from tkinter import *
import tkinter.ttk as ttk
import csv
from PIL import Image, ImageTk

root = Tk()

root.title("Facial Recognition Attendance System using OpenCV: PL and DBMS Project Semester 4")
root.geometry('967x600')

# Background Image

load = Image.open('assets/frontend.png')
render = ImageTk.PhotoImage(load) 
img = Label (root, image = render)
img.place (x = 0, y = 0)


# Take Pictures Button
take_pic = PhotoImage(file = 'assets/pic_take.png')
take_pic_button = Button(root, image = take_pic, command = capture_img, borderwidth = 0,  highlightbackground="light sky blue", highlightthickness = 0)
take_pic_button.place(x=200, y=332)

# Train Dataset Button
train_pic = PhotoImage(file = 'assets/pic_train.png')
train_pic_button = Button(root, image = train_pic, command = train_dataset, borderwidth = 0, highlightbackground="gray18", highlightthickness = 0)
train_pic_button.place(x=105, y=438)

# Test Dataset Button
test_pic = PhotoImage(file = 'assets/pic_test.png')
test_pic_button = Button(root, image = test_pic, command = clock_in, borderwidth = 0, highlightbackground="bisque4", highlightthickness = 0)
test_pic_button.place(x=703, y=438)

# Text Input Entries

txtfield1 = Entry(root, width=20 ,bd = 0, bg = "snow", fg = "black", highlightthickness = 0, font=('arial', 15))
txtfield1.config(insertbackground='black')
txtfield1.place(x=165, y=198)

txtfield2 = Entry(root, width=20 ,bd = 0, bg = "snow", fg = "black", highlightthickness = 0, font=('arial', 15))
txtfield2.config(insertbackground='black')
txtfield2.place(x=165, y=267)

# Notification Bar

notifier = Label(root, text="CONSOLE: null..", fg= 'medium sea green', anchor= 'w', bg = 'grey10', width=138, height=1, font=('source code pro', 14,))

notifier.place(x=0, y=578)



############################### CSV BAR GUI #################################


style = ttk.Style(root)
style.theme_use("clam")
style.configure("Treeview", background="gray14", fieldbackground="gray14", foreground="snow")


TableMargin = Frame(root, width=340, height = 315, borderwidth = 8, highlightthickness = 0)
TableMargin.pack(padx=(570,0),pady=(80,260))



tree = ttk.Treeview(TableMargin, columns=("Id", "Name", "Date", "Time"), height=400, selectmode="extended")


tree.heading('Id', text="Id", anchor=W)

tree.heading('Name', text="Name", anchor=W)

tree.heading('Date', text="Date", anchor=W)

tree.heading('Time', text="Time", anchor=W)

tree.column('#0', stretch=NO, minwidth=0, width=0)
tree.column('#1', stretch=NO, minwidth=0, width=40)
tree.column('#2', stretch=NO, minwidth=0, width=90)
tree.column('#3', stretch=NO, minwidth=0, width=67)
tree.column('#4', stretch=NO, minwidth=0, width=70)

tree.pack()

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')

def csv_updater():
    tree.delete(*tree.get_children())
    try:
        with open("attendance_sheets/attendance_sheets_" + date + ".csv") as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                Id = row['Id']
                Name = row['Name']
                Date = row['Date']
                Time = row['Time']
                tree.insert("", 0, values=(Id,Name,Date,Time))
        notifier.configure(text= 'CONSOLE: Attendance Records Found For ' + date)
    except:
        tree.insert("", 0, values=('null', 'null', 'null', 'null'))
        notifier.configure(text= 'CONSOLE: No Attendance Records Found For ' + date)

csv_updater()


root.mainloop()


