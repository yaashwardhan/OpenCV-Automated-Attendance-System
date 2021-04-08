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

########################## SQL ##########################

import mysql.connector as msql
from mysql.connector import Error
try:
    conn = msql.connect(host='localhost', user='root',
                        password='your_password', db='opencv')
    cursor = conn.cursor()

except Error as e:
    print("Error while connecting to MySQL", e)

########################## Basic Methods ##########################


def path_existence(path):
    """
    This function will create the directory if it does not exist
    """
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


########################## Take Images, Create Dataset ##########################


def capture_img():
    """
    This function takes images of the student and stores them in a 'dataset/' directory
    """

    notifier.configure(text='CONSOLE: Capturing Images.. Creating A Dataset..')
    # Firstly make sure all the directories are present
    path_existence("dataset/")
    path_existence("daily_generated_attendance_csv/")
    path_existence("student_details/")
    path_existence("trainer/")
    path_existence("trainer/trainer.yml")

    columns = ['SERIAL NO.',  'ID',  'NAME',
               'GENDER',  'AGE',  'PHONE NUMBER',  'ADDRESS']
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
    gender = (txtfield3.get())
    age = (txtfield4.get())
    phonenumber = (txtfield5.get())
    address = (txtfield6.get())
    """
    Checking if user with same id exists, if yes then update
    """
    lst = [0]
    idchecker = False

    cursor.execute('select * from student_details')
    for x in cursor.fetchall():
        lst.append(x[0])

    for i in lst:
        if Id == i:
            idchecker = True
            break
        else:
            pass

    if idchecker == True:
        """
        User exists, hence needs to be updated
        """
        notifier.configure(
            text='CONSOLE: User already exists, Please Update Manually In The View Database section..')
    else:
        """
        User doesnt exist, hence needs to be added
        """
        cap = cv2.VideoCapture(0)

        face_cascade = cv2.CascadeClassifier(
            'assets/haarcascade_frontalface_default.xml')

        count = 0

        while (True):

            ret, frame = cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_rects = face_cascade.detectMultiScale(
                gray, 1.3, 5)  # img, scaleFactor=1.1, minNeighbors=5,

            for (x, y, w, h) in face_rects:

                # Note: Unlike matplotlib.pyplot, cv2 takes images as BGR instead of RGB
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (225, 206, 128), 2)

                count += 1

                cv2.imwrite("dataset/ " + name + "." + str(serial) + "." +
                            Id + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                cv2.imshow('Learning Your Face', frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

            elif count >= 100:
                notifier.configure(
                    text='CONSOLE: Images Have Been Captured Successfully..')
                break

        cap.release()

        cv2.destroyAllWindows()
        """
        Saving Details to student_details.csv
        """
        row = [serial,  Id,  name,  gender,
               age,  phonenumber,  address]
        with open('student_details/student_details.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        """
        Inserting Details to SQL Database
        """
        cursor.execute('INSERT INTO student_details(studid, stdname, gender, age, phoneno, address) VALUES(%s,%s,%s,%s,%s,%s)',
                       (Id, name, gender, age, phonenumber, address))
        conn.commit()
        print("Record inserted")


########################## Train Dataset, Generate trainer.yml file ##########################


def train_dataset():
    """
    Fetching images from dataset and then using cascade classifiers to train from them and add to trainer.yml
    """
    notifier.configure(text='CONSOLE: Training and Learning Face..')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(
        "assets/haarcascade_frontalface_default.xml")

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

    notifier.configure(
        text='CONSOLE: Training Successful.. User\'s Face Learnt..')


########################## Clock Pre-Trained Users In #####################################

def clock_in():
    """
    Student selects a subject to give attendance for, then using cascade classifier and trainer.yml, the student is verified
    """
    notifier.configure(
        text='CONSOLE: Analysing and Rendering Facial Features..')
    start = time.time()
    period = 10

    face_cascade = cv2.CascadeClassifier(
        'assets/haarcascade_frontalface_default.xml')
    """
    Taking input from drop down menu
    """
    subjectchoice = (clicked.get())

    cap = cv2.VideoCapture(0)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')

    i = 0
    j = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    col_names = ['Id', 'Name',  'Gender',  'Age',
                 'Phone',  'Address',  'Subject',  'Date', 'Time']

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
                fetchedname = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                fetchedgender = df.loc[df['SERIAL NO.']
                                       == serial]['GENDER'].values
                fetchedage = df.loc[df['SERIAL NO.'] == serial]['AGE'].values
                fetchedphonenumber = df.loc[df['SERIAL NO.']
                                            == serial]['PHONE NUMBER'].values
                fetchedaddress = df.loc[df['SERIAL NO.']
                                        == serial]['ADDRESS'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                name2 = str(fetchedname)
                name2 = name2[2:-2]
                gender2 = str(fetchedgender)
                gender2 = gender2[2:-2]
                age2 = str(fetchedage)
                age2 = age2[1:-1]
                phonenumber2 = str(fetchedphonenumber)
                phonenumber2 = phonenumber2[1:-1]
                address2 = str(fetchedaddress)
                address2 = address2[2:-2]

                attendance = [str(ID),  name2,  gender2,  age2,  phonenumber2,  address2,  subjectchoice,  str(date),
                              str(timeStamp)]

            else:
                Id = '\n Unable To Recognize This Entity! \n'
                name2 = str(Id)

            cv2.putText(frame, "Name : " + str(name2) + " Confidence (Lower the Better):  " +
                        str(int(confidence)), (x, y - 10), font, 1, (120, 255, 120), 4)
        cv2.imshow('frame', frame)

        if time.time() > start + period:
            break

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')

    """
    Writing to daily_generated_attendance_csv + date + .csv to display in GUI and also as a CSV
    """
    exists = os.path.isfile(
        "daily_generated_attendance_csv/daily_generated_attendance_csv_" + date + ".csv")

    if exists:
        with open("daily_generated_attendance_csv/daily_generated_attendance_csv_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open("daily_generated_attendance_csv/daily_generated_attendance_csv_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()

    with open("daily_generated_attendance_csv/daily_generated_attendance_csv_" + date + ".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '

    csvFile1.close()
    """
    Inserting Details to SQL Database i.e Writing attendance[] and drop down menu choice to sql
    """
    subid = 'PyAB'
    teacherid = '1'
    if(subjectchoice == 'Python'):
        subid = 'PyAB'
        teacherid = '1'
    elif(subjectchoice == 'DBMS'):
        subid = 'DbKM'
        teacherid = '2'
    elif(subjectchoice == 'TCS'):
        subid = 'TcSY'
        teacherid = '3'
    elif(subjectchoice == 'OS'):
        subid = 'OsMR'
        teacherid = '4'

    myId = str(ID)
    myId = myId[1:-1]

    cursor.execute('INSERT INTO attends(studid, subid, attdate, atttime) VALUES(%s,%s,%s,%s)',
                   (myId, subid, str(date), str(timeStamp)))

    conn.commit()

    print("Attendance inserted")

    cap.release()
    cv2.destroyAllWindows()
    notifier.configure(
        text='CONSOLE: Thank You! Please Check the Attendance Sheet..')
    csv_updater()


#======================== F R O N T E N D ==========================#

######################### Root Window GUI ##########################


root = Tk()
root.title(
    "Facial Recognition Attendance System using OpenCV: PL and DBMS Project Semester 4")
root.geometry('967x600')

"""
Background Image
"""

load = Image.open('assets/frontend.png')
render = ImageTk.PhotoImage(load)
img = Label(root, image=render)
img.place(x=0, y=0)


def showadminportal():
    global adminimg
    if (txtfieldusername.get() == 'root') and (txtfieldpassword.get() == 'root'):
        adminportal = Toplevel()
        adminportal.title("Admin Portal")
        adminportal.geometry('430x507')

        adminimg = Label(adminportal)
        adminimg.place(x=0, y=0)
        img2 = ImageTk.PhotoImage(Image.open("assets/adminportal.png"))
        adminimg.config(image=img2)
        # save label image from garbage collection!
        adminimg.image = img2

        notifier.configure(
            text='CONSOLE: Logged In Successfully..')

        """
        Functions
        """
        ########################## Update Database #####################################
        def update_database():
            """
            Updates the csv and sql database
            """
            path_existence("student_details/")

            Idx = (txtfield1admin.get())
            namex = (txtfield2admin.get())
            genderx = (txtfield3admin.get())
            agex = (txtfield4admin.get())
            phonenumberx = (txtfield5admin.get())
            addressx = (txtfield6admin.get())

            # reading the csv file
            df = pd.read_csv("student_details/student_details.csv")

            lstx = []

            lstx = df[df['ID'] == Idx].index.values
            # updating the column value/data
            for i in lstx:
                df.loc[lstx[i], 'NAME'] = namex
                df.loc[lstx[i], 'GENDER'] = genderx
                df.loc[lstx[i], 'AGE'] = agex
                df.loc[lstx[i], 'PHONE NUMBER'] = phonenumberx
                df.loc[lstx[i], 'ADDRESS'] = addressx

            # writing into the file
            df.to_csv("student_details/student_details.csv", index=False)

            cursor.execute('UPDATE student_details SET stdname = %s, gender = %s, age = %s, phoneno = %s, address = %s where studid = %s',
                           (namex, genderx, agex, phonenumberx, addressx, Idx))
            conn.commit()
            notifier.configure(text='CONSOLE: Updated')

        ########################## Update Database #####################################

        def delete_database():
            """
            Deletes all attendance from database for selected student
            """
            path_existence("student_details/")

            date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
            Idxxx = ("'" + txtfield7admin.get() + "'")
            print("Deleting ID:")
            print(Idxxx)
            # reading the csv file
            df = pd.read_csv(
                "daily_generated_attendance_csv/daily_generated_attendance_csv_" + date + ".csv")

            lstxxx = df[df['Id'] == Idxxx].index.values
            for i in lstxxx:
                # updating the column value/data
                df = df.drop(i)

            # writing into the file
            df.to_csv("daily_generated_attendance_csv/daily_generated_attendance_csv_" +
                      date + ".csv", index=False)

            csv_updater()
            """
            Deleting user attendance from database for that day
            """
            Idxxx = txtfield7admin.get()
            cursor.execute('DELETE from attends where studid = %s', (Idxxx,))
            conn.commit()
            notifier.configure(
                text='CONSOLE: All Attendance Deleted from Database for the Student')

        """
        Admin portal fields
        """
        # ID
        txtfield1admin = Entry(adminportal, width=25, bd=0, bg="snow", fg="black",
                               highlightthickness=0, font=('arial', 15))
        txtfield1admin.config(insertbackground='black')
        txtfield1admin.place(x=144, y=95)

        # Name
        txtfield2admin = Entry(adminportal, width=25, bd=0, bg="snow", fg="black",
                               highlightthickness=0, font=('arial', 15))
        txtfield2admin.config(insertbackground='black')
        txtfield2admin.place(x=144, y=145)

        # Gender
        txtfield3admin = Entry(adminportal, width=8, bd=0, bg="snow", fg="black",
                               highlightthickness=0, font=('arial', 15))
        txtfield3admin.config(insertbackground='black')
        txtfield3admin.place(x=115, y=196)

        # Age
        txtfield4admin = Entry(adminportal, width=8, bd=0, bg="snow", fg="black",
                               highlightthickness=0, font=('arial', 15))
        txtfield4admin.config(insertbackground='black')
        txtfield4admin.place(x=308, y=196)

        # Phone Number
        txtfield5admin = Entry(adminportal, width=22, bd=0, bg="snow", fg="black",
                               highlightthickness=0, font=('arial', 15))
        txtfield5admin.config(insertbackground='black')
        txtfield5admin.place(x=168, y=248)

        # Address
        txtfield6admin = Entry(adminportal, width=27, bd=0, bg="snow", fg="black",
                               highlightthickness=0, font=('arial', 15))
        txtfield6admin.config(insertbackground='black')
        txtfield6admin.place(x=122, y=298)

        # Deletion Id
        txtfield7admin = Entry(adminportal, width=7, bd=0, bg="snow", fg="black",
                               highlightthickness=0, font=('arial', 15))
        txtfield7admin.config(insertbackground='black')
        txtfield7admin.place(x=340, y=419)

        # View Database Button
        view_pic_button = Button(adminportal, command=showviewdb, fg='black', text="View Database", width=8,
                                 borderwidth=1, highlightbackground="turquoise4", pady=2, padx=0)
        view_pic_button.place(x=22, y=435)

        # Update Database Button
        update_pic_button = Button(adminportal, command=update_database, fg='black', text="Update For ID", width=8,
                                   borderwidth=1, highlightbackground="turquoise4", pady=2, padx=0)
        update_pic_button .place(x=22, y=384)

        # Delete Database Button
        delete_pic_button = Button(adminportal, command=delete_database, fg='black', text="Delete Attendance for ID", width=14,
                                   borderwidth=1, highlightbackground="turquoise4", pady=2, padx=0)
        delete_pic_button .place(x=157, y=410)

    else:
        notifier.configure(
            text='CONSOLE: Incorrect Password..')


def showviewdb():
    viewdb = Tk()
    viewdb.title(
        "View Database")
    viewdb.configure(bg='grey14')
    viewdb.geometry('790x600')
    viewdblabelS = Label(viewdb, text="Student", bg='grey14', fg='white', height=1,
                         font=('source sans pro', 20,))
    viewdblabelD = Label(viewdb, text="Details", bg='grey14', fg='white', height=1,
                         font=('source sans pro', 20,))
    viewdblabelS.grid(row=0, column=2)
    viewdblabelD.grid(row=0, column=3)

    blanklabel = Label(viewdb, text="", bg='grey14')
    blanklabel.grid(row=1, column=2)

    rolllabel = Label(viewdb, text="Roll No.", bg='grey14', fg='grey', height=1,
                      font=('source sans pro', 12,))
    rolllabel.grid(row=2, column=0)
    namelabel = Label(viewdb, text="Name", bg='grey14', fg='grey', height=1,
                      font=('source sans pro', 12,))
    namelabel.grid(row=2, column=1)
    genderlabel = Label(viewdb, text="Gender", bg='grey14', fg='grey', height=1,
                        font=('source sans pro', 12,))
    genderlabel.grid(row=2, column=2)
    agelabel = Label(viewdb, text="Age", bg='grey14', fg='grey', height=1,
                     font=('source sans pro', 12,))
    agelabel.grid(row=2, column=3)
    phonelabel = Label(viewdb, text="Phone No.", bg='grey14', fg='grey', height=1,
                       font=('source sans pro', 12,))
    phonelabel.grid(row=2, column=4)
    addresslabel = Label(viewdb, text="Address", bg='grey14', fg='grey', height=1,
                         font=('source sans pro', 12,))
    addresslabel.grid(row=2, column=5)

    cursor.execute('SELECT * FROM student_details')
    ix = 4
    for student in cursor:
        for jx in range(len(student)):
            e = Label(viewdb, width=14, text=student[jx])
            e.grid(row=ix, column=jx)
        ix = ix + 1

    blanklabel2 = Label(viewdb, text="", bg='grey14')
    blanklabel2.grid(row=ix, column=2)

    viewdblabelP = Label(viewdb, text="Python", bg='grey14', fg='white', height=1,
                         font=('source sans pro', 20,))
    viewdblabelS2 = Label(viewdb, text="Students", bg='grey14', fg='white', height=1,
                          font=('source sans pro', 20,))
    viewdblabelD = Label(viewdb, text="DBMS", bg='grey14', fg='white', height=1,
                         font=('source sans pro', 20,))
    viewdblabelS3 = Label(viewdb, text="Students", bg='grey14', fg='white', height=1,
                          font=('source sans pro', 20,))
    viewdblabelP.grid(row=ix + 1, column=0)
    viewdblabelS2.grid(row=ix + 1, column=1)
    viewdblabelD.grid(row=ix + 1, column=4)
    viewdblabelS3.grid(row=ix + 1, column=5)

    blanklabel3 = Label(viewdb, text="", bg='grey14')
    blanklabel3.grid(row=ix + 2, column=2)

    rolllabel2 = Label(viewdb, text="Roll No.", bg='grey14', fg='grey', height=1,
                       font=('source sans pro', 12,))
    rolllabel2.grid(row=ix + 3, column=0)
    namelabel2 = Label(viewdb, text="Name", bg='grey14', fg='grey', height=1,
                       font=('source sans pro', 12,))
    namelabel2.grid(row=ix + 3, column=1)

    ix2 = ix + 4
    cursor.execute(
        'SELECT studid,stdname FROM student_details WHERE studid IN (SELECT studid FROM attends WHERE subid="PyAB");')
    for student in cursor:
        for jx2 in range(len(student)):
            e2 = Label(viewdb, width=14, text=student[jx2])
            e2.grid(row=ix2, column=jx2)
        ix2 = ix2 + 1

    rolllabel3 = Label(viewdb, text="Roll No.", bg='grey14', fg='grey', height=1,
                       font=('source sans pro', 12,))
    rolllabel3.grid(row=ix + 3, column=4)
    namelabel3 = Label(viewdb, text="Name", bg='grey14', fg='grey', height=1,
                       font=('source sans pro', 12,))
    namelabel3.grid(row=ix + 3, column=5)

    ix3 = ix + 4
    cursor.execute(
        'SELECT studid,stdname FROM student_details WHERE studid IN (SELECT studid FROM attends WHERE subid="DbKM");')
    for student in cursor:
        for jx2 in range(len(student)):
            e3 = Label(viewdb, width=14, text=student[jx2])
            e3.grid(row=ix3, column=jx2 + 4)
        ix3 = ix3 + 1

    blanklabel2 = Label(viewdb, text="", bg='grey14')
    blanklabel2.grid(row=ix3 + 1, column=2)

    viewdblabelT = Label(viewdb, text="TCS", bg='grey14', fg='white', height=1,
                         font=('source sans pro', 20,))
    viewdblabelS4 = Label(viewdb, text="Students", bg='grey14', fg='white', height=1,
                          font=('source sans pro', 20,))
    viewdblabelO = Label(viewdb, text="OS", bg='grey14', fg='white', height=1,
                         font=('source sans pro', 20,))
    viewdblabelS5 = Label(viewdb, text="Students", bg='grey14', fg='white', height=1,
                          font=('source sans pro', 20,))
    viewdblabelT.grid(row=ix3 + 2, column=0)
    viewdblabelS4.grid(row=ix3 + 2, column=1)
    viewdblabelO.grid(row=ix3 + 2, column=4)
    viewdblabelS5.grid(row=ix3 + 2, column=5)

    blanklabelx = Label(viewdb, text="", bg='grey14')
    blanklabelx.grid(row=ix3 + 3, column=2)

    rolllabelx = Label(viewdb, text="Roll No.", bg='grey14', fg='grey', height=1,
                       font=('source sans pro', 12,))
    rolllabelx.grid(row=ix3 + 4, column=0)
    namelabelx = Label(viewdb, text="Name", bg='grey14', fg='grey', height=1,
                       font=('source sans pro', 12,))
    namelabelx.grid(row=ix3 + 4, column=1)

    rolllabelx2 = Label(viewdb, text="Roll No.", bg='grey14', fg='grey', height=1,
                        font=('source sans pro', 12,))
    rolllabelx2.grid(row=ix3 + 4, column=4)
    namelabelx2 = Label(viewdb, text="Name", bg='grey14', fg='grey', height=1,
                        font=('source sans pro', 12,))
    namelabelx2.grid(row=ix3 + 4, column=5)

    ixx = ix3 + 5
    cursor.execute(
        'SELECT studid,stdname FROM student_details WHERE studid IN (SELECT studid FROM attends WHERE subid="TcSY");')
    for student in cursor:
        for jx2 in range(len(student)):
            ex = Label(viewdb, width=14, text=student[jx2])
            ex.grid(row=ixx, column=jx2)
        ixx = ixx + 1

    ixx2 = ix3 + 5
    cursor.execute(
        'SELECT studid,stdname FROM student_details WHERE studid IN (SELECT studid FROM attends WHERE subid="OsMR");')
    for student in cursor:
        for jx2 in range(len(student)):
            ex2 = Label(viewdb, width=14, text=student[jx2])
            ex2.grid(row=ixx2, column=jx2 + 4)
        ixx2 = ixx2 + 1

    blanklabelx2 = Label(viewdb, text="", bg='grey14')
    blanklabelx2.grid(row=ixx2 + 2, column=2)

    viewdblabelTx = Label(viewdb, text="Teaching", bg='grey14', fg='white', height=1,
                          font=('source sans pro', 20,))
    viewdblabelFx = Label(viewdb, text="Faculty", bg='grey14', fg='white', height=1,
                          font=('source sans pro', 20,))
    viewdblabelTx.grid(row=ixx2 + 3, column=2)
    viewdblabelFx.grid(row=ixx2 + 3, column=3)

    blanklabelx3 = Label(viewdb, text="", bg='grey14')
    blanklabelx3.grid(row=ixx2 + 4, column=2)

    viewdblabelTxx = Label(viewdb, text="Subject", bg='grey14', fg='grey', height=1,
                           font=('source sans pro', 12,))
    viewdblabelFxx = Label(viewdb, text="Name", bg='grey14', fg='grey', height=1,
                           font=('source sans pro', 12,))
    viewdblabelTxx.grid(row=ixx2 + 5, column=2)
    viewdblabelFxx.grid(row=ixx2 + 5, column=3)

    ixx3 = ixx2 + 6
    cursor.execute(
        'SELECT subject_details.subname, teacher_details.teachname FROM subject_details INNER JOIN teacher_details ON subject_details.teachid = teacher_details.teachid')
    for student in cursor:
        for jx2 in range(len(student)):
            exx = Label(viewdb, width=14, text=student[jx2])
            exx.grid(row=ixx3, column=jx2 + 2)
        ixx3 = ixx3 + 1

    viewdb.mainloop()


"""
Buttons
"""

# Take Pictures Button
take_pic = PhotoImage(file='assets/pic_take.png')
take_pic_button = Button(root, image=take_pic, command=capture_img, borderwidth=0,
                         highlightbackground="light sky blue", highlightthickness=0)
take_pic_button.place(x=172, y=458)

# Train Dataset Button
train_pic = PhotoImage(file='assets/pic_train.png')
train_pic_button = Button(root, image=train_pic, command=train_dataset,
                          borderwidth=0, highlightbackground="gray18", highlightthickness=0)
train_pic_button.place(x=172, y=504)

# Test Dataset Button
test_pic = PhotoImage(file='assets/pic_test.png')
test_pic_button = Button(root, image=test_pic, command=clock_in,
                         borderwidth=0, highlightbackground="bisque4", highlightthickness=0)
test_pic_button.place(x=808, y=363)


# Login
login_pic = PhotoImage(file='assets/login_button.png')
login_pic_button = Button(root, image=login_pic, command=showadminportal,
                          borderwidth=0, highlightbackground="gray18", highlightthickness=0)
login_pic_button.place(x=715, y=545)


"""
Drop Down Menu
"""

options = ["Python", "DBMS", "TCS", "OS"]

clicked = StringVar()
clicked.set(options[0])

drop = OptionMenu(root, clicked, *options)
drop.config(fg="black", bg="white")

drop.place(x=671, y=364)


"""
Text Input Entries
"""

# ID
txtfield1 = Entry(root, width=25, bd=0, bg="snow", fg="black",
                  highlightthickness=0, font=('arial', 15))
txtfield1.config(insertbackground='black')
txtfield1.place(x=144, y=170)

# Name
txtfield2 = Entry(root, width=25, bd=0, bg="snow", fg="black",
                  highlightthickness=0, font=('arial', 15))
txtfield2.config(insertbackground='black')
txtfield2.place(x=144, y=218)

# Gender
txtfield3 = Entry(root, width=8, bd=0, bg="snow", fg="black",
                  highlightthickness=0, font=('arial', 15))
txtfield3.config(insertbackground='black')
txtfield3.place(x=114, y=267)

# Age
txtfield4 = Entry(root, width=8, bd=0, bg="snow", fg="black",
                  highlightthickness=0, font=('arial', 15))
txtfield4.config(insertbackground='black')
txtfield4.place(x=300, y=267)

# Phone Number
txtfield5 = Entry(root, width=22, bd=0, bg="snow", fg="black",
                  highlightthickness=0, font=('arial', 15))
txtfield5.config(insertbackground='black')
txtfield5.place(x=168, y=317)

# Address
txtfield6 = Entry(root, width=27, bd=0, bg="snow", fg="black",
                  highlightthickness=0, font=('arial', 15))
txtfield6.config(insertbackground='black')
txtfield6.place(x=122, y=365)


# Username
txtfieldusername = Entry(root, width=14, bd=0, bg="snow", fg="black",
                         highlightthickness=0, font=('arial', 15))
txtfieldusername.config(insertbackground='black')
txtfieldusername.place(x=698, y=475)


# Password
txtfieldpassword = Entry(root, show='*', width=14, bd=0, bg="snow", fg="black",
                         highlightthickness=0, font=('arial', 15))
txtfieldpassword.config(insertbackground='black')
txtfieldpassword.place(x=698, y=510)


"""
Notification Bar
"""

notifier = Label(root, text="CONSOLE: null..", fg='medium sea green', anchor='w',
                 bg='grey10', width=138, height=1, font=('source code pro', 14,))

notifier.place(x=0, y=578)


############################### CSV BAR GUI #################################


style = ttk.Style(root)
style.theme_use("clam")
style.configure("Treeview", background="gray14",
                fieldbackground="gray14", foreground="snow")


TableMargin = Frame(root, width=340, height=315,
                    borderwidth=12, highlightthickness=0)
TableMargin.pack(padx=(590, 25), pady=(85, 260))


tree = ttk.Treeview(TableMargin, columns=(
    "Id", "Name", "Subject", "Date", "Time"), height=400, selectmode="extended")


tree.heading('Id', text="Id", anchor=W)

tree.heading('Name', text="Name", anchor=W)

tree.heading('Subject', text="Subject", anchor=W)

tree.heading('Date', text="Date", anchor=W)

tree.heading('Time', text="Time", anchor=W)

tree.column('#0', stretch=NO, minwidth=0, width=0)
tree.column('#1', stretch=NO, minwidth=0, width=48)
tree.column('#2', stretch=NO, minwidth=0, width=94)
tree.column('#3', stretch=NO, minwidth=0, width=50)
tree.column('#4', stretch=NO, minwidth=0, width=70)
tree.column('#5', stretch=NO, minwidth=0, width=70)

tree.pack()

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')


def csv_updater():
    tree.delete(*tree.get_children())
    try:
        with open("daily_generated_attendance_csv/daily_generated_attendance_csv_" + date + ".csv") as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                Id = row['Id']
                Name = row['Name']
                Subject = row['Subject']
                Date = row['Date']
                Time = row['Time']
                tree.insert("", 0, values=(Id, Name, Subject, Date, Time))
        notifier.configure(
            text='CONSOLE: Attendance Records Found For ' + date)
    except:
        tree.insert("", 0, values=('', '', '', '', ''))
        notifier.configure(
            text='CONSOLE: No Attendance Records Found For ' + date)


csv_updater()


root.mainloop()
