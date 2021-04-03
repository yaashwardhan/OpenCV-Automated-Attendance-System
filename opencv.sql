create database opencv;
use opencv;

create table student_details(
    studid varchar(10) primary key,
    stdname varchar(25),
    gender varchar(10),
    age varchar(10),
    phoneno varchar(10),
    address varchar(30)
);

create table subject_details(
    subname varchar(25),
    subid varchar(10) primary key,
    teachid varchar(10),
    constraint fk_teachid
	foreign key (teachid)
	references teacher_details(teachid)
);

create table attends(
	studid varchar(10),
    constraint fk_studid
	foreign key(studid)
	references student_details(studid),
    subid varchar(10),
    constraint fk_subid
	foreign key(subid)
	references subject_details(subid),
    attdate varchar(15),
    atttime varchar(15)
);

create table teacher_details(
	teachid varchar(10) primary key,
	teachname varchar(25),
    teachaddress varchar(30)
);


insert into teacher_details values
(1,'Ms. Ameyaa Biwalkar','Mumbai'),
(2,'Sir Kamal Mistry','Mumbai'),
(3,'Ms. Supriya Yadav','Banglore'),
(4,'Ms. Mohini Reddy','Pune');

insert into subject_details values
('Python', 'PyAB', '1'),
('DBMS', 'DbKM', '2'),
('TCS', 'TcSY', '3'),
('OS', 'OsMR', '4');

select * from student_details;

select * from teacher_details;

select * from subject_details;

select * from attends;

drop table attends;

drop table subject_details;

drop table teacher_details;

drop table student_details;

DELETE from student_details;
