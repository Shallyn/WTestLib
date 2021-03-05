#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed, 06 Jan 2021 02:29:03 +0000

@author: Shallyn
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.header import Header

pwd = Path(__file__).absolute().parent
def send_email(title, content, header = None, sender = None, receivers = None):
    if sender is None:
        sender = 'bnu.server@reminder.com'
    if receivers is None:
        receivers = ['201731160003@mail.bnu.edu.cn']  # 接收邮件，可设置为你的QQ邮箱或者其他邮箱
    if header is None:
        header = 'bnuserver'
    # 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(header, 'utf-8')   # 发送者
    message['To'] =  Header("reciver", 'utf-8')        # 接收者

    subject = title
    message['Subject'] = Header(subject, 'utf-8')


    try:
        smtpObj = smtplib.SMTP('localhost')
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("email sended")
    except smtplib.SMTPException:
        print("Error: fail to send email")
