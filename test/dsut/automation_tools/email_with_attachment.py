# -*- coding: utf-8 -*-
"""
Constructing email with attachment
See orig at https://andrewpwheeler.com/2015/03/02/emailing-with-python-and-spss/

See email_functions to actually open up 
server and send email from HMS

@author: Andy Wheeler
"""

from os.path import basename
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import COMMASPACE, formatdate

def make_mail(send_from, send_to, subject, text, files=None):
    assert isinstance(send_to, list)

    msg = MIMEMultipart(
        From=send_from,
        To=COMMASPACE.join(send_to),
        Date=formatdate(localtime=True),
        Subject=subject
    )
    msg.attach(MIMEText(text))

    if files != None:
      for f in files:              
        part = MIMEBase('application', 'base64')
        part.set_payload(open(f,"rb").read())
        part.add_header('Content-Disposition', 'attachment', filename=basename(f))    
        encoders.encode_base64(part)      
        msg.attach(part)
    return msg  #.as_string()

#You still need to open up the server to send the email
#This just formats it nicely and attaches a file if you want

#This is giving me problems not recognizing the header content
#Need to build in 
#From: Person <email@hms.com>
#To: PA <email@hms.com>, PB <email@hms.com>
#Subject: Whatever
#????Message here????
#Logic into the message to get it to recognize correctly