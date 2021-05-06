"""

Email functions for Python automation task

"""
#################################### libraries and modules ####################################
# region

import typing
import smtplib
import win32com.client
from datetime import date
import pandas as pd
import glob
import os

# endregion
#################################### email functions ####################################
# region

def email_out(from_address: typing.List[str],
              receiver_list: typing.List[str],
              subject: str,
              message: str) -> None:
    """
    Function to email out using SMTP service
    :param from_address: email address to send from
    :param receiver_list: email address(s) to send to
    :param subject: subject of email
    :param message: message body
    :return: None
    """
    # server name
    server_name = 'smtp.hms.hmsy.com'
    from_address = f'{from_address}@hms.com'

    # prep message
    message = f"""Subject: {subject} \n\n
    {message}
    """

    # send email
    server = smtplib.SMTP(server_name)
    server.sendmail(from_address, receiver_list, message)
    server.quit()


def get_email_att(folder_name, att_name, archive_path):
    """
    Function to save CSVs from emails to archive, will check against archive to make sure it's a new file.

    Note:
        - Ignore archive sections if there isn't one yet
        - Section to add date this is run to df in archive

    Dependencies:
        - import win32com.client
        - from datetime import date
        - import pandas as pd
        - import glob
        - import os

    Author(s):
        - Maddie Karwich
    """
    # We need to grab the last attachment that we saved off to check to see if there was a new email attachment or same
    list_of_files = glob.glob(archive_path + '*.csv')
    last_att = max(list_of_files, key=os.path.getctime)
    last_att_df = pd.read_csv(last_att)

    # Opens Microsoft Outlook, gets namespace
    Outlook = win32com.client.Dispatch('Outlook.Application').GetNamespace('MAPI')

    # Gets to subfolder that will have the emails with the Return Mail Report
    inbox = Outlook.GetDefaultFolder(6)
    subFolder = inbox.Folders(folder_name)

    # Get most recent message in folder
    subFolderMessages = subFolder.Items
    last_message = subFolderMessages.GetLast()

    # Get attachments (if they exist, won't fail if there are none)
    # (BUT- This would be if you wanted the message: body_content = last_message.body)
    attachments = last_message.Attachments

    # Find today (date report is for), reformat for file name
    today = date.today()
    today = today.strftime("%Y_%m_%d")

    # Save attachment(s) into archive
    atts = []
    countatt = attachments.count + 1
    for i in range(1, countatt):
        attachment = attachments.Item(i)
        att = archive_path + 'att_name' + today + '_att_' + str(i) + '.csv'
        attachment.SaveAsFile(att)
        atts = atts.append(att)

        # Put csv attachment into df
    try:
        att
    except NameError:
        print('No attachments on most recent email.')
    else:
        att_df = pd.read_csv(att)
        # Check to see if the dfs are the same (AKA there's not a new email)
        if att_df.equals(last_att_df) is False:
            # Add column that will contain today's date
            att_df['EMAIL_DT'] = date.today()
            # Master data file for returned mail
            master_csv = path + 'Master_Data.csv'
            # Append today's df to existing master csv
            att_df.to_csv(master_csv, mode='a', header=False, index=False)
        else:
            # Remove the files in archive since they're dups
            for i in atts:
                os.remove(i)
            print('No new email.')

# endregion

# EOF
