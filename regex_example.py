# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:59:03 2018

@author: dxuser22
"""

import re

text = 'Cookie+Cookle-Cookae'

regex = re.search(r'C..k.e', text).group()
regex2 = re.findall(r'C..k.e', text)

print(regex)

re.search(r'c\d\dkie', 'c00kie').group()

re.search(r'Number: [0-6]', 'Number: 5').group()

# This treats '\s' as an escape character because it lacks '\' at the start of '\s'
re.search(r'Back\stail', 'Back tail').group()

heading  = r'<h1>TITLE</h1>'
re.match(r'<.*>', heading).group()

re.match(r'<.*?>', heading).group()

re.search(r'<.*>', heading).group()


#The match() function checks for a match only at the beginning of the string (by default) whereas the search() function checks for a match anywhere in the string.

pattern = "C"
sequence1 = "IceCream"

# No match since "C" is not at the start of "IceCream"
re.match(pattern, sequence1)
re.search(pattern, sequence1)



email_address = "Please contact us at: support@datacamp.com, xyz@datacamp.com, support@datacampcom, sup.port@datacamp.com, sup.port@, sup-port@data-camp.com"

#'addresses' is a list that stores all the possible match
addresses = re.findall(r'[\w\.-]+@[\w\.-]+', email_address)
#addresses = re.findall(r'[\w\.]+@[\w\.]+', email_address)
for address in addresses: 
    print(address)
    
    
re.findall(r'[\w.]+@[\w.]+', 'support-center@xyz.com')
#However, to match the entire 'support-center@xyz.com' you will have to include the dash symbol '-' within the '[]'.  


col = '2016_price_delta_pct'

bu_size_code = re.match(r'\d+', col).group()

col2 = '2016_vs_12335_cross_elas'

bu_size_codes = re.findall(r'\d+', col2)