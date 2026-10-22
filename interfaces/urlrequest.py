import urllib.parse
import urllib.request
import time
import torch

#---EXTERNAL URL REQUEST LIBRARY---------------------------------------#
def sendurlrequest(url, dictofvalues):
    responsedata = None
    data = urllib.parse.urlencode(dictofvalues)
    data = data.encode('ascii') # data should be bytes
    req = urllib.request.Request(url, data)
    #if data is returned
    with urllib.request.urlopen(req) as response:
        responsedata = response.read().decode('utf-8')
    return responsedata