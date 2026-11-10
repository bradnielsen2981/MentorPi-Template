import json
import urllib.error
import urllib.parse
import urllib.request

#---EXTERNAL URL REQUEST LIBRARY---------------------------------------#
# All functions return None (and print the reason) if the request fails,
# so a network problem won't crash the robot's main loop.

DEFAULT_TIMEOUT = 10 # seconds to wait before giving up on a server

def _send(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            charset = response.headers.get_content_charset() or 'utf-8'
            return response.read().decode(charset)
    except urllib.error.HTTPError as e:
        print("URL request failed: " + req.full_url + " returned HTTP " + str(e.code) + " " + str(e.reason))
    except urllib.error.URLError as e:
        print("URL request failed: could not reach " + req.full_url + " (" + str(e.reason) + ")")
    except (TimeoutError, OSError) as e:
        print("URL request failed: " + req.full_url + " (" + str(e) + ")")
    return None

# POST form data (e.g. {'name': 'robot', 'score': 5}) and return the response text
def sendurlrequest(url, dictofvalues=None, timeout=DEFAULT_TIMEOUT):
    data = urllib.parse.urlencode(dictofvalues or {}).encode('utf-8') # data must be bytes
    req = urllib.request.Request(url, data=data, method='POST')
    return _send(req, timeout)

# GET a url, adding dictofvalues to the end as ?key=value&... and return the response text
def geturlrequest(url, dictofvalues=None, timeout=DEFAULT_TIMEOUT):
    if dictofvalues:
        separator = '&' if '?' in url else '?'
        url = url + separator + urllib.parse.urlencode(dictofvalues)
    req = urllib.request.Request(url, method='GET')
    return _send(req, timeout)

# POST dictofvalues as JSON and return the response as a python dict/list
# (or the plain text if the server didn't reply with JSON)
def sendjsonrequest(url, dictofvalues=None, timeout=DEFAULT_TIMEOUT):
    data = json.dumps(dictofvalues or {}).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST',
                                 headers={'Content-Type': 'application/json'})
    responsedata = _send(req, timeout)
    if responsedata is None:
        return None
    try:
        return json.loads(responsedata)
    except ValueError:
        return responsedata
