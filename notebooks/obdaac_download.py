#!/usr/bin/env python
#
# Requires a valid .netrc file in the user home ($HOME), e.g.:
#    machine urs.earthdata.nasa.gov login USERNAME password PASSWD
#
# from obdaac_download import httpdl
#
# server = 'oceandata.sci.gsfc.nasa.gov'
# request = '/ob/getfile/T2017004001500.L1A_LAC.bz2'
#
# status = httpdl(server, request, uncompress=True)
#
from contextlib import closing
import os
import sys
import re
import argparse
import subprocess
import logging
import requests
from requests.adapters import HTTPAdapter
from datetime import datetime, timedelta
import time
from urllib.parse import urlparse

DEFAULT_CHUNK_SIZE = 131072

# requests session object used to keep connections around
obpgSession = None

def getSession(verbose=0, ntries=5):
    global obpgSession

    if not obpgSession:
        # turn on debug statements for requests
        if verbose > 1:
            print("Session started")
            logging.basicConfig(level=logging.DEBUG)

        obpgSession = requests.Session()
        obpgSession.mount('https://', HTTPAdapter(max_retries=ntries))
            
    else:
        if verbose > 1:
            print("Reusing existing session")

    return obpgSession

def isRequestAuthFailure(req) :
    ctype = req.headers.get('Content-Type')
    if ctype and ctype.startswith('text/html'):
        if "<title>Earthdata Login</title>" in req.text:
            return True
    return False

def httpdl(server, request, localpath='.', outputfilename=None, ntries=5,
           uncompress=False, timeout=30., verbose=0, force_download=False,
           chunk_size=DEFAULT_CHUNK_SIZE):

    status = 0
    urlStr = 'https://' + server + request

    global obpgSession

    getSession(verbose=verbose, ntries=ntries)

    modified_since = None
    headers = {}

    if not force_download:
        if outputfilename:
            ofile = os.path.join(localpath, outputfilename)
            modified_since = get_file_time(ofile)
        else:
            ofile = os.path.join(localpath, os.path.basename(request.rstrip()))
            modified_since = get_file_time(ofile)

        if modified_since:
            headers = {"If-Modified-Since":modified_since.strftime("%a, %d %b %Y %H:%M:%S GMT")}

    with closing(obpgSession.get(urlStr, stream=True, timeout=timeout, headers=headers)) as req:

        if req.status_code != 200:
            status = req.status_code
        elif isRequestAuthFailure(req):
            status = 401
        else:
            if not os.path.exists(localpath):
                os.umask(0o02)
                os.makedirs(localpath, mode=0o2775)

            if not outputfilename:
                cd = req.headers.get('Content-Disposition')
                if cd:
                    outputfilename = re.findall("filename=(.+)", cd)[0]
                else:
                    outputfilename = urlStr.split('/')[-1]

            ofile = os.path.join(localpath, outputfilename)

            # This is here just in case we didn't get a 304 when we should have...
            #Tue, 11 Dec 2012 10:10:24 GMT
            download = True
            if 'last-modified' in req.headers:
                remote_lmt = req.headers['last-modified']
                remote_ftime = datetime.strptime(remote_lmt, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=None)
                if modified_since and not force_download:
                    if (remote_ftime - modified_since).total_seconds() < 0:
                        download = False
                        if verbose:
                            print("Skipping download of %s" % outputfilename)

            if download:
                with open(ofile, 'wb') as fd:
                    for chunk in req.iter_content(chunk_size=chunk_size):
                        if chunk: # filter out keep-alive new chunks
                            fd.write(chunk)

                if uncompress and re.search(".(Z|gz|bz2)$", ofile):
                    compressStatus = uncompressFile(ofile)
                    if compressStatus:
                        status = compressStatus
                else:
                    status = 0

    return status


def uncompressFile(compressed_file):
    """
    uncompress file
    compression methods:
        bzip2
        gzip
        UNIX compress
    """

    compProg = {"gz": "gunzip -f ", "Z": "gunzip -f ", "bz2": "bunzip2 -f "}
    exten = os.path.basename(compressed_file).split('.')[-1]
    unzip = compProg[exten]
    p = subprocess.Popen(unzip + compressed_file, shell=True)
    status = os.waitpid(p.pid, 0)[1]
    if status:
        print("Warning! Unable to decompress %s" % compressed_file)
        return status
    else:
        return 0

def get_file_time(localFile):
    ftime = None
    if not os.path.isfile(localFile):
        localFile = re.sub(r".(Z|gz|bz2)$", '', localFile)

    if os.path.isfile(localFile):
        ftime = datetime.fromtimestamp(os.path.getmtime(localFile))

    return ftime

def retrieveURL(request,localpath='.', uncompress=False, verbose=0,force_download=False):
        if args.verbose:
            print("Retrieving %s" % request.rstrip())

        server = "oceandata.sci.gsfc.nasa.gov"
        parsedRequest = urlparse(request)
        netpath = parsedRequest.path
        if parsedRequest.netloc:
            server = parsedRequest.netloc
        else:
            if not re.match("getfile",netpath):
                netpath = '/ob/getfile/' + netpath

        if parsedRequest.query:
            netpath = netpath + '?' + parsedRequest.query

        return httpdl(server, netpath, localpath=localpath, uncompress=uncompress, verbose=verbose,force_download=force_download)


if __name__ == "__main__":
    # parse command line
    parser = argparse.ArgumentParser(
        description='Download files archived at the OB.DAAC',
        epilog='''
Provide one of either filename, --filelist or --http_manifest.
NOTE: For authentication, a valid .netrc file in the user home ($HOME) is required, e.g.:
    machine urs.earthdata.nasa.gov login USERNAME password PASSWD
'''
    )
    parser.add_argument('-v', '--verbose', help='print status messages',
                        action='count',default=0)
    parser.add_argument('filename', nargs='?', help='name of the file (or the URL of the file) to retreive')
    parser.add_argument('--filelist',
                        help='file containing list of filenames to retreive, one per line')    
    parser.add_argument('--http_manifest',
                        help='URL to http_manifest file for OB.DAAC data order')
    parser.add_argument('--odir',
                        help='full path to desired output directory; defaults to current working directory: %s' % os.getcwd(),
                        default=os.getcwd())
    parser.add_argument('--uncompress',action="store_true",
                        help='uncompress the retrieved files (if compressed)',
                        default=False)
    parser.add_argument('--force',action='store_true',
                        help='force download even if file already exists locally',
                        default=False)
    args = parser.parse_args()

    filelist = []

    if args.http_manifest:
        status = retrieveURL(args.http_manifest,verbose=args.verbose,force_download=True)
        if status:
            print("Whoops...problem retrieving %s (received status %d)" % (args.http_manifest,status))
            sys.exit("Bailing out...")
        else:
            with open('http_manifest.txt') as flist:
                for filename in flist:
                    filelist.append(filename.rstrip())
    elif args.filename:
        filelist.append(args.filename)
    elif args.filelist:
        with open(args.filelist) as flist:
            for filename in flist:
                filelist.append(filename.rstrip())

    if not len(filelist):
        parser.print_usage()
        sys.exit("Please provide a filename (or list file) to retrieve")

    outpath = os.path.abspath(os.path.expanduser(os.path.expandvars(args.odir)))
    if args.verbose:
        print("Output directory: %s" % outpath)

    for request in filelist:
        status = retrieveURL(request,localpath=outpath, uncompress=args.uncompress, verbose=args.verbose,force_download=args.force)
        if status:
            if status == 304: 
                if args.verbose:
                    print("%s is not newer than local copy, skipping download" % request)
            else:
                print("Whoops...problem retrieving %s (received status %d)" % (request,status))
