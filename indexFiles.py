IGNORE_FOLDERS = ["/sys"]
import os,sys
# FolderDict = dict()

def logger(*argv,logfile="log.txt",singleLine = False):
  """
  Writes Logs to LogFile
  """
  try:
    with open(logfile, 'a+') as f:
      for arg in argv:
        if type(arg) == dict:
          for key,val in arg.items():
            f.write(str(key) + " : "+ str(val))
            f.flush()
        else:
          f.write(str(arg))
          f.flush()
        if singleLine==False:
          f.write("\n")
      f.write("\n")
  except Exception as e:
    print('Logger crashed for Args: ',argv)
    pass

def recurseDir(path,thresh = 500):
  # global FolderDict
  contents = os.listdir(path)
  length = len(contents)
  run = False if length>thresh else True
  if path in IGNORE_FOLDERS:
    run=False
  string_path = str(path)
  end_part = string_path.split('\\')[-1]
  run = False if end_part.startswith('~')  or end_part.startswith('$')or end_part=='Thumbs.db' else run
  if run==False:
    print("Ignoring :",path)
    print(length,"SubFiles and SubFolders Found, Greater than Threshold.")
  if run==True:
    print("Exploring :",path)
    print(length,"SubFiles and SubFolders Found")

    logger("FolderName :",path," Contents :",contents,singleLine=True)
    # FolderDict[path] = contents
    for each_path in contents:
      epath = os.path.join(path,each_path)
      if os.path.isdir(epath):
        try:
          recurseDir(epath)
        except:
          pass
recurseDir(path=os.getcwd()) # os.getcwd()

# print(FolderDict)
#f.startswith('~') and f!='Thumbs.db'
# try files and folders separately thresholding