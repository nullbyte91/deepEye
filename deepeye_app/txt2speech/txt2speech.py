import socket
import logging
import sched
import time 
from threading import Thread, Lock

# ROS
import rospy
from std_msgs.msg import String

# Text 2 speech
from google_speech import Speech

import subprocess

lang = "en"

# Init with False
networkConnection = False

# schedular instance is created 
s = sched.scheduler(time.time, time.sleep)

# Mutex 
mutex = Lock()

# BT Connected flag
bt_flag = True

def isInternetConnected():
    global networkConnection

    # Mutex lock before CS
    mutex.acquire()
    
    try:
        # Try to ping google
        # To do: Is there way we can do local ping rather than google
        sock = socket.create_connection(("www.google.com", 80))
        if sock is not None:
            sock.close
            # On Success update the status 
            networkConnection = True
            print("Network ping ... success")
    except OSError:
        # On Failure update
        networkConnection = False
    
    # Mutex unlock after CS
    mutex.release()

    # Recursive call
    s.enter(1, 1, isInternetConnected, ())

def callback(data):
    global networkConnection
    print("Debug Data: ".format(data))

    if bt_flag == False:
        if networkConnection == True:
            # Google speech block
            speech = Speech(data.data, lang)
            speech.play()
            time.sleep(0.5)
        else:
            # On device mycroft mimic v1
            p = subprocess.Popen(["/home/nullbyte/Desktop/myGit/mimic1/mimic", data.data], stdout = subprocess.PIPE)
            (output, err) = p.communicate() 
            p_status = p.wait()
    else:
        cmd = "echo " + data.data + " > /dev/rfcomm0"
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        (stderr,stdout) = p.communicate()
        print("stderr: ".format(stderr))
        print("stdout: ".format(stdout))

def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    # ROS callback
    rospy.init_node('subscriper', anonymous=True)

    rospy.Subscriber("depthai", String, callback)

    s.enter(1, 1, isInternetConnected, ())
    s.run()
    
    rospy.spin()

if __name__ == "__main__":
    main()