import socket
import logging
import sched
import time 
from threading import Thread, Lock

# ROS
import rospy
from std_msgs.msg import String

# Init with False
networkConnection = False

# schedular instance is created 
s = sched.scheduler(time.time, time.sleep)

# Mutex 
mutex = Lock()

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
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    

def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    # ROS callback
    rospy.init_node('subscriper', anonymous=True)

    rospy.Subscriber("txt_simulator", String, callback)

    s.enter(1, 1, isInternetConnected, ())
    s.run()
    
    rospy.spin()

if __name__ == "__main__":
    main()