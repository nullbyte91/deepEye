import rospy
from std_msgs.msg import String

phrases_list = ["slow down a little",
                "take this right",
                "you missed your turn again!",
                "A car gonna take a left, slow down",
                "Red Signal on",
                "Green Signal on, You can walk now",
                "take this right",
                "you missed your turn again!",
                "A car gonna take a left, slow down",
                "Red Signal on",
                "Green Signal on, You can walk now",
                "take this right",
                "you missed your turn again!",
                "A car gonna take a left, slow down",
                "Red Signal on",
                "Green Signal on, You can walk now",
                "take this right",
                "you missed your turn again!",
                "A car gonna take a left, slow down",
                "Red Signal on",
                "Green Signal on, You can walk now",
                "take this right",
                "you missed your turn again!",
                "A car gonna take a left, slow down",
                "Red Signal on",
                "Green Signal on, You can walk now",
                "take this right",
                "you missed your turn again!",
                "A car gonna take a left, slow down",
                "Red Signal on",
                "Green Signal on, You can walk now",
                ]

def txt_simulator():
    rospy.init_node('publisher', anonymous=True)
    pub = rospy.Publisher('txt_simulator', String, queue_size=10)
    rate = rospy.Rate(10) # 10hz

    for i in range(len(phrases_list)):
        rospy.loginfo(phrases_list[i])
        pub.publish(phrases_list[i])
        rate.sleep()

if __name__ == '__main__':
    try:
        txt_simulator()
    except rospy.ROSInterruptException:
        pass