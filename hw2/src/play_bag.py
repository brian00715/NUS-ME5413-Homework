#!/usr/bin/env python
import rospy
from tf2_msgs.msg import TFMessage
import rosbag
from sensor_msgs.msg import LaserScan

ignored_frames = ["left_wheel_link", "right_wheel_link"]


def publish_tf(publisher, message):
    new_message = TFMessage()
    for transform in message.transforms:
        if transform.child_frame_id not in ignored_frames:
            new_message.transforms.append(transform)
    publisher.publish(new_message)


def main():
    rospy.init_node("tf_filter")

    bag = rosbag.Bag(r"/home/simon/LocalDiskExt/Datasets/HW2_SLAM/Task 1/2dlidar.bag", "r")
    tf_pub = rospy.Publisher("/tf_filtered", TFMessage, queue_size=10)
    tf_static_pub = rospy.Publisher("/tf_static_filtered", TFMessage, queue_size=10, latch=True)
    scan_pub = rospy.Publisher("/scan", LaserScan, queue_size=10)

    while not rospy.is_shutdown():
        for topic, message, t in bag.read_messages(topics=["scan", "tf", "tf_static"]):
            print(message)
            print("\rTime: %f" % t.to_sec(), end="")
            if topic == "tf":
                publish_tf(tf_pub, message)
            elif topic == "tf_static":
                publish_tf(tf_static_pub, message)
            elif topic == "scan":
                scan_pub.publish(message)

    rospy.spin()


if __name__ == "__main__":
    main()
