 #! /bin/bash

 # Make BT Disoverable
 #hciconfig hci0 piscan

 # listen to RFCOMM Port 3
 rfcomm listen /dev/rfcomm0 3 &

 #chmod 777 /dev/rfcomm0
