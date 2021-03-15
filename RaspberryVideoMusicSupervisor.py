import os
import time
import setproctitle

setproctitle.setproctitle("PVMSupervisor")
restartCount = 0
while True:
    stream = os.popen("ps -e | grep PiVideoMusic")
    processes = stream.read()
    processes = processes.split('\n')
    processes.pop() #remove empty last line
    if len(processes) == 0: #no uniac process running. Launch.
        restartCount += 1
        print("Did not find living PiVideoMusic process. Oh No! Launching PiVideoMusic. Restart count = {} at time: {}".format(restartCount, time.asctime(time.localtime())))
        os.popen("/usr/bin/python3 -u /home/pi/RaspberryVideoMusic/RaspberryVideoMusic.py 1>>/home/pi/RaspberryVideoMusic/RaspberryVideoMusic_log.txt 2>&1 &")
        print("OK, sleeping.")
    else:
        processes.pop(0) #pop first UNIAC process
        for process in processes:
                print("Found living {} PiVideoMusic process: {}".format(len(processes)+1,process))
                os.popen("kill {}".format(process.split[0]))
                print("All is well. Sleeping.")
    time.sleep(20)