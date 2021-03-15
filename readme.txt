#setup Rapsberry Video Music:

1. Install Raspbian
2. Using Raspi-Config: Set up Wifi, Password, Keyboard, Enable SSH
3. SFTP RaspberryVideoMusicSupervisor.py and RaspberryVideoMusic.py to the device
4. Install required Python Packages:
	pip3: sudo apt install python3-pip
	pygame: sudo apt-get install python3-pygame
	pyaudio: sudo apt-get install python3-pyaudio 
	scipy: sudo apt install -y python3-scipy
	setproctitle: sudo pip3 install setproctitle
5. Setup Audio: USB soundcard should automatically be setup.
6. Select the correct audio device: Likely automatically found, but if not:
	run 'sudo python3 RaspberryVideoMusic/RaspberryVideoMusic.py -l' and look for the table that has the header 
	'ID      Name                                            Input Ch        Output Ch       Sample Rate     Sel' 
	in the printout.
7. Setup Crontab: edit 'sudo nano /etc/rc.local' and add the following lines NEAR the end, before 'exit 0'!
	#start up Raspberry Video Music Supervisor
	sudo python3 /home/pi/RaspberryVideoMusic/RaspberryVideoMusicSupervisor.py 1>>/home/pi/RaspberryVideoMusic/RaspberryVideoMusicSupervisor_log.txt 2>&1 &