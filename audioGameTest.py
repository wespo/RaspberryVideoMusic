#audioGameTest
# import the pygame module, so you can use it
import pygame
import pyaudio
import numpy as np
from scipy import signal
from scipy.fftpack import rfft, fftshift
import time
import random

chunk=4096
RATE=44100
WIDTH=640
HEIGHT=480
S_t = 0

def timeSignal(data, screen):
	downsampled = signal.resample(data, WIDTH)
	downsampled = downsampled/20
	screen.fill((0,0,0))
	count = 0;
	lastsample = 0;
	for sample in downsampled:
		pygame.draw.rect(screen, (0, 255, 255), pygame.Rect(count, HEIGHT/2-lastsample, 1, lastsample-sample))
		lastsample = sample
		count = count + 1
def fftSignal(data, screen):
	global S_t
	fft_data = np.absolute(rfft(data))/100
	downsampled = signal.resample(fft_data, WIDTH)
	downsampled = downsampled/20
	colorRed = round(sum(fft_data[0:5])/6)*6
	if colorRed > S_t and colorRed > 10:
		S_t = colorRed
	else:
		S_t = 0.66*S_t

	screen.fill((min(255,S_t),0,0))
	count = 0;
	lastsample = 0;
	for sample in downsampled:
		if lastsample > sample:
			bottom = lastsample
			height = lastsample - sample
		else:
			bottom = sample
			height = sample - lastsample
		pygame.draw.rect(screen, (0, 255, 255), pygame.Rect(count, HEIGHT-bottom, 1, height))
		lastsample = sample
		count = count + 1
currentDisplay = timeSignal
displayFunctions =[timeSignal,fftSignal]
nextChangeTime = time.time()

# define a main function
def main():
	# initialize the pygame module
	pygame.init()
	# load and set the logo
	logo = pygame.image.load("logo32x32.png")
	pygame.display.set_icon(logo)
	pygame.display.set_caption("minimal program")

	# create a surface on screen that has the size of 640 x 480
	screen = pygame.display.set_mode((WIDTH,HEIGHT))
	pygame.draw.rect(screen, (0, 255, 255), pygame.Rect(10, 10, 20, 20))
	pygame.display.update()
	# define a variable to control the main loop
	running = True
	#audio handle
	p=pyaudio.PyAudio()
	def processBuffer(in_data, frame_count, time_info, status):
		global nextChangeTime,currentDisplay,displayFunctions
		#data=np.fromstring(stream.read(chunk,exception_on_overflow = False),dtype=np.int16)
		data = np.fromstring(in_data, dtype=np.int16)
		raw_data = data
		currentTime = time.time();
		if(currentTime > nextChangeTime):
			currentDisplay = random.choice(displayFunctions)
			nextChangeTime = currentTime + random.randrange(3,7)
		currentDisplay(data, screen)
		pygame.display.update()
		return (raw_data, pyaudio.paContinue)
	
	#input stream setup
	stream=p.open(format = pyaudio.paInt16,rate=RATE,channels=1, input_device_index = 2, input=True, frames_per_buffer=chunk, stream_callback=processBuffer) #on Win7 PC, 1 = Microphone, 2 = stereo mix (enabled in sound control panel)

	# main loop
	while running:
		# event handling, gets all event from the event queue
		for event in pygame.event.get():
			# only do something if the event is of type QUIT
			if event.type == pygame.QUIT:
				# change the value to False, to exit the main loop
				running = False
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__=="__main__":
    # call the main function
    main()