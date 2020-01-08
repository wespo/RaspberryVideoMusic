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
peak = 0

peakDiamondSize = 5
peakArraySize = (2*peakDiamondSize-1)
peakArray = [0] * peakArraySize #np.zeros((1,2*peakDiamondSize-1))

meanDiamondSize = 32
meanArraySize = (2*meanDiamondSize-1)
meanArray = [0] * meanArraySize #np.zeros((1,2*peakDiamondSize-1))

tStart = time.time()
def tic():
	global tStart
	tic = time.time()
	print("Interval:" + str(round(1000*(tic-tStart))), end= '\t')
	tStart = tic
def toc():
	global tStart
	print("Frame Time:" + str(round(1000*(time.time()-tStart))))
def meanDiamonds(data,screen):
	global peak, peakArray
	#tic()
	
	screen.fill((0,0,0))
	lastsample = 0;
	
	peak = np.exp(np.mean(data)*3)
	# if max(downsampled) > peak:
	# 	peak = max(downsampled) * 3
	# else:
	# 	peak = peak * 0.25

	meanArray.pop(0)
	meanArray.append(peak)
	
	arrayRows = meanDiamondSize
	bufferCols = meanArraySize
	bufferArray = meanArray
	resultMatrix = np.array([bufferArray.copy()[arrayRows-1:bufferCols]])
	for offs in range(1,arrayRows):
		resultMatrix = np.append(resultMatrix,[bufferArray.copy()[arrayRows-1-offs:bufferCols-offs]], axis = 0)
	
	surf = pygame.surfarray.make_surface(resultMatrix)
	surfa = pygame.transform.scale(surf,(320,240))
	surfb = pygame.transform.rotate(surf,90)
	surfb = pygame.transform.scale(surfb,(320,240))
	surfc = pygame.transform.rotate(surf,180)
	surfc = pygame.transform.scale(surfc,(320,240))
	surfd = pygame.transform.rotate(surf,270)
	surfd = pygame.transform.scale(surfd,(320,240))
	
	screen.blit(surfb, (0,0))
	screen.blit(surfa, (320,0))
	screen.blit(surfc, (0,240))
	screen.blit(surfd, (320,240))
	# screen.blit(surfa, (pxlcount*(idxX+1), pxlcount*idxY))
	# screen.blit(surfc, (pxlcount*idxX, pxlcount*(idxY+1)))
	# screen.blit(surfd, (pxlcount*(idxX+1), pxlcount*(idxY+1)))

	# timeSignal(data,screen,(0,255,0))
	#toc()
def peakDiamonds(data,screen):
	global peak, peakArray
	#tic()

	screen.fill((0,0,0))
	lastsample = 0;
	
	peak = np.exp(max(data)/30)
	# if max(downsampled) > peak:
	# 	peak = max(downsampled) * 3
	# else:
	# 	peak = peak * 0.25

	peakArray.pop(0)
	peakArray.append(peak)
	
	#resultMatrix = np.array([peakArray.copy()[4:peakArraySize],peakArray.copy()[3:peakArraySize-1],peakArray.copy()[2:peakArraySize-2],peakArray.copy()[1:peakArraySize-3],peakArray.copy()[0:peakArraySize-4]])
	arrayRows = peakDiamondSize
	bufferCols = peakArraySize
	bufferArray = peakArray
	resultMatrix = np.array([bufferArray.copy()[arrayRows-1:bufferCols]])
	
	for offs in range(1,arrayRows):
		resultMatrix = np.append(resultMatrix,[bufferArray.copy()[arrayRows-1-offs:bufferCols-offs]], axis = 0)

	pxlcount = 48
	surf = pygame.surfarray.make_surface(resultMatrix)
	surfa = pygame.transform.scale(surf,(pxlcount,pxlcount))
	surfb = pygame.transform.rotate(surfa,90)
	surfc = pygame.transform.rotate(surfa,180)
	surfd = pygame.transform.rotate(surfa,270)
	for idxX in range(0,16,2):
		for idxY in range(0,16,2):
			screen.blit(surfb, (pxlcount*idxX, pxlcount*idxY))
			screen.blit(surfa, (pxlcount*(idxX+1), pxlcount*idxY))
			screen.blit(surfc, (pxlcount*idxX, pxlcount*(idxY+1)))
			screen.blit(surfd, (pxlcount*(idxX+1), pxlcount*(idxY+1)))

	timeSignal(data,screen,(0,255,0))
	#toc()
def timeSignal(data, screen, color = (0, 255, 255)):
	downsampled = signal.resample(data, WIDTH)
	downsampled = downsampled/20
	count = 0;
	lastsample = 0;
	for sample in downsampled:
		pygame.draw.rect(screen, color, pygame.Rect(count, HEIGHT/2-lastsample, 1, lastsample-sample))
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
displayFunctions = [meanDiamonds,timeSignal,fftSignal,peakDiamonds]
nextChangeTime = time.time()

# define a main function
def main():
	# initialize the pygame module
	pygame.init()
	# load and set the logo
	logo = pygame.image.load("logo32x32.png")
	pygame.display.set_icon(logo)
	pygame.display.set_caption("RaspberryVideoMusic")

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
		screen.fill((0,0,0))
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