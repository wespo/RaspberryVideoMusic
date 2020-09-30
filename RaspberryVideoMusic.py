#audioGameTest
# import the pygame module, so you can use it
import pygame
import pyaudio
import numpy as np
from scipy import signal
from scipy.fftpack import rfft, fftshift
import time
import random
import colorsys
import os

CHUNK=2205
RATE=44100
WIDTH=640
HEIGHT=480
time_to_buf_fft = 4 #sec
signal_pk = 0

tau = 0.05 #set tau = 0.2 -> 5*tau = 1sec
numBars = 12

#AGC gain and peak detect
max_gain = 1
min_gain = 0.001
digitalGain = max_gain
max_gain_holdoff = 0
max_gain_holdoff_limit = 100
min_gain_holdoff = 0
min_gain_holdoff_limit = 5
count_target = 1000

#old, weird variables....
S_t = 0
peak = 0


sampleBuffer = np.zeros(RATE*4)
fftArray = np.zeros([CHUNK,int(time_to_buf_fft*RATE/CHUNK)])
fftBargraphPeaks = np.zeros(numBars)
minfft = np.zeros(numBars)-1
FRAMEREADY = pygame.USEREVENT+1

peakDiamondSize = 10
peakArraySize = (2*peakDiamondSize-1)
peakArray = [0] * peakArraySize #np.zeros((1,2*peakDiamondSize-1))

meanDiamondSize = 64 #was 16 before
meanArraySize = (2*meanDiamondSize-1)
meanArray = [0] * meanArraySize #np.zeros((1,2*peakDiamondSize-1))

freqDiamondSize = 40 #was 16 before
freqArraySize = (2*freqDiamondSize-1)
freqArrayr = [0] * freqArraySize #np.zeros((1,2*peakDiamondSize-1))
freqArrayg = [0] * freqArraySize #np.zeros((1,2*peakDiamondSize-1))
freqArrayb = [0] * freqArraySize #np.zeros((1,2*peakDiamondSize-1))

tStart = time.time()
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

#### Non-signal support functions
def tic():
	global tStart
	tic = time.time()
	print("Interval:" + str(round(1000*(tic-tStart))), end='\t')
	tStart = tic
def toc():
	global tStart
	print("Frame Time:" + str(round(1000*(time.time()-tStart))))


#### Signal Support functions
# shift arr2 into the end of arr1, dropping len(arr2) samples at the start of arr1
def shiftIn(arr1, arr2):
    l1 = len(arr1)
    l2 = len(arr2)
    ld = l1-l2
    result = np.zeros(l1)
    result[:ld] = arr1[l2:]
    result[ld:] = arr2
    return result

def shiftIn2DCols(arr1, arr2):
	l1 = arr1.shape[1]
	l2 = arr2.shape[1]
	ld = l1-l2
	result = np.zeros(arr1.shape)
	result[:,:ld] = arr1[:,l2:]
	result[:,ld:] = arr2
	return result

#### Signal display functions

def meanDiamonds(data,screen):
	global signal_pk, peakArray

	meanArray.pop(0)
	meanArray.append(signal_pk)
	
	resultMatrix = []
	for offs in range(0,meanDiamondSize):
		resultMatrix.append(meanArray[meanArraySize-meanDiamondSize-offs:meanArraySize-offs])
	
	resArray = np.asarray(resultMatrix)
	zerArray = np.zeros(resArray.shape+(3,))
	zerArray[:,:,2] = resArray

	surf = pygame.surfarray.make_surface(zerArray)
	surfa = pygame.transform.scale(surf,(320,240))
	surfb = pygame.transform.flip(surfa,True, False)
	surfc = pygame.transform.flip(surfa, False ,True)
	surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfb, (0,0))
	screen.blit(surfa, (320,0))
	screen.blit(surfd, (0,240))
	screen.blit(surfc, (320,240))

	#timeSignal(data,screen,(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
	# timeSignal(data,screen,(0,255,0))
	# toc()
def meanFreq(data,screen):
	global peak, peakArray
	# tic()
	
	screen.fill((0,0,0))
	
	fft_data = np.absolute(rfft(data))/100
	p1 = round(len(fft_data)/50);
	p2 = round(len(fft_data)/30);
	p3 = round(len(fft_data)/4);
	peakr = clamp(pow(2.71,1+max(abs(fft_data[0:p1]))/80),0,64)
	peakg = clamp(pow(2.71,1+max(abs(fft_data[p1:p2]))/40),0,128)
	peakb = clamp(pow(2.71,1+max(abs(fft_data[p2:p3]))/20),0,255)
		# print(peak)
	

	freqArrayr.pop(0)
	freqArrayr.append(peakr)
	freqArrayg.pop(0)
	freqArrayg.append(peakg)
	freqArrayb.pop(0)
	freqArrayb.append(peakb)

	resultMatrixR = []
	resultMatrixG = []
	resultMatrixB = []
	for offs in range(0,freqDiamondSize):
		resultMatrixR.append(freqArrayr[freqArraySize-freqDiamondSize-offs:freqArraySize-offs])
		resultMatrixG.append(freqArrayg[freqArraySize-freqDiamondSize-offs:freqArraySize-offs])
		resultMatrixB.append(freqArrayb[freqArraySize-freqDiamondSize-offs:freqArraySize-offs])
	
	resArray = np.asarray(resultMatrixR)
	zerArray = np.zeros(resArray.shape+(3,))
	
	zerArray[:,:,0] = resArray
	resArray = np.asarray(resultMatrixG)
	zerArray[:,:,1] = resArray
	resArray = np.asarray(resultMatrixB)
	zerArray[:,:,2] = resArray

	surf = pygame.surfarray.make_surface(zerArray)
	surfa = pygame.transform.scale(surf,(320,240))
	surfb = pygame.transform.flip(surfa,True, False)
	surfc = pygame.transform.flip(surfa, False ,True)
	surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfb, (0,0))
	screen.blit(surfa, (320,0))
	screen.blit(surfd, (0,240))
	screen.blit(surfc, (320,240))

	# fftSignal(data,screen,(0,255,0))
	# toc()
def peakDiamonds(data,screen):
	global signal_pk, peakArray

	peakArray.pop(0)
	peakArray.append(signal_pk/10)
	
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
	surfb = pygame.transform.flip(surfa,True, False)
	surfc = pygame.transform.flip(surfa, False ,True)
	surfd = pygame.transform.flip(surfa,True, True)
	
	for idxX in range(0,16,2):
		for idxY in range(0,16,2):
			screen.blit(surfa, (pxlcount*idxX, pxlcount*idxY))
			screen.blit(surfb, (pxlcount*(idxX+1), pxlcount*idxY))
			screen.blit(surfc, (pxlcount*idxX, pxlcount*(idxY+1)))
			screen.blit(surfd, (pxlcount*(idxX+1), pxlcount*(idxY+1)))

	# timeSignal(data,screen,(0,255,0))
	#toc()
def background(data,screen):
	global signal_pk,digitalGain
	colorBG = np.clip(signal_pk/1200,0,1)
	colorBG = np.clip((1-colorBG)*1.5,0,1)
	colorBG = colorsys.hsv_to_rgb(0, 0.5*(1-colorBG), 0.77)
	colorBG = tuple(round(i * 255) for i in colorBG)
	screen.fill(colorBG)

def peakLine(screen):
	global signal_pk
	pygame.draw.line(screen, (255,0,0), (0,HEIGHT/2-signal_pk), (WIDTH,HEIGHT/2-signal_pk), 3)
def timeBackground(data, screen, color = (0, 255, 255)):
	background(data,screen)
	timeSignal(data, screen, color)
	
def timeBackgroundPeak(data,screen, color = (0,255,255)):
	timeBackground(data, screen, color)
	peakLine(screen)
def timeSignal(data, screen, color = (0, 255, 255)):
	global digitalGain
	downsampled = signal.resample(data, WIDTH)
	downsampled = downsampled * digitalGain / 4

	sampledTuple =[]
	xoff = 0 
	xstep = WIDTH/len(downsampled)
	yoff = HEIGHT/2
	for sample in downsampled:
		sampledTuple.append((round(xoff),round(yoff-sample)))
		xoff = xoff + xstep

	pygame.draw.lines(screen,color,False,np.array(sampledTuple).astype(np.int64),5)

def fftBackground(data, screen, color = (0, 255, 255)):
	background(data,screen)
	fftSignal(data, screen, color)
	
def array_reshape(input_vector,result_height,result_width,row_stride,row_repeat):
	resArray = np.ones([result_height,result_width])
	input_offset = result_height
	for idx in range(0,resArray.shape[1], row_repeat):
		start_idx = input_offset
		end_idx = input_offset+result_height
		if end_idx == 0:
			end_idx = None
		input_offset = input_offset + row_stride
		col_data = input_vector[start_idx:end_idx]
		# col_data = signal.resample(200*np.log10(rfft(signal.resample(col_data, int(HEIGHT/10)))),HEIGHT)
		col_data = np.clip(40*np.log10(rfft(col_data)),0,200)+50
		for idx2 in range(0,row_repeat):
			resArray[:,idx+idx2] = col_data
	return resArray

def spectrogram(data,screen):
	global WIDTH, HEIGHT, fftArray
	specArray = signal.resample (fftArray, 64, axis = 0)
	blitArray = np.zeros((specArray.shape[0],specArray.shape[1],3))
	blitArray[:,:,0] = np.clip(specArray/5,0,255)
	surf = pygame.surfarray.make_surface(np.transpose(blitArray,(1,0,2)))
	surfa = pygame.transform.scale(surf,(640,480))
	# surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfa, (0,0))


def barchart(data, screen):
	global numBars, tau, HEIGHT, WIDTH
	# background(data,screen)
	downsampled = signal.resample(fftArray[:,-1], numBars)
	downsampled = downsampled * 0.75
	#compute peaks
	for channel in range(len(downsampled)):
		if minfft[channel] == -1:
			minfft[channel] = np.abs(downsampled[channel])
		if minfft[channel] > downsampled[channel]:
			minfft[channel] = np.abs(downsampled[channel])
		downsampled[channel] = np.abs(downsampled[channel] - minfft[channel])
		fftBargraphPeaks[channel] = peakDecay(downsampled[channel], fftBargraphPeaks[channel], tau*5)
	#draw bars
	barWidth = int(WIDTH/(numBars+4))
	for xIdx in range(numBars):
		xPos = (xIdx+1) / (numBars+1) * WIDTH
		pygame.draw.line(screen, (255,0,255), (xPos,HEIGHT), (xPos,HEIGHT-downsampled[xIdx]), barWidth)

		pygame.draw.line(screen, (0,128,128), (xPos-barWidth/2,HEIGHT-fftBargraphPeaks[xIdx]), (xPos+barWidth/2,HEIGHT-fftBargraphPeaks[xIdx]), 2)
	return

def fftSignal(data, screen, color = (0, 255, 255)):
	global fftArray
	
	downsampled = signal.resample(fftArray[:,-1], WIDTH)
	downsampled = downsampled/4
	
	sampledTuple =[]
	xoff = 0 
	xstep = WIDTH/len(downsampled)
	yoff = round(HEIGHT * 0.9)

	for sample in downsampled:
		sampledTuple.append((round(xoff),round(yoff-sample)))
		xoff = xoff + xstep

	pygame.draw.lines(screen,color,False,np.array(sampledTuple).astype(np.int64),5)

def fftwindow(data):
	global digitalGain, CHUNK
	window = signal.hamming(len(data))
	w_data = data * window
	fft_data = np.absolute(rfft(w_data))
	fft_data = 200 * np.log10(fft_data*digitalGain)
	return fft_data
#### System Functions
def peakDecay(val, oldVal, tau):
	val = np.abs(val)
	if val > oldVal:
		return val
	else:
		alpha = 1 - np.exp(-(CHUNK/RATE)/tau)
		val = alpha * val + (1 - alpha) * oldVal
		return val
def agc(data,newPk): #AGC gain and peak detect
	global max_gain, min_gain, digitalGain, max_gain_holdoff, min_gain_holdoff, max_gain_holdoff_limit, min_gain_holdoff_limit, count_target

	if newPk*digitalGain < count_target*0.9:
		max_gain_holdoff += 1
		if max_gain_holdoff > max_gain_holdoff_limit:
			#increase gain
			digitalGain = digitalGain * 1.05
			#clamp
			digitalGain = min(max_gain, digitalGain)
	else:
		max_gain_holdoff = 0
	
	if newPk*digitalGain > count_target*1.1:
		min_gain_holdoff += 1
		if min_gain_holdoff > min_gain_holdoff_limit:
			#increase gain
			digitalGain = digitalGain * 0.9
			#clamp
			digitalGain = max(min_gain, digitalGain)
	else:
		min_gain_holdoff = 0

	return digitalGain
	
	# print(newPk*digitalGain,digitalGain)

def updateData(screen,data,signal_pk): #handle processing functions that do not directly draw to the screen, like computing FFT, peaks, etc.
	global CHUNK, tau, fftArray
	#compute peak amplitude w/ decay
	newPk = np.amax(np.abs(data[-CHUNK:]))
	digitalGain = agc(data,newPk)
	signal_pk = peakDecay(newPk*digitalGain, signal_pk, tau)
	#compute fft

	newFrame1 = np.array(fftwindow(data[-CHUNK:]))
	newFrame2 = np.array(fftwindow(data[-int(CHUNK*3/2):-int(CHUNK/2)]))
	newFrame = np.column_stack((newFrame1,newFrame2))
	fftArray = shiftIn2DCols(fftArray, newFrame)
	return signal_pk

def updateFrame(screen, data, displayFunctions, backgroundFunctions, nextChangeTime, currentDisplay, currentBackground, signal_pk): #update the screen
	global WIDTH, HEIGHT
	currentTime = time.time();
	if(currentTime > nextChangeTime):
		currentDisplay = random.choice(displayFunctions)
		currentBackground = random.choice(backgroundFunctions)
		nextChangeTime = currentTime + random.randrange(3,7)
			
	screen.fill((0,0,0))
	currentBackground(data[-5000:], screen)
	currentDisplay(data[-5000:], screen)
	return nextChangeTime,currentDisplay, currentBackground

def processBuffer(in_data, frame_count, time_info, status):
	global sampleBuffer
	#data=np.fromstring(stream.read(CHUNK,exception_on_overflow = False),dtype=np.int16)
	data = np.frombuffer(in_data, dtype=np.int16)
	raw_data = data
	sampleBuffer = shiftIn(sampleBuffer,data)
	
	my_event = pygame.event.Event(FRAMEREADY, message="Bad cat!")
	pygame.event.post(my_event)

	return (raw_data, pyaudio.paContinue)

# define a main function
def main(displayFunctions,backgroundFunctions):
	global signal_pk, CHUNK, RATE, WIDTH, HEIGHT
	# initialize the pygame module
	disp_no = os.getenv("DISPLAY")

	# Check which frame buffer drivers are available
	# Start with fbcon since directfb hangs with composite output
	drivers = ['fbcon', 'directfb', 'svgalib']
	found = False
	for driver in drivers:
		# Make sure that SDL_VIDEODRIVER is set
		if not os.getenv('SDL_VIDEODRIVER'):
			os.putenv('SDL_VIDEODRIVER', driver)
		try:
			pygame.display.init()
		except pygame.error:
			print('Driver: {0} failed.'.format(driver))
			continue
		found = True
		break

	if not found:
	    raise Exception('No suitable video driver found!')
	size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
	flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN
	screen = pygame.display.set_mode(size, flags)
	surface = pygame.Surface((640,480),pygame.FULLSCREEN)
	pygame.display.set_mode((640,480),pygame.FULLSCREEN)
	pySurface = surface.convert()
	# Initialise font support
	pygame.font.init()
	pygame.init()
	# load and set the logo
	# logo = pygame.image.load("logo32x32.png")
	# pygame.display.set_icon(logo)
	pygame.display.set_caption("RaspberryVideoMusic")

	# create a surface on screen that has the size of 640 x 480
	# screen = pygame.display.set_mode((WIDTH,HEIGHT),pygame.FULLSCREEN)
	pygame.draw.rect(screen, (0, 255, 255), pygame.Rect(10, 10, 20, 20))
	pygame.mouse.set_visible(False)
	pygame.display.update()
	# define a variable to control the main loop
	running = True
	#audio handle
	p=pyaudio.PyAudio()
	
	#input stream setup
	stream=p.open(format = pyaudio.paInt16,rate=RATE,channels=1, input_device_index = 0, input=True, frames_per_buffer=CHUNK, stream_callback=processBuffer) #on Win7 PC, 1 = Microphone, 2 = stereo mix (enabled in sound control panel)

	#display state initialization:
	nextChangeTime = time.time()
	currentDisplay = displayFunctions[0]
	currentBackground = backgroundFunctions[0]


	# main loop
	while running:
		# event handling, gets all event from the event queue
		for event in pygame.event.get():
			# only do something if the event is of type QUIT
			if event.type == FRAMEREADY:
				signal_pk = updateData(screen, sampleBuffer, signal_pk)
				nextChangeTime, currentDisplay, currentBackground = updateFrame(screen, sampleBuffer, displayFunctions, backgroundFunctions, nextChangeTime, currentDisplay, currentBackground, signal_pk)
				pygame.display.update()
				pygame.event.clear(FRAMEREADY)
			if event.type == pygame.QUIT:
				# change the value to False, to exit the main loop
				running = False
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)

if __name__=="__main__":
    # call the main function
    # main([timeSignal,meanFreq,meanDiamonds,timeBackground,fftBackground,peakDiamonds,barchart,spectrogram])
    main([timeSignal,fftSignal,barchart],[peakDiamonds, meanDiamonds, spectrogram, background])