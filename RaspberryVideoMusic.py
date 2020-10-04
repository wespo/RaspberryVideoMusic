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


#old, weird variables....
S_t = 0
peak = 0

FRAMEREADY = pygame.USEREVENT+1

tStart = time.time()
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

def display_none(self,data,screen):
	return
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
def peakDecay(val, oldVal, tau, CHUNK, RATE):
	val = np.abs(val)
	if val > oldVal:
		return val
	else:
		alpha = 1 - np.exp(-(CHUNK/RATE)/tau)
		val = alpha * val + (1 - alpha) * oldVal
		return val
class piVideoMusic:
	def __init__(self,displayFunctions=None,backgroundFunctions=None):
		self.CHUNK=2205
		self.RATE=44100
		self.WIDTH=640
		self.HEIGHT=480
		self.time_to_buf_fft = 4 #sec
		self.signal_pk = 0

		self.tau = 0.05 #set tau = 0.2 -> 5*tau = 1sec
		self.numBars = 12

		#AGC gain and peak detect
		self.max_gain = 1
		self.min_gain = 0.001
		self.digitalGain = self.max_gain
		self.max_gain_holdoff = 0
		self.max_gain_holdoff_limit = 100
		self.min_gain_holdoff = 0
		self.min_gain_holdoff_limit = 5
		self.count_target = 1000
		self.sampleBuffer = np.zeros(self.RATE*4)

		self.fftArray = np.zeros([self.CHUNK,int(self.time_to_buf_fft*self.RATE/self.CHUNK)])

		self.persistantDisplayData={}
		self.currentDisplayData={}

		if displayFunctions == None:
			self.displayFunctions = [display_none]
		else:
			self.displayFunctions = displayFunctions
		if backgroundFunctions == None:
			self.backgroundFunctions = [display_none]
		else:
			self.backgroundFunctions = backgroundFunctions

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
		self.screen = pygame.display.set_mode(size, flags)
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
		pygame.draw.rect(self.screen, (0, 255, 255), pygame.Rect(10, 10, 20, 20))
		pygame.mouse.set_visible(False)
		pygame.display.update()
		# define a variable to control the main loop
		self.running = True
		#audio handle
		p=pyaudio.PyAudio()
		
		#input stream setup
		self.stream=p.open(format = pyaudio.paInt16,rate=self.RATE,channels=1, input_device_index = 0, input=True, frames_per_buffer=self.CHUNK, stream_callback=self.processBuffer) #on Win7 PC, 1 = Microphone, 2 = stereo mix (enabled in sound control panel)

		#display state initialization:
		self.nextChangeTime = time.time()
		self.currentDisplay = self.displayFunctions[0]
		self.currentBackground = self.backgroundFunctions[0]

	def processEvent(self):
		for event in pygame.event.get():
			# only do something if the event is of type QUIT
			if event.type == FRAMEREADY:
				self.updateData(self.screen, self.sampleBuffer, self.signal_pk)
				self.nextChangeTime, self.currentDisplay, self.currentBackground, self.currentDisplayData = self.updateFrame(self.screen, self.sampleBuffer, self.displayFunctions, self.backgroundFunctions, self.nextChangeTime, self.currentDisplay, self.currentDisplayData, self.currentBackground, self.signal_pk)
				pygame.display.update()
				pygame.event.clear(FRAMEREADY)
			if event.type == pygame.QUIT:
				# change the value to False, to exit the main loop
				self.running = False
	def fftwindow(self, data):
		window = signal.hamming(len(data))
		w_data = data * window
		fft_data = np.absolute(rfft(w_data))
		fft_data = 200 * np.log10(fft_data*self.digitalGain)
		return fft_data

	def agc(self,data,newPk): #AGC gain and peak detect
		if newPk*self.digitalGain < self.count_target*0.9:
			self.max_gain_holdoff += 1
			if self.max_gain_holdoff > self.max_gain_holdoff_limit:
				#increase gain
				self.digitalGain = self.digitalGain * 1.05
				#clamp
				self.digitalGain = min(self.max_gain, self.digitalGain)
		else:
			self.max_gain_holdoff = 0
		
		if newPk*self.digitalGain > self.count_target*1.1:
			self.min_gain_holdoff += 1
			if self.min_gain_holdoff > self.min_gain_holdoff_limit:
				#increase gain
				self.digitalGain = self.digitalGain * 0.9
				#clamp
				self.digitalGain = max(self.min_gain, self.digitalGain)
		else:
			self.min_gain_holdoff = 0

		return self.digitalGain
	
	# print(newPk*digitalGain,digitalGain)

	def updateData(self,screen,data,signal_pk): #handle processing functions that do not directly draw to the screen, like computing FFT, peaks, etc.
		#compute peak amplitude w/ decay
		newPk = np.amax(np.abs(data[-self.CHUNK:]))
		self.digitalGain = self.agc(data,newPk)
		self.signal_pk = peakDecay(newPk*self.digitalGain, self.signal_pk, self.tau, self.CHUNK, self.RATE)
		#compute fft
		newFrame1 = np.array(self.fftwindow(data[-self.CHUNK:]))
		newFrame2 = np.array(self.fftwindow(data[-int(self.CHUNK*3/2):-int(self.CHUNK/2)]))
		newFrame = np.column_stack((newFrame1,newFrame2))
		self.fftArray = shiftIn2DCols(self.fftArray, newFrame)
		return signal_pk

	def updateFrame(self,screen, data, displayFunctions, backgroundFunctions, nextChangeTime, currentDisplay, currentDisplayData, currentBackground, signal_pk): #update the screen
		currentTime = time.time();
		if(currentTime > nextChangeTime):
			currentDisplay = random.choice(displayFunctions)
			currentBackground = random.choice(backgroundFunctions)
			nextChangeTime = currentTime + random.randrange(3,7)
			currentDisplayData={}
				
		screen.fill((0,0,0))
		currentBackground(self,data[-5000:], screen)
		currentDisplay(self,data[-5000:], screen)
		return nextChangeTime,currentDisplay, currentBackground, currentDisplayData

	def processBuffer(self,in_data, frame_count, time_info, status):
		#data=np.fromstring(stream.read(CHUNK,exception_on_overflow = False),dtype=np.int16)
		data = np.frombuffer(in_data, dtype=np.int16)
		raw_data = data
		self.sampleBuffer = shiftIn(self.sampleBuffer,data)
		
		my_event = pygame.event.Event(FRAMEREADY, message="Bad cat!")
		pygame.event.post(my_event)

		return (raw_data, pyaudio.paContinue)

#### Signal display functions

def meanDiamonds(self,data,screen):
	if not 'meanArray' in self.persistantDisplayData:
		self.persistantDisplayData['meanDiamondSize'] = 64 #was 16 before
		self.persistantDisplayData['meanArraySize'] = (2*self.persistantDisplayData['meanDiamondSize']-1)
		self.persistantDisplayData['meanArray'] = [0] * self.persistantDisplayData['meanArraySize'] #np.zeros((1,2*peakDiamondSize-1))

	self.persistantDisplayData['meanArray'].pop(0)
	self.persistantDisplayData['meanArray'].append(self.signal_pk)
	
	resultMatrix = []
	for offs in range(0,self.persistantDisplayData['meanDiamondSize']):
		resultMatrix.append(self.persistantDisplayData['meanArray'][self.persistantDisplayData['meanArraySize']-self.persistantDisplayData['meanDiamondSize']-offs:self.persistantDisplayData['meanArraySize']-offs])
	
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
def meanFreq(self,data,screen): #needs tuning
	global peak, peakArray
	# tic()
	
	if not 'freqArrayr' in self.persistantDisplayData:
		self.persistantDisplayData['freqDiamondSize'] = 40 #was 16 before
		self.persistantDisplayData['freqArraySize'] = (2*self.persistantDisplayData['freqDiamondSize']-1)
		self.persistantDisplayData['freqArrayr'] = [0] * self.persistantDisplayData['freqArraySize'] #np.zeros((1,2*peakDiamondSize-1))
		self.persistantDisplayData['freqArrayg'] = [0] * self.persistantDisplayData['freqArraySize'] #np.zeros((1,2*peakDiamondSize-1))
		self.persistantDisplayData['freqArrayb'] = [0] * self.persistantDisplayData['freqArraySize'] #np.zeros((1,2*peakDiamondSize-1))

	screen.fill((0,0,0))
	
	fft_data = np.absolute(rfft(data))/100
	p1 = round(len(fft_data)/50);
	p2 = round(len(fft_data)/30);
	p3 = round(len(fft_data)/4);
	peakr = clamp(pow(2.71,1+max(abs(fft_data[0:p1]))/80),0,64)
	peakg = clamp(pow(2.71,1+max(abs(fft_data[p1:p2]))/40),0,128)
	peakb = clamp(pow(2.71,1+max(abs(fft_data[p2:p3]))/20),0,255)
		# print(peak)
	

	self.persistantDisplayData['freqArrayr'].pop(0)
	self.persistantDisplayData['freqArrayr'].append(peakr)
	self.persistantDisplayData['freqArrayg'].pop(0)
	self.persistantDisplayData['freqArrayg'].append(peakg)
	self.persistantDisplayData['freqArrayb'].pop(0)
	self.persistantDisplayData['freqArrayb'].append(peakb)

	resultMatrixR = []
	resultMatrixG = []
	resultMatrixB = []
	for offs in range(0,self.persistantDisplayData['freqDiamondSize']):
		resultMatrixR.append(self.persistantDisplayData['freqArrayr'][self.persistantDisplayData['freqArraySize']-self.persistantDisplayData['freqDiamondSize']-offs:self.persistantDisplayData['freqArraySize']-offs])
		resultMatrixG.append(self.persistantDisplayData['freqArrayg'][self.persistantDisplayData['freqArraySize']-self.persistantDisplayData['freqDiamondSize']-offs:self.persistantDisplayData['freqArraySize']-offs])
		resultMatrixB.append(self.persistantDisplayData['freqArrayb'][self.persistantDisplayData['freqArraySize']-self.persistantDisplayData['freqDiamondSize']-offs:self.persistantDisplayData['freqArraySize']-offs])
	
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
def peakDiamonds(self, data,screen):
	if not 'peakArray' in self.persistantDisplayData:
		self.persistantDisplayData['peakDiamondSize'] = 10
		self.persistantDisplayData['peakArraySize'] = (2*self.persistantDisplayData['peakDiamondSize']-1)
		self.persistantDisplayData['peakArray'] = [0] * self.persistantDisplayData['peakArraySize'] #np.zeros((1,2*peakDiamondSize-1))
	
	self.persistantDisplayData['peakArray'].pop(0)
	self.persistantDisplayData['peakArray'].append(self.signal_pk/10)
	
	#resultMatrix = np.array([peakArray.copy()[4:peakArraySize],peakArray.copy()[3:peakArraySize-1],peakArray.copy()[2:peakArraySize-2],peakArray.copy()[1:peakArraySize-3],peakArray.copy()[0:peakArraySize-4]])
	arrayRows = self.persistantDisplayData['peakDiamondSize']
	bufferCols = self.persistantDisplayData['peakArraySize']
	bufferArray = self.persistantDisplayData['peakArray']
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
gl_count = 0
def background(self, data, screen):
	colorBG = np.clip(self.signal_pk/1200,0,1)
	colorBG = np.clip((1-colorBG)*1.5,0,1)
	colorBG = colorsys.hsv_to_rgb(0, 0.5*(1-colorBG), 0.77)
	colorBG = tuple(round(i * 255) for i in colorBG)
	screen.fill(colorBG)

def peakLine(self, data, screen):
	pygame.draw.line(screen, (255,0,0), (0,self.HEIGHT/2-self.signal_pk), (self.WIDTH,self.HEIGHT/2-signal_pk), 3)

def timeSignal(self, data, screen, color = (0, 255, 255)):
	downsampled = signal.resample(data, self.WIDTH)
	downsampled = downsampled * self.digitalGain / 4

	sampledTuple =[]
	xoff = 0 
	xstep = self.WIDTH/len(downsampled)
	yoff = self.HEIGHT/2
	for sample in downsampled:
		sampledTuple.append((round(xoff),round(yoff-sample)))
		xoff = xoff + xstep

	pygame.draw.lines(screen,color,False,np.array(sampledTuple).astype(np.int64),5)



def spectrogram(self,data,screen):
	specArray = signal.resample (self.fftArray, 64, axis = 0)
	blitArray = np.zeros((specArray.shape[0],specArray.shape[1],3))
	blitArray[:,:,0] = np.clip(specArray/5,0,255)
	surf = pygame.surfarray.make_surface(np.transpose(blitArray,(1,0,2)))
	surfa = pygame.transform.scale(surf,(640,480))
	# surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfa, (0,0))


def barchart(self, data, screen):
	# background(data,screen)
	if not 'minfft' in self.currentDisplayData:
		self.currentDisplayData['minfft'] = np.zeros(self.numBars)-1
		self.persistantDisplayData['fftBargraphPeaks'] = np.zeros(self.numBars)

	downsampled = signal.resample(self.fftArray[:,-1], self.numBars)
	downsampled = downsampled * 0.75
	#compute peaks
	for channel in range(len(downsampled)):
		if self.currentDisplayData['minfft'][channel] == -1:
			self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
		if self.currentDisplayData['minfft'][channel] > downsampled[channel]:
			self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
		downsampled[channel] = np.abs(downsampled[channel] - self.currentDisplayData['minfft'][channel])
		self.persistantDisplayData['fftBargraphPeaks'][channel] = peakDecay(downsampled[channel], self.persistantDisplayData['fftBargraphPeaks'][channel], self.tau*5, self.CHUNK, self.RATE)
	#draw bars
	barWidth = int(self.WIDTH/(self.numBars+4))
	for xIdx in range(self.numBars):
		xPos = (xIdx+1) / (self.numBars+1) * self.WIDTH
		pygame.draw.line(screen, (255,0,255), (xPos,self.HEIGHT), (xPos,self.HEIGHT-downsampled[xIdx]), barWidth)

		pygame.draw.line(screen, (0,128,128), (xPos-barWidth/2,self.HEIGHT-self.persistantDisplayData['fftBargraphPeaks'][xIdx]), (xPos+barWidth/2,self.HEIGHT-self.persistantDisplayData['fftBargraphPeaks'][xIdx]), 2)
	return

def fftSignal(self, data, screen, color = (0, 255, 255)):
	downsampled = signal.resample(self.fftArray[:,-1], self.WIDTH)
	downsampled = downsampled/4
	
	sampledTuple =[]
	xoff = 0 
	xstep = self.WIDTH/len(downsampled)
	yoff = round(self.HEIGHT * 0.9)

	for sample in downsampled:
		sampledTuple.append((round(xoff),round(yoff-sample)))
		xoff = xoff + xstep

	pygame.draw.lines(self.screen,color,False,np.array(sampledTuple).astype(np.int64),5)
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)

if __name__=="__main__":
    # call the main function
    # main([timeSignal,meanFreq,meanDiamonds,timeBackground,fftBackground,peakDiamonds,barchart,spectrogram])
    PVM = piVideoMusic([timeSignal, fftSignal, barchart],[spectrogram,background,peakDiamonds,meanDiamonds]) #
    while PVM.running:
    	PVM.processEvent()