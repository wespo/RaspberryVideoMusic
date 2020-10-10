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
import argparse
import setproctitle

#old, weird variables....
S_t = 0
peak = 0

setproctitle.setproctitle("PiVideoMusic")

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

def ListDiff(list1, list2): 
    return (list(list(set(list1)-set(list2)) + list(set(list2)-set(list1)))) 
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
	def __init__(self,displayFunctions=None,backgroundFunctions=None,listFlag=False,videoInterface=0,audioInterface=0,fullscreen=True):
		self.CHUNK=2205
		self.RATE=44100
		self.WIDTH=640
		self.HEIGHT=480
		self.time_to_buf_fft = 4 #sec
		self.signal_pk = 0
		self.colorlist = range(0,256,32)
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
		self.currentDisplayData=self.refreshCurrentDisplay()
		

		if displayFunctions == None:
			self.displayFunctions = [display_none]
		else:
			self.displayFunctions = displayFunctions
		if backgroundFunctions == None:
			self.backgroundFunctions = [display_none]
		else:
			self.backgroundFunctions = backgroundFunctions

		if fullscreen == True:
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
		else:
			self.screen = pygame.display.set_mode((self.WIDTH,self.HEIGHT))
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
		
		audioInterfaceString = "{:2}\t{:40}\t{:8}\t{:9}\t{:11}\t{:3}"
		if listFlag:
			print(audioInterfaceString.format("ID","Name","Input Ch", "Output Ch", "Sample Rate", "Sel"))
			for i in range(p.get_device_count()):
				if i == audioInterface:
					marker = "<--"
				else:
					marker = ""
				interfaceInfo = p.get_device_info_by_index(i)
				print(audioInterfaceString.format(interfaceInfo['index'],interfaceInfo['name'],interfaceInfo['maxInputChannels'],interfaceInfo['maxOutputChannels'],interfaceInfo['defaultSampleRate'],marker))
		#input stream setup
		self.stream=p.open(format = pyaudio.paInt16,rate=self.RATE,channels=1, input_device_index = audioInterface, input=True, frames_per_buffer=self.CHUNK, stream_callback=self.processBuffer) #on Win7 PC, 1 = Microphone, 2 = stereo mix (enabled in sound control panel)

		#display state initialization:
		self.nextChangeTime = time.time()
		self.currentDisplay = self.displayFunctions[0]
		self.currentBackground = self.backgroundFunctions[0]
	def generateColorTuples(self):
		fg1 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))
		fg2 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))
		bg1 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))
		bg2 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))
		
		if len([fg1,fg2,bg1,bg2]) == len(set([fg1,fg2,bg1,bg2])):
			fg1 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))
			fg2 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))
			bg1 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))
			bg2 = (random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist))

		if bg1 == (0,0,0):
			bg1 = list(bg1)
			bg1[random.choice([0,1,2])] = random.choice(self.colorlist[1:])
			bg1 = tuple(bg1)
		if bg2 == (0,0,0):
			bg2 = list(bg2)
			bg2[random.choice([0,1,2])] = random.choice(self.colorlist[1:])
			bg2 = tuple(bg2)


		# print(colorsys.rgb_to_hsv(fg1[0]/255,fg1[1]/255,fg1[2]/255),colorsys.rgb_to_hsv(fg2[0]/255,fg2[1]/255,fg2[2]/255),colorsys.rgb_to_hsv(bg1[0]/255,bg1[1]/255,bg1[2]/255),colorsys.rgb_to_hsv(bg2[0]/255,bg2[1]/255,bg2[2]/255))

		#ensure that 

		return fg1, fg2, bg1, bg2
	def refreshCurrentDisplay(self):
		newData = {}
		newData['FGcolorTuple'], newData['FGcolorTuple2'], newData['BGcolorTuple1'], newData['BGcolorTuple2'] = self.generateColorTuples()
		# self.currentDisplayData = newData
		return newData
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
			newFG = ListDiff(displayFunctions,[currentDisplay])
			newBG = ListDiff(backgroundFunctions,[currentBackground])
			if newFG:
				currentDisplay = random.choice(newFG) #don't hop to the same display
			if newBG:
				currentBackground = random.choice(newBG) #don't hop to the same display
			nextChangeTime = currentTime + random.randrange(3,7)
			currentDisplayData=self.refreshCurrentDisplay()
				
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
	tempColorTuple = (self.currentDisplayData['BGcolorTuple1'][0]/255,self.currentDisplayData['BGcolorTuple1'][1]/255,self.currentDisplayData['BGcolorTuple1'][2]/255)
	zerArray[:,:,0] = resArray*tempColorTuple[0]
	zerArray[:,:,1] = resArray*tempColorTuple[1]
	zerArray[:,:,2] = resArray*tempColorTuple[2]

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
	if not 'backgroundFunctionColor' in self.currentDisplayData:
		self.currentDisplayData['backgroundFunctionColor'] = colorsys.rgb_to_hsv(self.currentDisplayData['BGcolorTuple1'][0]/255,self.currentDisplayData['BGcolorTuple1'][1]/255,self.currentDisplayData['BGcolorTuple1'][2]/255)[0]
	colorBG = np.clip(self.signal_pk/1600,0,1)
	colorBG = np.clip((1-colorBG)*1.5,0,1)
	colorBG = colorsys.hsv_to_rgb(self.currentDisplayData['backgroundFunctionColor'], 1, 0.6*(1-colorBG)+0.4)
	colorBG = tuple(round(i * 255) for i in colorBG)
	screen.fill(colorBG)

def peakLine(self, data, screen):
	pygame.draw.line(screen, (255,0,0), (0,self.HEIGHT/2-self.signal_pk), (self.WIDTH,self.HEIGHT/2-signal_pk), 3)

def timeSignal(self, data, screen, color = None):
	if color == None:
		color = self.currentDisplayData['FGcolorTuple']
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
	specRes = np.clip(specArray/5,0,255)
	blitArray[:,:,0] = specRes * self.currentDisplayData['BGcolorTuple1'][0]/255
	blitArray[:,:,1] = specRes * self.currentDisplayData['BGcolorTuple1'][1]/255
	blitArray[:,:,2] = specRes * self.currentDisplayData['BGcolorTuple1'][2]/255

	surf = pygame.surfarray.make_surface(np.transpose(blitArray,(1,0,2)))
	surfa = pygame.transform.scale(surf,(640,480))
	# surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfa, (0,0))


def barchart(self, data, screen):
	# background(data,screen)
	if not 'minfft' in self.currentDisplayData:
		self.currentDisplayData['minfft'] = np.zeros(self.numBars)-1
		self.currentDisplayData['fftBargraphPeaks'] = np.zeros(self.numBars)

	downsampled = signal.resample(self.fftArray[:,-1], self.numBars)
	downsampled = downsampled * 0.75
	#compute peaks
	for channel in range(len(downsampled)):
		if self.currentDisplayData['minfft'][channel] == -1: #check if minfft is initialized.
			self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
		if self.currentDisplayData['minfft'][channel] > downsampled[channel]: #if the new sample is less than the minimum, update the minimum.
			self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
		downsampled[channel] = np.abs(downsampled[channel] - self.currentDisplayData['minfft'][channel])
		self.currentDisplayData['fftBargraphPeaks'][channel] = peakDecay(downsampled[channel], self.currentDisplayData['fftBargraphPeaks'][channel], self.tau*5, self.CHUNK, self.RATE)
	#draw bars
	barWidth = int(self.WIDTH/(self.numBars+4))
	for xIdx in range(self.numBars):
		xPos = int((xIdx+1) / (self.numBars+1) * self.WIDTH)
		pygame.draw.line(screen, self.currentDisplayData['FGcolorTuple'], (xPos,self.HEIGHT), (int(xPos),int(self.HEIGHT-downsampled[xIdx])), barWidth)
		pygame.draw.line(screen, self.currentDisplayData['FGcolorTuple2'], (int(xPos-barWidth/2),int(self.HEIGHT-self.currentDisplayData['fftBargraphPeaks'][xIdx])), (int(xPos+barWidth/2),int(self.HEIGHT-self.currentDisplayData['fftBargraphPeaks'][xIdx])), 2)
	return

def fftline(frame,width,xoff=0,yoff=0,nsamp=None):
	if nsamp == None:
		nsamp = width
	downsampled = signal.resample(frame,nsamp)
	downsampled = downsampled/4	
	sampledTuple =[]
	xstep = width/len(downsampled)
	for sample in downsampled:
		sampledTuple.append((round(xoff),round(yoff-sample)))
		xoff = xoff + xstep
	return sampledTuple
def fftWaterfall(self, data, screen):
	color = self.currentDisplayData['FGcolorTuple']
	nrows = 15

	x_max = self.WIDTH/10
	y_max = self.HEIGHT*1.2
	x_stride = -x_max/nrows
	y_stride = y_max/nrows
	z_stride = 4

	x_start = x_max
	y_start = 0 #self.HEIGHT * 1.1
	z_start = 0


	for row in range(0,nrows-1):
		xoff = x_start + x_stride * row
		yoff = y_start + y_stride * row
		# yoff = y_start/2  np.abs(y_start/2 - y_stride * row)
		zoff = -(row+1)
		sampledTuple = fftline(np.array(self.fftArray[:,zoff])/2, self.WIDTH,xoff,yoff,100)
		pygame.draw.lines(self.screen,color,False,np.array(sampledTuple).astype(np.int64),5)

def bargraphHill(self, data, screen):
	if not 'minfft' in self.currentDisplayData:
		self.currentDisplayData['minfft'] = np.zeros(self.numBars)-1
		self.currentDisplayData['fftBargraphPeaks'] = np.zeros(self.numBars)

	color = self.currentDisplayData['FGcolorTuple']
	nrows = 15

	x_max = self.WIDTH/5
	y_max = self.HEIGHT
	x_stride = x_max/nrows
	y_stride = y_max/nrows
	z_stride = 4

	x_start = x_max
	# y_start = c
	z_start = 0
	
	if not 'bargrahpHill_a' in self.persistantDisplayData:
		c1 = (0,self.HEIGHT/4)
		c2 = (nrows/3,self.HEIGHT/2)
		c3 = (nrows,-100)
		x_mat = np.array([[c1[0]**2,c1[0],1], [c2[0]**2,c2[0],1], [c3[0]**2,c3[0],1]])
		y_mat = np.array([[c1[1]],[c2[1]],[c3[1]]])
		x_mat_inv = np.linalg.inv(x_mat)
		c_mat = np.matmul(x_mat_inv, y_mat)
		a = c_mat[0,0]
		b = c_mat[1,0]
		c = c_mat[2,0]
		self.persistantDisplayData['bargrahpHill_a'] = a
		self.persistantDisplayData['bargrahpHill_b'] = b
		self.persistantDisplayData['bargrahpHill_c'] = c
	

	for row in range(0,nrows-1):
		xoff = x_max - x_stride * row - 30
		yoff = y_max - (self.persistantDisplayData['bargrahpHill_a'] * (row ** 2) + self.persistantDisplayData['bargrahpHill_b'] * row + self.persistantDisplayData['bargrahpHill_c']) #y_start - y_step_shrink * row * row - y_stride * row
		# yoff = y_start/2  np.abs(y_start/2 - y_stride * row)
		zoff = -(row+1)

		downsampled = signal.resample(self.fftArray[:,zoff], self.numBars)
		downsampled = downsampled * 0.75
		
		#compute peaks
		for channel in range(len(downsampled)):
			if self.currentDisplayData['minfft'][channel] == -1: #check if minfft is initialized.
				self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
			if self.currentDisplayData['minfft'][channel] > downsampled[channel]: #if the new sample is less than the minimum, update the minimum.
				self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
			downsampled[channel] = np.abs(downsampled[channel] - self.currentDisplayData['minfft'][channel])
		#draw bars
		barWidth = int(self.WIDTH/(self.numBars+4))
		for xIdx in range(self.numBars):
			xPos = int((xIdx+1) / (self.numBars+1) * self.WIDTH)
			barcolor = list(self.currentDisplayData['FGcolorTuple'])
			rs = (row+1)/nrows
			barcolor[0] = rs * barcolor[0]
			barcolor[1] = rs * barcolor[1]
			barcolor[2] = rs * barcolor[2]
			barcolor = tuple(barcolor)
			pygame.draw.line(screen, barcolor, (xPos+xoff,yoff), (int(xPos+xoff),int(yoff-downsampled[xIdx])), barWidth)

		# sampledTuple = fftline(np.array(self.fftArray[:,zoff])/2, self.WIDTH,xoff,yoff,100)
		# pygame.draw.lines(self.screen,color,False,np.array(sampledTuple).astype(np.int64),5)

def fftHill(self, data, screen):
	color = self.currentDisplayData['FGcolorTuple']
	nrows = 20

	x_max = self.WIDTH/5
	y_max = self.HEIGHT*1.2
	x_stride = x_max/nrows
	y_stride = y_max/nrows
	z_stride = 4

	x_start = x_max
	# y_start = c
	z_start = 0

	if not 'fftHill_a' in self.persistantDisplayData:
		c1 = (0,self.HEIGHT*7/8)
		c2 = (nrows/3,self.HEIGHT)
		c3 = (nrows,0)
		x_mat = np.array([[c1[0]**2,c1[0],1], [c2[0]**2,c2[0],1], [c3[0]**2,c3[0],1]])
		y_mat = np.array([[c1[1]],[c2[1]],[c3[1]]])
		x_mat_inv = np.linalg.inv(x_mat)
		c_mat = np.matmul(x_mat_inv, y_mat)
		a = c_mat[0,0]
		b = c_mat[1,0]
		c = c_mat[2,0]
		self.persistantDisplayData['fftHill_a'] = a
		self.persistantDisplayData['fftHill_b'] = b
		self.persistantDisplayData['fftHill_c'] = c
	


	for row in range(0,nrows-1):
		xoff = x_max - x_stride * row
		yoff = y_max - (self.persistantDisplayData['fftHill_a'] * (row ** 2) + self.persistantDisplayData['fftHill_b'] * row + self.persistantDisplayData['fftHill_c']) #y_start - y_step_shrink * row * row - y_stride * row
		# yoff = y_start/2  np.abs(y_start/2 - y_stride * row)
		zoff = -(row+1)
		sampledTuple = fftline(np.array(self.fftArray[:,zoff])/2, self.WIDTH,xoff,yoff,100)
		pygame.draw.lines(self.screen,color,False,np.array(sampledTuple).astype(np.int64),5)


def fftSignal(self, data, screen):
	color = self.currentDisplayData['FGcolorTuple']	
	sampledTuple = fftline(self.fftArray[:,-1], self.WIDTH,0,round(self.HEIGHT * 0.9),100)
	pygame.draw.lines(self.screen,color,False,np.array(sampledTuple).astype(np.int64),5)
     
# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)

if __name__=="__main__":
    # call the main function
    # main([timeSignal,meanFreq,meanDiamonds,timeBackground,fftBackground,peakDiamonds,barchart,spectrogram])
    
	parser = argparse.ArgumentParser(description='Raspberry Pi Video Music. Inspired by the Atari Video Music system.')
	parser.add_argument('-l', action="store_true", help="list audio and video interfaces")
	parser.add_argument('-v', action="store", type=int, default = 0, help="Specify video interface (int)")
	parser.add_argument('-a', action="store", type=int, default = 0, help="Specify audio interface (int)")
	parser.add_argument('-w', action="store_false", help="windowed mode")
	args = parser.parse_args()
	bgs = [spectrogram,background,peakDiamonds,meanDiamonds]
	fgs = [bargraphHill,timeSignal, fftSignal, fftHill, barchart]
	PVM = piVideoMusic(fgs,bgs,listFlag=args.l,videoInterface=args.v,audioInterface=args.a,fullscreen=args.w) #
	while PVM.running:
		PVM.processEvent()