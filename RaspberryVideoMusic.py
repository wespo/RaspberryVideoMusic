#audioGameTest
# import the pygame module, so you can use it
import pygame
import pyaudio
import numpy as np
from scipy import signal
from scipy import interpolate
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

f = np.array(
    [
        20,
        25,
        31.5,
        40,
        50,
        63,
        80,
        100,
        125,
        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000,
        6300,
        8000,
        10000,
        12500,
    ]
)
af = np.array(
    [
        0.532,
        0.506,
        0.480,
        0.455,
        0.432,
        0.409,
        0.387,
        0.367,
        0.349,
        0.330,
        0.315,
        0.301,
        0.288,
        0.276,
        0.267,
        0.259,
        0.253,
        0.250,
        0.246,
        0.244,
        0.243,
        0.243,
        0.243,
        0.242,
        0.242,
        0.245,
        0.254,
        0.271,
        0.301,
    ]
)
Lu = np.array(
    [
        -31.6,
        -27.2,
        -23.0,
        -19.1,
        -15.9,
        -13.0,
        -10.3,
        -8.1,
        -6.2,
        -4.5,
        -3.1,
        -2.0,
        -1.1,
        -0.4,
        0.0,
        0.3,
        0.5,
        0.0,
        -2.7,
        -4.1,
        -1.0,
        1.7,
        2.5,
        1.2,
        -2.1,
        -7.1,
        -11.2,
        -10.7,
        -3.1,
    ]
)
Tf = np.array(
    [
        78.5,
        68.7,
        59.5,
        51.1,
        44.0,
        37.5,
        31.5,
        26.5,
        22.1,
        17.9,
        14.4,
        11.4,
        8.6,
        6.2,
        4.4,
        3.0,
        2.2,
        2.4,
        3.5,
        1.7,
        -1.3,
        -4.2,
        -6.0,
        -5.4,
        -1.5,
        6.0,
        12.6,
        13.9,
        12.3,
    ]
)

def printArray(arrayToPlot,cls=True):
	if(cls):
		os.system('cls')
	print("Prettyimage: ")
	print('\n'.join([''.join(['{:2.4}\t'.format(item) for item in row]) for row in arrayToPlot]))

def elc(phon, frequencies=None):
    """Returns an equal-loudness contour.

    Args:
        phon (float): Phon value of the contour.
        frequencies (:obj:`np.ndarray`, optional): Frequencies to evaluate. If not
            passed, all 29 points of the ISO standard are returned. Any frequencies not
            present in the standard are found via spline interpolation.

    Returns:
        contour (np.ndarray): db SPL values.

    """
    assert 0 <= phon <= 90, f"{phon} is not [0, 90]"
    Ln = phon
    Af = (
        4.47e-3 * (10 ** (0.025 * Ln) - 1.15)
        + (0.4 * 10 ** (((Tf + Lu) / 10) - 9)) ** af
    )
    Lp = ((10.0 / af) * np.log10(Af)) - Lu + 94

    if frequencies is not None:

        assert frequencies.min() >= f.min(), "Frequencies are too low"
        assert frequencies.max() <= f.max(), "Frequencies are too high"
        tck = interpolate.splrep(f, Lp, s=0)
        Lp = interpolate.splev(frequencies, tck, der=0)

    return Lp

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
def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
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
	def __init__(self,displayFunctions=None,backgroundFunctions=None,listFlag=False,videoInterface=0,audioInterface=0,fullscreen=True, verbose=False, thin=False):
		self.CHUNK=2205
		self.nFFTFrames = 1
		self.FFTLEN = self.nFFTFrames*(self.CHUNK//2 + 1)
		self.FFTOverlap = 1/2 #offset of next frame. 0=frameA&B are identical.
		self.RATE=44100
		self.WIDTH=650
		self.HEIGHT=450
		self.time_to_buf_fft = 4 #sec
		self.signal_pk = 0
		self.colorlist = range(0,256,32)
		self.tau = 0.05 #set tau = 0.2 -> 5*tau = 1sec
		self.numBars = 12
		self.verbose = verbose
		self.diagCount = 0
		self.fgSet = 'dark'
		self.thin = thin

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

		self.logSpaceIndices = np.floor(np.logspace(0.41,3.34,self.CHUNK)).astype(int) #consen for visual pleasantness.
		# self.LogPts = [0,20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]
		self.LogPts = [0,80,160,200,250,315,400,500,630,800,1000,1250,1600,2000,2500,3150,4000,5000,6300,8000,10000,12500,16000,20000]

		self.colorsDark = [(0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170), (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170), (85, 85, 85)] #EGA pallet
		self.colorsBright = [(85, 85, 255), (85, 255, 85), (85, 255, 255), (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255)]

		# self.Lp = elc(60,np.linspace(20,12500,1250)) #1250 is a magic constant, should be computed from CHUNK and RATE to get the the number of bins between 20 and 12500Hz (limits of elc function)
		# self.Lp = np.pad(self.Lp,(0,self.CHUNK-len(self.Lp)),'edge')
		# self.Lp = self.Lp / np.mean(self.Lp) / 1.2

		self.fftArray = np.zeros([self.FFTLEN,int(self.time_to_buf_fft*self.RATE/self.CHUNK)]) #np.zeros([self.FFTLEN,int(self.time_to_buf_fft*self.RATE/self.CHUNK)])

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
			drivers = ['fbcon', 'directfb', 'svgalib','dummy']
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
			# pygame.display.set_mode((480, 640))
			# pygame.display.set_mode((self.WIDTH,self.HEIGHT))
			# size = (pygame.display.Info().current_w, pygame.display.Info().current_h)
			size = (self.WIDTH,self.HEIGHT)
			self.size = size
			# self.WIDTH = pygame.display.Info().current_w
			# self.HEIGHT = pygame.display.Info().current_h
			print(pygame.display.list_modes())
			print(size)
			flags = pygame.DOUBLEBUF | pygame.HWSURFACE | pygame.FULLSCREEN
			self.screen = pygame.display.set_mode((size), flags)
			surface = pygame.Surface(size,pygame.FULLSCREEN)
			# pygame.display.set_mode((640,480),pygame.FULLSCREEN)
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
		self.stream=p.open(format = pyaudio.paInt16,rate=self.RATE,channels=1, input_device_index = audioInterface, input=True, frames_per_buffer=self.CHUNK, stream_callback=self.processBuffer) #on Win7 PC, 1 = Microphone, 3 = stereo mix (enabled in sound control panel)

		#display state initialization:
		self.nextChangeTime = time.time()
		self.currentDisplay = self.displayFunctions[0]
		self.currentBackground = self.backgroundFunctions[0]

	def generateColorTuplesSet(self):
		#bright = [(235, 52, 52),(235, 122, 52),(255, 230, 0),(0, 255, 128),(0, 255, 255),(120, 122, 255),(201, 120, 255),(255, 120, 237),(255, 120, 169),(255, 120, 120),(255,255,255),(219, 182, 15)] #arbitrary bill generated pallet
		#dark = [(0,0,0),(156, 0, 0),(156, 117, 0),(102, 77, 48),(107, 107, 107),(17, 0, 255),(67, 0, 138),(150, 57, 153),(21, 115, 24),(219, 182, 15)]
		dark = self.colorsDark
		bright = self.colorsBright

		fgGroup = random.choice(['Bright','Dark'])
		if fgGroup == 'Bright':
			self.fgSet = 'bright'
			fg1 = random.choice(bright)
			fg2 = random.choice(bright)
			bg1 = random.choice(dark)
			bg2 = random.choice(dark)
		else:
			self.fgSet = 'dark'
			bg1 = random.choice(bright)
			bg2 = random.choice(bright)
			fg1 = random.choice(dark)
			fg2 = random.choice(dark)
		return fg1, fg2, bg1, bg2
	def generateColorTuplesOld(self):
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
		return fg1, fg2, bg1, bg2
	def generateColorTuplesNew(self):
		minBackgroundVal = 0 #at least one channel of the bg1 color must be higher than this value.
		minDelta = 128 #at least one channel of bg1 and fg1 colors must be more than this amount apart.
		fg1 = [random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist)]
		bg1 = [random.choice(self.colorlist),random.choice(self.colorlist),random.choice(self.colorlist)]
		if max(bg1) <= minBackgroundVal:
			minAllowedColor = list(filter(lambda i: i > minBackgroundVal, self.colorlist))[0]
			minAllowedColorIdx = self.colorlist.index(minAllowedColor)
			bg1[random.choice([0,1,2])] = random.choice(self.colorlist[minAllowedColorIdx:])
		fgbg = np.abs((fg1[0]-bg1[0],fg1[1]-bg1[1],fg1[2]-bg1[2]))
		iter_break = 0
		if self.verbose:
			print("Tweaking colors:", fg1, bg1, 0)
		while max(fgbg) < minDelta:
			if iter_break > 100:
				print("ERROR! Too many tries to select color!")
				break
			else:
				iter_break = iter_break + 1

			#find which column has the max delta
			maxdelta = max(fgbg)
			maxdelta_idx = list(fgbg).index(maxdelta)
			#pull the FG and BG values
			fgMaxDelta = fg1[maxdelta_idx]
			bgMaxDelta = bg1[maxdelta_idx]
			cll = len(self.colorlist)-1
			if fgMaxDelta > bgMaxDelta: #fg is bigger
				if (255-fgMaxDelta) > bgMaxDelta: #fg is further from top than bg is from bottom, increase fg
					# fg1[maxdelta_idx] = self.colorlist[self.colorlist.index(fgMaxDelta)+1] #older code, only increases largest delta channel -- missing bugfix for max or min value.
					fg1[0] = self.colorlist[min(self.colorlist.index(fg1[0])+1, cll)] #newer code, increaseas all channels
					fg1[1] = self.colorlist[min(self.colorlist.index(fg1[1])+1, cll)]
					fg1[2] = self.colorlist[min(self.colorlist.index(fg1[2])+1, cll)]
				else: #bg is further from bottom than fg is from top, decrease bg
					# bg1[maxdelta_idx] = self.colorlist[self.colorlist.index(bgMaxDelta)-1] #older code, only increases largest delta channel
					bg1[0] = self.colorlist[max(self.colorlist.index(bg1[0])-1, 0)]
					bg1[1] = self.colorlist[max(self.colorlist.index(bg1[1])-1, 0)]
					bg1[2] = self.colorlist[max(self.colorlist.index(bg1[2])-1, 0)]
			else: #bg is bigger
				if (255-bgMaxDelta) > fgMaxDelta: #bg is further from top than fg is from bottom, increase bg
					# bg1[maxdelta_idx] = self.colorlist[self.colorlist.index(bgMaxDelta)+1] #older code, only increases largest delta channel
					bg1[0] = self.colorlist[min(self.colorlist.index(bg1[0])+1, cll)]
					bg1[1] = self.colorlist[min(self.colorlist.index(bg1[1])+1, cll)]
					bg1[2] = self.colorlist[min(self.colorlist.index(bg1[2])+1, cll)]
				else: #fg is further from bottom than bg is from top, decrease fg
					# fg1[maxdelta_idx] = self.colorlist[self.colorlist.index(fgMaxDelta)-1] #older code, only increases largest delta channel
					fg1[0] = self.colorlist[max(self.colorlist.index(fg1[0])-1, 0)]
					fg1[1] = self.colorlist[max(self.colorlist.index(fg1[1])-1, 0)]
					fg1[2] = self.colorlist[max(self.colorlist.index(fg1[2])-1, 0)]
			fgbg = np.abs((fg1[0]-bg1[0],fg1[1]-bg1[1],fg1[2]-bg1[2]))
			if self.verbose:
				print("Tweaking colors:", fg1, bg1, iter_break)

		#create FG2 by applying a random HSV hue to FG1
		fg2 = colorsys.rgb_to_hsv(fg1[0]/255,fg1[1]/255,fg1[2]/255)
		fg2 = list(colorsys.rgb_to_hsv(random.random(),fg2[1],fg2[2]))
		fg2[0] = closest(self.colorlist,fg2[0]*255)
		fg2[1] = closest(self.colorlist,fg2[1]*255)
		fg2[2] = closest(self.colorlist,fg2[2]*255)

		#create BG2 by applying a random HSV hue to BG1
		bg2 = colorsys.rgb_to_hsv(bg1[0]/255,bg1[1]/255,bg1[2]/255)
		bg2 = list(colorsys.rgb_to_hsv(random.random(),bg2[1],bg2[2]))
		bg2[0] = closest(self.colorlist,bg2[0]*255)
		bg2[1] = closest(self.colorlist,bg2[1]*255)
		bg2[2] = closest(self.colorlist,bg2[2]*255)

		bg2 = bg1
		#convert to tuples.
		fg1 = tuple(fg1)
		fg2 = tuple(fg2)
		bg1 = tuple(bg1)
		bg2 = tuple(bg2)

		return fg1, fg2, bg1, bg2
	def generateColorTuples(self):
		return self.generateColorTuplesSet()
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
	def lintoaudio(self,vector_in):
		linBinSize = round(self.RATE/(2*self.fftArray.shape[0]))
		result = np.zeros(len(self.LogPts)-1)
		# if self.diagCount < 20:
		# 	print("Diagnostics:")
		# print(f'\t{vector_in}')
		for idx in range(len(self.LogPts)-1):
			startFreq = self.LogPts[idx]
			endFreq = self.LogPts[idx+1]
			startBin = int(np.floor(startFreq/linBinSize))
			endBin = int(np.floor(endFreq/linBinSize))

			startBinMaxF = (startBin+1)*linBinSize
			startBinFrac = (startBinMaxF - startFreq)/linBinSize

			endBinMinF = endBin*linBinSize
			endBinFrac = (endFreq-endBinMinF)/linBinSize

			if startBin == endBin:
				endBinFrac = (endFreq-startFreq)/linBinSize
				startBinFrac = 0


			binAccum = 0
			binAccum = vector_in[startBin]*startBinFrac
			for binNum in range(startBin+1,endBin):
				binAccum = binAccum + vector_in[binNum]
			binAccum = binAccum + vector_in[endBin]*endBinFrac
			scaleFac = (endFreq-startFreq)
			result[idx] = binAccum/scaleFac*75
			#diagnostics.
			# if self.diagCount < 20:
			# 	print(f'\tidx:{idx}\tstartBin:{startBin}\tendBin:{endBin}\tstartFreq:{startFreq}\tendFreq:{endFreq}\tstartBinMaxF:{startBinMaxF}\tendBinMinF:{endBinMinF}\tstartBinFrac:{startBinFrac}\tendBinFrac:{endBinFrac}\tscaleFac:{scaleFac}\tbinAccum:{binAccum}')

		# if self.diagCount < 20:
		# 	self.diagCount = self.diagCount + 1
		# 	print(result)
		return result


	def fftwindow(self, data):
		window = signal.hamming(len(data))
		w_data = data * window
		fft_data = np.absolute(rfft(w_data))
		fft_data = 100 * np.log(fft_data*self.digitalGain)
		fft_data = np.clip(np.nan_to_num(fft_data),0,1000)
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
		data = np.diff(data)
		newFrame1 = np.array(self.fftwindow(data[-self.CHUNK*self.nFFTFrames::1]))
		newFrame2 = np.array(self.fftwindow(data[-int(self.CHUNK*(1+self.FFTOverlap)*self.nFFTFrames):-int(self.CHUNK*self.FFTOverlap*self.nFFTFrames):1]))
		newFrame = np.column_stack((newFrame1,newFrame2))
		self.fftArray = shiftIn2DCols(self.fftArray, newFrame)

		return signal_pk

	def updateFrame(self,screen, data, displayFunctions, backgroundFunctions, nextChangeTime, currentDisplay, currentDisplayData, currentBackground, signal_pk): #update the screen
		currentTime = time.time();

		if 'extendChangeTime' in self.currentDisplayData:
			nextChangeTime = nextChangeTime + self.currentDisplayData['extendChangeTime']
			del self.currentDisplayData['extendChangeTime']

		if(currentTime > nextChangeTime):
			newFG = ListDiff(displayFunctions,[currentDisplay])
			newBG = ListDiff(backgroundFunctions,[currentBackground])
			currentDisplayData=self.refreshCurrentDisplay()
			if newFG:
				currentDisplay = random.choice(newFG) #don't hop to the same display
			if newBG:
				currentBackground = random.choice(newBG) #don't hop to the same display
			nextChangeTime = currentTime + random.randrange(3,7)
		if 'noBackground' in self.currentDisplayData:
			currentBackground = skip

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
	surfa = pygame.transform.scale(surf,(int(self.WIDTH/2),int(self.HEIGHT/2)))
	surfb = pygame.transform.flip(surfa,True, False)
	surfc = pygame.transform.flip(surfa, False ,True)
	surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfb, (0,0))
	screen.blit(surfa, (int(self.WIDTH/2),0))
	screen.blit(surfd, (0,int(self.HEIGHT/2)))
	screen.blit(surfc, (int(self.WIDTH/2),int(self.HEIGHT/2)))

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
	surfa = pygame.transform.scale(surf,(int(self.WIDTH/2),int(self.HEIGHT/2)))
	surfb = pygame.transform.flip(surfa,True, False)
	surfc = pygame.transform.flip(surfa, False ,True)
	surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfb, (0,0))
	screen.blit(surfa, (int(self.WIDTH/2),0))
	screen.blit(surfd, (0,int(self.HEIGHT/2)))
	screen.blit(surfc, (int(self.WIDTH/2),int(self.HEIGHT/2)))

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
	pygame.draw.line(screen, (255,0,0), (0,int(self.HEIGHT/2)-self.signal_pk), (self.WIDTH,int(self.HEIGHT/2)-signal_pk), 3)

def timeSignal(self, data, screen, color = None):
	if color == None:
		color = self.currentDisplayData['FGcolorTuple']
	downsampled = signal.resample(data, self.WIDTH)
	downsampled = downsampled * self.digitalGain / 4

	sampledTuple =[]
	xoff = 0
	xstep = self.WIDTH/len(downsampled)
	yoff = int(self.HEIGHT/2) + np.mean(downsampled)
	for sample in downsampled:
		sampledTuple.append((round(xoff),round(yoff-sample)))
		xoff = xoff + xstep

	pygame.draw.lines(screen,color,False,np.array(sampledTuple).astype(np.int64),5)



def spectrogram(self,data,screen):
	if not 'blitArray' in self.currentDisplayData:
		specArray = np.transpose(signal.resample (self.fftArray, 64, axis = 0))
		blitArray = np.zeros((specArray.shape[0],specArray.shape[1],3))
		specRes = np.clip(specArray/5,0,255)
		if self.fgSet == 'dark':
			blitArray[:,:,0] = specRes * (self.currentDisplayData['BGcolorTuple1'][0]/511 + 0.5)
			blitArray[:,:,1] = specRes * (self.currentDisplayData['BGcolorTuple1'][1]/511 + 0.5)
			blitArray[:,:,2] = specRes * (self.currentDisplayData['BGcolorTuple1'][2]/511 + 0.5)
		else:
			blitArray[:,:,0] = 1 - specRes * (self.currentDisplayData['BGcolorTuple1'][0]/511 + 0.5)
			blitArray[:,:,1] = 1 - specRes * (self.currentDisplayData['BGcolorTuple1'][1]/511 + 0.5)
			blitArray[:,:,2] = 1 - specRes * (self.currentDisplayData['BGcolorTuple1'][2]/511 + 0.5)
		self.currentDisplayData['blitArray'] = blitArray
	else:
		self.currentDisplayData['blitArray'] = np.roll(self.currentDisplayData['blitArray'],-1,0)
		fftSlice =  np.clip(signal.resample (self.fftArray[:,-1], 64)/5,0,255)/255
		self.currentDisplayData['blitArray'][-1,:,0] = fftSlice * self.currentDisplayData['BGcolorTuple1'][0]
		self.currentDisplayData['blitArray'][-1,:,1] = fftSlice * self.currentDisplayData['BGcolorTuple1'][1]
		self.currentDisplayData['blitArray'][-1,:,2] = fftSlice * self.currentDisplayData['BGcolorTuple1'][2]

	surf = pygame.surfarray.make_surface(self.currentDisplayData['blitArray'])
	surfa = pygame.transform.scale(surf,self.size)
	# surfd = pygame.transform.flip(surfa,True, True)

	screen.blit(surfa, (0,0))


def barchartLogspace(self, data, screen):
	# background(data,screen)
	downsampled = (self.lintoaudio(self.fftArray[:,-1])/750)**4
	if not 'barchart_logspace' in self.currentDisplayData:
		self.currentDisplayData['barchart_logspace'] = np.zeros(len(downsampled))-1
		self.currentDisplayData['barchart_logspacePeaks'] = np.zeros(len(downsampled))


	# print(len(downsampled))
	#compute peaks
	for channel in range(len(downsampled)):
		if self.currentDisplayData['barchart_logspace'][channel] == -1: #check if minfft is initialized.
			self.currentDisplayData['barchart_logspace'][channel] = np.abs(downsampled[channel])
		if self.currentDisplayData['barchart_logspace'][channel] > downsampled[channel]: #if the new sample is less than the minimum, update the minimum.
			self.currentDisplayData['barchart_logspace'][channel] = np.abs(downsampled[channel])
		downsampled[channel] = np.abs(downsampled[channel] - self.currentDisplayData['barchart_logspace'][channel])
		self.currentDisplayData['barchart_logspacePeaks'][channel] = peakDecay(downsampled[channel], self.currentDisplayData['barchart_logspacePeaks'][channel], self.tau*5, self.CHUNK, self.RATE)
	#draw bars
	barWidth = int(self.WIDTH/(len(self.LogPts)-1+4))
	for xIdx in range(len(downsampled)):
		xPos = int((xIdx+1) / (len(self.LogPts)-1+1) * self.WIDTH)
		pygame.draw.line(screen, self.currentDisplayData['FGcolorTuple'], (xPos,self.HEIGHT), (int(xPos),int(self.HEIGHT-downsampled[xIdx])), barWidth)
		pygame.draw.line(screen, self.currentDisplayData['FGcolorTuple2'], (int(xPos-barWidth/2),int(self.HEIGHT-self.currentDisplayData['barchart_logspacePeaks'][xIdx])), (int(xPos+barWidth/2),int(self.HEIGHT-self.currentDisplayData['barchart_logspacePeaks'][xIdx])), 7)
	return

def barchart(self, data, screen):
	# background(data,screen)
	if not 'barchart_minfft' in self.currentDisplayData:
		self.currentDisplayData['barchart_minfft'] = np.zeros(self.numBars)-1
		self.currentDisplayData['barchart_fftBargraphPeaks'] = np.zeros(self.numBars)

	downsampled = signal.resample(self.fftArray[:,-1], self.numBars)
	downsampled = downsampled * 0.75
	#compute peaks
	for channel in range(len(downsampled)):
		if self.currentDisplayData['barchart_minfft'][channel] == -1: #check if minfft is initialized.
			self.currentDisplayData['barchart_minfft'][channel] = np.abs(downsampled[channel])
		if self.currentDisplayData['barchart_minfft'][channel] > downsampled[channel]: #if the new sample is less than the minimum, update the minimum.
			self.currentDisplayData['barchart_minfft'][channel] = np.abs(downsampled[channel])
		downsampled[channel] = np.abs(downsampled[channel] - self.currentDisplayData['barchart_minfft'][channel])
		self.currentDisplayData['barchart_fftBargraphPeaks'][channel] = peakDecay(downsampled[channel], self.currentDisplayData['barchart_fftBargraphPeaks'][channel], self.tau*5, self.CHUNK, self.RATE)
	#draw bars
	barWidth = int(self.WIDTH/(self.numBars+4))
	for xIdx in range(self.numBars):
		xPos = int((xIdx+1) / (self.numBars+1) * self.WIDTH)
		pygame.draw.line(screen, self.currentDisplayData['FGcolorTuple'], (xPos,self.HEIGHT), (int(xPos),int(self.HEIGHT-downsampled[xIdx])), barWidth)
		pygame.draw.line(screen, self.currentDisplayData['FGcolorTuple2'], (int(xPos-barWidth/2),int(self.HEIGHT-self.currentDisplayData['barchart_fftBargraphPeaks'][xIdx])), (int(xPos+barWidth/2),int(self.HEIGHT-self.currentDisplayData['barchart_fftBargraphPeaks'][xIdx])), 7)
	return

def fftline(frame,width,xoff=0,yoff=0,nsamp=None,scale_denom=4):
	if nsamp == None:
		nsamp = width
		downsampled = frame
	else:
		downsampled = signal.resample(frame,nsamp)
	if scale_denom is not None:
		downsampled = downsampled/scale_denom
	sampledTuple =[]
	xstep = width/len(downsampled)
	for sample in downsampled:
		sampledTuple.append((round(xoff),round(yoff-sample)))
		xoff = xoff + xstep
	return sampledTuple


def fftSurf(frame,width,color=(0,0,0),nsamp=None,scale_denom=4, thickness=5): #draws an FFT frame on a surface and returns it.

	if nsamp == None:
		nsamp = width
		downsampled = frame
	else:
		downsampled = signal.resample(frame,nsamp)
	if scale_denom is not None:
		downsampled = downsampled/scale_denom
	sampledTuple =[]
	xoff = 0
	xstep = width/len(downsampled)
	frame_max = max(downsampled)
	for sample in downsampled:
		sampledTuple.append((round(xoff),round(frame_max - sample)))
		xoff = xoff + xstep
	size = (int(width), int(frame_max))
	surf = pygame.Surface(size, pygame.SRCALPHA)
	pygame.draw.lines(surf,color,False,np.array(sampledTuple).astype(np.int64),thickness)
	return surf
def fftWaterfall(self, data, screen): #old, inefficient, not pretty
	color = self.currentDisplayData['FGcolorTuple']
	nrows = 15
	thickness = 5

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
		pygame.draw.lines(self.screen,color,False,np.array(sampledTuple).astype(np.int64),thickness)

#faster version, but has issues with initialization
def bargraphHill(self, data, screen):
	color = self.currentDisplayData['FGcolorTuple']
	nrows = 15
	if not 'bargrahpHill_a' in self.persistantDisplayData:
		c1 = (0,self.HEIGHT/4)
		c2 = (nrows/3,int(self.HEIGHT/2))
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
		self.persistantDisplayData['bargrahpHill_xoff'] = np.zeros(nrows)
		self.persistantDisplayData['bargrahpHill_yoff'] = np.zeros(nrows)
		x_max = self.WIDTH/5
		y_max = self.HEIGHT
		x_stride = x_max/nrows
		y_stride = y_max/nrows
		z_stride = 4

		x_start = x_max
		z_start = 0

		for row in range(0,nrows-1):
			self.persistantDisplayData['bargrahpHill_xoff'][row] = x_max - x_stride * row - 30
			self.persistantDisplayData['bargrahpHill_yoff'][row] = y_max - (self.persistantDisplayData['bargrahpHill_a'] * (row ** 2) + self.persistantDisplayData['bargrahpHill_b'] * row + self.persistantDisplayData['bargrahpHill_c']) #y_start - y_step_shrink * row * row - y_stride * row
	if not 'bargrahpHill_downsampled' in self.currentDisplayData:
		self.currentDisplayData['bargrahpHill_downsampled'] = np.zeros((nrows,self.numBars))
		self.currentDisplayData['minfft'] = np.zeros((self.numBars))-1
		for row in range(0,nrows-1):
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
			self.currentDisplayData['bargrahpHill_downsampled'][row] = downsampled
			#draw bars
	else:
		downsampled = signal.resample(self.fftArray[:,-1], self.numBars)
		downsampled = downsampled * 0.75
		for channel in range(len(downsampled)):
			if self.currentDisplayData['minfft'][channel] == -1: #check if minfft is initialized.
				self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
			if self.currentDisplayData['minfft'][channel] > downsampled[channel]: #if the new sample is less than the minimum, update the minimum.
				self.currentDisplayData['minfft'][channel] = np.abs(downsampled[channel])
			downsampled[channel] = np.abs(downsampled[channel] - self.currentDisplayData['minfft'][channel])
		self.currentDisplayData['bargrahpHill_downsampled'] = np.roll(self.currentDisplayData['bargrahpHill_downsampled'],1,0)
		self.currentDisplayData['bargrahpHill_downsampled'][0] = downsampled
	barWidth = int(self.WIDTH/(self.numBars+4))
	for row in range(0,nrows-1):
		for xIdx in range(self.numBars):
			xPos = int((xIdx+1) / (self.numBars+1) * self.WIDTH)
			barcolor = list(self.currentDisplayData['FGcolorTuple'])
			rs = (row+1)/nrows
			barcolor[0] = rs * barcolor[0]
			barcolor[1] = rs * barcolor[1]
			barcolor[2] = rs * barcolor[2]
			barcolor = tuple(barcolor)
			pygame.draw.line(screen, barcolor, (int(xPos+self.persistantDisplayData['bargrahpHill_xoff'][row]),int(self.persistantDisplayData['bargrahpHill_yoff'][row])), (int(xPos+self.persistantDisplayData['bargrahpHill_xoff'][row]),int(self.persistantDisplayData['bargrahpHill_yoff'][row]-self.currentDisplayData['bargrahpHill_downsampled'][row][xIdx])), barWidth)

def fftHill(self, data, screen):
	color = self.currentDisplayData['FGcolorTuple']
	nrows = 17
	nsamp = 25
	x_max = self.WIDTH/5
	y_max = self.HEIGHT*1.5
	x_stride = x_max/nrows
	y_stride = y_max/nrows
	z_stride = 4

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
		self.persistantDisplayData['fftHill_xoff'] = np.zeros(nrows)
		self.persistantDisplayData['fftHill_yoff'] = np.zeros(nrows)
		for row in range(0,nrows-1):
			self.persistantDisplayData['fftHill_xoff'][row] = x_max - x_stride * row
			self.persistantDisplayData['fftHill_yoff'][row] = y_max - (a * (row ** 2) + b * row + c) #y_start - y_step_shrink * row * row - y_stride * row
	if not 'fftHill_buffer' in self.currentDisplayData:
		self.currentDisplayData['fftHill_scale_fac'] = 3
		self.currentDisplayData['fftHill_buffer'] = []
		for row in range(0,nrows-1):
			zoff = -(row+1)
			surf = fftSurf(np.array(self.fftArray[:,zoff]), self.WIDTH, color=color,nsamp=nsamp,scale_denom=self.currentDisplayData['fftHill_scale_fac'])
			self.currentDisplayData['fftHill_buffer'].append(surf)
	else:
		self.currentDisplayData['fftHill_buffer'].pop()
		surf = fftSurf(np.array(self.fftArray[:,-1]), self.WIDTH, color=color,nsamp=nsamp,scale_denom=self.currentDisplayData['fftHill_scale_fac'])
		self.currentDisplayData['fftHill_buffer'].insert(0,surf)


	for row in range(0,nrows-1):
		zoff = -(row+1)
		currentRow = self.currentDisplayData['fftHill_buffer'][row]
		w, h = currentRow.get_size()
		self.screen.blit(currentRow, (int(self.persistantDisplayData['fftHill_xoff'][row]),int(self.persistantDisplayData['fftHill_yoff'][row]-h)))

def fftSignal(self, data, screen):
	if self.thin:
		thickness = 5
	else:
		thickness = 15

	if not 'fftSignal_flag' in self.currentDisplayData:
		self.currentDisplayData['fftSignal_flag'] = True
		self.currentDisplayData['oldSamples'] = np.zeros(self.FFTLEN)
	color = self.currentDisplayData['FGcolorTuple']
	yoff = round(self.HEIGHT * 1.0)
	floor = 100
	samples = np.clip(self.fftArray[:,-1],0,None)
	samples = [peakDecay(samples[x],self.currentDisplayData['oldSamples'][x],0.05,self.CHUNK,self.RATE) for x in range(len(samples))]
	self.currentDisplayData['oldSamples'] = samples
	sampledTuple = fftline(samples,self.WIDTH,0,yoff,50,2)
	pygame.draw.lines(self.screen,color,False,np.array(sampledTuple).astype(np.int64),thickness)
	pygame.draw.line(self.screen,color,(0,yoff-floor),(self.WIDTH,yoff-floor),thickness)

def fftQuad(self, data, screen):
	if self.thin:
		thickness = 5
	else:
		thickness = 15
	color = self.currentDisplayData['FGcolorTuple']
	sampledTuple1 = fftline(self.fftArray[:,-1], int(self.WIDTH/2),int(self.WIDTH/2),round(self.HEIGHT * 0.5),25)
	sampledTuple2 = fftline(self.fftArray[:,-1], -int(self.WIDTH/2),int(self.WIDTH/2),round(self.HEIGHT * 0.5),25)
	sampledTuple3 = fftline(-1*self.fftArray[:,-1], int(self.WIDTH/2),int(self.WIDTH/2),round(self.HEIGHT * 0.5),25)
	sampledTuple4 = fftline(-1*self.fftArray[:,-1], -int(self.WIDTH/2),int(self.WIDTH/2),round(self.HEIGHT * 0.5),25)
	pygame.draw.lines(self.screen,color,False,np.array(sampledTuple1).astype(np.int64),thickness)
	pygame.draw.lines(self.screen,color,False,np.array(sampledTuple2).astype(np.int64),thickness)
	pygame.draw.lines(self.screen,color,False,np.array(sampledTuple3).astype(np.int64),thickness)
	pygame.draw.lines(self.screen,color,False,np.array(sampledTuple4).astype(np.int64),thickness)

def skip(self,data,screen):
	#dummy placeholded function, do nothing.
	return

def fftZoom(self,data,screen):
	if self.thin:
		thickness = 2
	else:
		thickness = 3
	zoomMax = 162
	zoomMin = 2
	color = self.currentDisplayData['FGcolorTuple']
	yoff = round(self.HEIGHT * 1.0)
	# innerSample = 10*np.log((10**(self.fftArray[2:162,-1]/200))**2)**4/1000-10
	innerSample = (self.fftArray[zoomMin:zoomMax,-1]/200)**4-30
	samplesIn = np.clip(innerSample,0,None)
	if not 'fftzoom_flag' in self.currentDisplayData:
		self.currentDisplayData['fftzoom_flag'] = True
		self.currentDisplayData['oldSamples'] = np.zeros(zoomMax-zoomMin)
	samples = [peakDecay(samplesIn[x],self.currentDisplayData['oldSamples'][x],0.05,self.CHUNK,self.RATE) for x in range(len(samplesIn))]
	self.currentDisplayData['oldSamples'] = samples
	XOFF = int(self.WIDTH/2)
	for sample,x in zip(samples,range(len(samples))):
		pygame.draw.line(self.screen,color,(int(XOFF+x*2*thickness),int(self.HEIGHT)),(int(XOFF+x*2*thickness),int(self.HEIGHT-sample)),thickness)
		pygame.draw.line(self.screen,color,(int(XOFF-x*2*thickness),int(self.HEIGHT)),(int(XOFF-x*2*thickness),int(self.HEIGHT-sample)),thickness)
	return

def multiDrops(self,data,screen):
	# background(data,screen)
	if not 'multiDrops' in self.currentDisplayData:
		self.currentDisplayData['multiDrops'] = True
		self.currentDisplayData['barchart_minfft'] = np.zeros(self.numBars)-1
		self.currentDisplayData['barchart_fftBargraphPeaks'] = np.zeros(self.numBars)
		self.currentDisplayData['surf'] = pygame.Surface((self.WIDTH,self.HEIGHT), pygame.SRCALPHA)
		self.currentDisplayData['surf'].fill(random.choice(self.colorsDark+self.colorsBright))

	downsampled = signal.resample(self.fftArray[:,-1], self.numBars)
	downsampled = downsampled * 0.75
	#compute peaks
	for channel in range(len(downsampled)):
		if self.currentDisplayData['barchart_minfft'][channel] == -1: #check if minfft is initialized.
			self.currentDisplayData['barchart_minfft'][channel] = np.abs(downsampled[channel])
		if self.currentDisplayData['barchart_minfft'][channel] > downsampled[channel]: #if the new sample is less than the minimum, update the minimum.
			self.currentDisplayData['barchart_minfft'][channel] = np.abs(downsampled[channel])
		downsampled[channel] = np.abs(downsampled[channel] - self.currentDisplayData['barchart_minfft'][channel])
		self.currentDisplayData['barchart_fftBargraphPeaks'][channel] = peakDecay(downsampled[channel], self.currentDisplayData['barchart_fftBargraphPeaks'][channel], self.tau*5, self.CHUNK, self.RATE)
	#draw bars
	barWidth = int(self.WIDTH/(self.numBars+4))
	for xIdx in range(self.numBars):
		color = random.choice(self.colorsDark+self.colorsBright)
		yPos = random.randrange(0,self.HEIGHT)
		# xPos = int((xIdx+1) / (self.numBars+1) * self.WIDTH)
		xPos = random.randrange(0,self.WIDTH)
		pygame.draw.circle(self.currentDisplayData['surf'],color,(xPos,yPos),int(downsampled[xIdx]/10))
	screen.blit(self.currentDisplayData['surf'], (0,0))
	return

def drops(self,data,screen):
	# background(data,screen)
	if not 'drops' in self.currentDisplayData:
		self.currentDisplayData['drops'] = True
		self.currentDisplayData['surf'] = pygame.Surface((self.WIDTH,self.HEIGHT), pygame.SRCALPHA)
		self.currentDisplayData['surf'].fill(random.choice(self.colorsDark+self.colorsBright))

	color = random.choice(self.colorsDark+self.colorsBright)
	yPos = random.randrange(0,self.HEIGHT)
	# xPos = int((xIdx+1) / (self.numBars+1) * self.WIDTH)
	xPos = random.randrange(0,self.WIDTH)
	pygame.draw.circle(self.currentDisplayData['surf'],color,(xPos,yPos),int(self.signal_pk/100))
	screen.blit(self.currentDisplayData['surf'], (0,0))
	return

def outRun(self, data, screen):
	nrows = 9 #number of rows to display
	ncols = 50 #number of columns to display
	nbuf = 20 #number of rows to buffer
	color = (85,255,255)
	sunColor = (85,255,255)
	if self.thin:
		thickness = 1
	else:
		thickness = 2
	if not 'outrun_flag' in self.currentDisplayData:
		#Setup basic flags
		self.currentDisplayData['outrun_flag'] = True
		self.currentDisplayData['noBackground'] = True
		self.currentDisplayData['sunPos'] = [[(180,205),(500,205)],[(180,190),(500,190)],[(180,175),(500,175)],[(180,160),(461,160)],[(178,145),(460,145)],[(182,130),(457,130)],[(185,115),(453,115)]]
		picture = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"outrunbg.png"))
		picture = pygame.transform.scale(picture, self.size)
		self.currentDisplayData['outrunbg'] = picture

		# self.currentDisplayData['extendChangeTime'] = random.randrange(5,10) #for some reason, simply extending the change time didn't work...
		if self.thin:
			imagefg = "outrunfg.png"
		else:
			imagefg = "outrunfg_thick.png"
		picture = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),imagefg))

		for n, x in enumerate(self.currentDisplayData['sunPos']):
			self.currentDisplayData['sunPos'][n][0] = (x[0][0]*self.WIDTH/picture.get_width(),x[0][1])
			self.currentDisplayData['sunPos'][n][1] = (x[1][0]*self.WIDTH/picture.get_width(),x[1][1])

		picture = pygame.transform.scale(picture, (self.WIDTH,picture.get_height()))
		self.currentDisplayData['outrunfg'] = picture
		self.currentDisplayData['beatBuffer'] = np.zeros(len(self.currentDisplayData['sunPos']))

	#compute grid
	if not 'outrun_buffer' in self.currentDisplayData: #compute spectrogram buffer.
		self.currentDisplayData['specArray'] = np.zeros((ncols,nbuf))
		self.currentDisplayData['outrun_buffer'] = True
		self.currentDisplayData['minArray'] = np.full(ncols, np.inf)

		for x in range(min(self.fftArray.shape[1],nbuf)):
			sampleData = self.fftArray[:,-(x+1)]
			# sampleData = np.concatenate((np.flip(self.fftArray[0:300,-(x+1)]), self.fftArray[0:300,-(x+1)]))*1.1
			self.currentDisplayData['specArray'][:,x] = signal.resample(sampleData, ncols, axis = 0)**2 #output buffer is filled with columns (axis 2) containing FFTs from most recent to least recent. Most recent has index 0, least recent has index N
			denom = 300000 #np.amax(self.currentDisplayData['specArray'][:,x])
			if denom == 0:
				denom = 1
			self.currentDisplayData['specArray'][:,x] = self.currentDisplayData['specArray'][:,x] / denom * 40
			self.currentDisplayData['minArray'] = np.minimum(self.currentDisplayData['specArray'][:,x], self.currentDisplayData['minArray'])
			self.currentDisplayData['specArray'][:,x] = self.currentDisplayData['specArray'][:,x] - self.currentDisplayData['minArray']#self.currentDisplayData['minArray']
			#note to self, try squaring data...
	else: #update spectrogram buffer.
		self.currentDisplayData['specArray'] = np.roll(self.currentDisplayData['specArray'],1,1) #roll array right by one.
		sampleData = self.fftArray[:,-1]
		# sampleData = np.concatenate((np.flip(self.fftArray[0:300,-1]), self.fftArray[0:300,-1]))*1.1
		self.currentDisplayData['specArray'][:,0] = signal.resample(sampleData, ncols, axis = 0)**2 #overwrite the newest element with the newest one from the fft array
		denom = 300000 #np.amax(self.currentDisplayData['specArray'][:,0])
		if denom == 0:
			denom = 1
		self.currentDisplayData['specArray'][:,0] = self.currentDisplayData['specArray'][:,0] / denom*40
		self.currentDisplayData['minArray'] = np.minimum(self.currentDisplayData['specArray'][:,0], self.currentDisplayData['minArray'])
		self.currentDisplayData['specArray'][:,0] = self.currentDisplayData['specArray'][:,0] - self.currentDisplayData['minArray']#self.currentDisplayData['minArray']

	#compute sun lines
	colorBG = np.clip(self.signal_pk/1600,0,1)
	colorBG = np.clip((1-colorBG)*1.5,0,1)
	self.currentDisplayData['beatBuffer'] = np.roll(self.currentDisplayData['beatBuffer'],1,0)
	self.currentDisplayData['beatBuffer'][0] = colorBG


	dsa = signal.resample(self.currentDisplayData['specArray'],nrows,axis=1)
	# printArray(dsa, False)

	#background image
	screen.blit(self.currentDisplayData['outrunbg'], (0,0))

	#draw beat here
	for idx in range(len(self.currentDisplayData['sunPos'])):
		colorRef = self.currentDisplayData['beatBuffer'][idx]
		colorBG = 0.6*(1.0-colorRef)+0.4
		colorBG = colorsys.hsv_to_rgb(0.83, 1, colorBG)
		colorBG = (colorBG[0]*255,colorBG[1]*255,colorBG[2]*255)
		linePts = self.currentDisplayData['sunPos'][idx]
		pygame.draw.line(self.screen, colorBG, linePts[0], linePts[1], 7-idx)


	#foregorund image
	screen.blit(self.currentDisplayData['outrunfg'], (0,58))

	#draw grid here
	#draw FFT here.
	w = self.WIDTH
	w_delta = 100
	w_step = np.ceil(w/ncols)
	y_zero = 215
	y_step = 20
	y = y_zero
	h = self.HEIGHT - y
	y_delta = 1.2

	lastRow = []
	x = 0
	x_0 = 0
	for col in range(ncols):
		x = x_0 + w_step*col
		lastRow.append((int(x),int(y_zero)))
	for row in range(nrows):
		#generate row:
		rowTuples = []

		w_step = (w+(row+1)*w_delta)/ncols
		x_0 = -w_delta/2 *(row+1)

		for col in range(ncols):
			x = x_0 + w_step*col
			rowTuples.append((int(x),int(y-dsa[col,row])))
		y = y + y_step
		y_step = np.ceil(y_step*y_delta)

		for col in range(0,ncols-1,2):
			if col != 0:
				pygame.draw.polygon(self.screen,(0,0,0),[lastRow[col-1], rowTuples[col-1],rowTuples[col],rowTuples[col+1],lastRow[col+1],lastRow[col]])
				# pygame.draw.polygon(self.screen,color,[lastRow[col-1], rowTuples[col-1],rowTuples[col],rowTuples[col+1],lastRow[col+1],lastRow[col]],1)
			if (col % 2) == 0:
				pygame.draw.line(self.screen, color, lastRow[col], rowTuples[col], thickness)
		pygame.draw.lines(self.screen,color,False,np.array(lastRow).astype(np.int64),thickness)
		lastRow = rowTuples

def outRunZoom(self, data, screen):
	nrows = 9 #number of rows to display
	ncols = 50 #number of columns to display
	nbuf = 20 #number of rows to buffer
	color = (85,255,255)
	sunColor = (85,255,255)
	zoomMax = 102
	zoomMin = 2
	if self.thin:
		thickness = 1
	else:
		thickness = 2
	if not 'outrun_flag' in self.currentDisplayData:
		#Setup basic flags
		self.currentDisplayData['outrun_flag'] = True
		self.currentDisplayData['noBackground'] = True
		self.currentDisplayData['sunPos'] = [[(180,205),(500,205)],[(180,190),(500,190)],[(180,175),(500,175)],[(180,160),(461,160)],[(178,145),(460,145)],[(182,130),(457,130)],[(185,115),(453,115)]]
		picture = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),"outrunbg.png"))
		picture = pygame.transform.scale(picture, self.size)
		self.currentDisplayData['outrunbg'] = picture
		# self.currentDisplayData['extendChangeTime'] = random.randrange(5,10) #for some reason, simply extending the change time didn't work...
		if self.thin:
			imagefg = "outrunfg.png"
		else:
			imagefg = "outrunfg_thick.png"
		picture = pygame.image.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),imagefg))

		for n, x in enumerate(self.currentDisplayData['sunPos']):
			self.currentDisplayData['sunPos'][n][0] = (x[0][0]*self.WIDTH/picture.get_width(),x[0][1])
			self.currentDisplayData['sunPos'][n][1] = (x[1][0]*self.WIDTH/picture.get_width(),x[1][1])

		picture = pygame.transform.scale(picture, (self.WIDTH,picture.get_height()))
		self.currentDisplayData['outrunfg'] = picture
		self.currentDisplayData['beatBuffer'] = np.zeros(len(self.currentDisplayData['sunPos']))

	#compute grid
	if not 'outrun_buffer' in self.currentDisplayData: #compute spectrogram buffer.
		self.currentDisplayData['specArray'] = np.zeros((ncols,nbuf))
		self.currentDisplayData['outrun_buffer'] = True
		self.currentDisplayData['minArray'] = np.full(ncols, np.inf)
		self.currentDisplayData['outrun_lastframe'] = np.zeros((zoomMax-zoomMin))

		for x in range(min(self.fftArray.shape[1],nbuf)):
			# sampleData = self.fftArray[:,-(x+1)]
			sampleData = (self.fftArray[zoomMin:zoomMax,-(x+1)]/170)**4-30
			sampleData = np.clip(sampleData,0,None)
			sampleData = [peakDecay(sampleData[x],self.currentDisplayData['outrun_lastframe'][x],0.05,self.CHUNK,self.RATE) for x in range(len(sampleData))]
			self.currentDisplayData['outrun_lastframe'] = sampleData
			sampleData = np.concatenate((np.flip(sampleData),sampleData))
			# sampleData = np.concatenate((np.flip(self.fftArray[0:300,-(x+1)]), self.fftArray[0:300,-(x+1)]))*1.1
			self.currentDisplayData['specArray'][:,x] = signal.resample(sampleData, ncols, axis = 0)**2 #output buffer is filled with columns (axis 2) containing FFTs from most recent to least recent. Most recent has index 0, least recent has index N
			denom = 300000 #np.amax(self.currentDisplayData['specArray'][:,x])
			if denom == 0:
				denom = 1
			self.currentDisplayData['specArray'][:,x] = self.currentDisplayData['specArray'][:,x] / denom * 40
			self.currentDisplayData['minArray'] = np.minimum(self.currentDisplayData['specArray'][:,x], self.currentDisplayData['minArray'])
			self.currentDisplayData['specArray'][:,x] = self.currentDisplayData['specArray'][:,x] - self.currentDisplayData['minArray']#self.currentDisplayData['minArray']
			#note to self, try squaring data...
	else: #update spectrogram buffer.
		self.currentDisplayData['specArray'] = np.roll(self.currentDisplayData['specArray'],1,1) #roll array right by one.
		# sampleData = self.fftArray[:,-1]
		sampleData = (self.fftArray[zoomMin:zoomMax,-1]/170)**4-30
		sampleData = np.clip(sampleData,0,None)
		sampleData = [peakDecay(sampleData[x],self.currentDisplayData['outrun_lastframe'][x],0.05,self.CHUNK,self.RATE) for x in range(len(sampleData))]
		self.currentDisplayData['outrun_lastframe'] = sampleData
		sampleData = np.concatenate((np.flip(sampleData),sampleData))
		# sampleData = np.concatenate((np.flip(self.fftArray[0:300,-1]), self.fftArray[0:300,-1]))*1.1
		self.currentDisplayData['specArray'][:,0] = signal.resample(sampleData, ncols, axis = 0)**2 #overwrite the newest element with the newest one from the fft array
		denom = 300000 #np.amax(self.currentDisplayData['specArray'][:,0])
		if denom == 0:
			denom = 1
		self.currentDisplayData['specArray'][:,0] = self.currentDisplayData['specArray'][:,0] / denom*40
		self.currentDisplayData['minArray'] = np.minimum(self.currentDisplayData['specArray'][:,0], self.currentDisplayData['minArray'])
		self.currentDisplayData['specArray'][:,0] = self.currentDisplayData['specArray'][:,0] - self.currentDisplayData['minArray']#self.currentDisplayData['minArray']

	#compute sun lines
	colorBG = np.clip(self.signal_pk/1600,0,1)
	colorBG = np.clip((1-colorBG)*1.5,0,1)
	self.currentDisplayData['beatBuffer'] = np.roll(self.currentDisplayData['beatBuffer'],1,0)
	self.currentDisplayData['beatBuffer'][0] = colorBG


	dsa = signal.resample(self.currentDisplayData['specArray'],nrows,axis=1)
	# printArray(dsa, False)

	#background image
	screen.blit(self.currentDisplayData['outrunbg'], (0,0))

	#draw beat here
	for idx in range(len(self.currentDisplayData['sunPos'])):
		colorRef = self.currentDisplayData['beatBuffer'][idx]
		colorBG = 0.6*(1.0-colorRef)+0.4
		colorBG = colorsys.hsv_to_rgb(0.83, 1, colorBG)
		colorBG = (colorBG[0]*255,colorBG[1]*255,colorBG[2]*255)
		linePts = self.currentDisplayData['sunPos'][idx]
		pygame.draw.line(self.screen, colorBG, linePts[0], linePts[1], 7-idx)


	#foregorund image
	screen.blit(self.currentDisplayData['outrunfg'], (0,58))

	#draw grid here
	#draw FFT here.
	w = self.WIDTH
	w_delta = 100
	w_step = np.ceil(w/ncols)
	y_zero = 215
	y_step = 20
	y = y_zero
	h = self.HEIGHT - y
	y_delta = 1.2

	lastRow = []
	x = 0
	x_0 = 0
	for col in range(ncols):
		x = x_0 + w_step*col
		lastRow.append((int(x),int(y_zero)))
	for row in range(nrows):
		#generate row:
		rowTuples = []

		w_step = (w+(row+1)*w_delta)/ncols
		x_0 = -w_delta/2 *(row+1)

		for col in range(ncols):
			x = x_0 + w_step*col
			rowTuples.append((int(x),int(y-dsa[col,row])))
		y = y + y_step
		y_step = np.ceil(y_step*y_delta)

		for col in range(0,ncols-1,2):
			if col != 0:
				pygame.draw.polygon(self.screen,(0,0,0),[lastRow[col-1], rowTuples[col-1],rowTuples[col],rowTuples[col+1],lastRow[col+1],lastRow[col]])
				# pygame.draw.polygon(self.screen,color,[lastRow[col-1], rowTuples[col-1],rowTuples[col],rowTuples[col+1],lastRow[col+1],lastRow[col]],1)
			if (col % 2) == 0:
				pygame.draw.line(self.screen, color, lastRow[col], rowTuples[col], thickness)
		pygame.draw.lines(self.screen,color,False,np.array(lastRow).astype(np.int64),thickness)
		lastRow = rowTuples

# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
def main():
    parser = argparse.ArgumentParser(description='Raspberry Pi Video Music. Inspired by the Atari Video Music system.')
    parser.add_argument('-l', action="store_true", help="list audio and video interfaces")
    parser.add_argument('-v', action="store", type=int, default = 0, help="Specify video interface (int)")
    parser.add_argument('-a', action="store", type=int, default = 0, help="Specify audio interface (int)")
    parser.add_argument('-t', action="store_true", help="Use thinner lines for high res screens")
    parser.add_argument('-w', action="store_false", help="windowed mode")
    parser.add_argument('-verbose', action="store_true", help="talk more")
    args = parser.parse_args()
    bgs = [drops,spectrogram,background,peakDiamonds,meanDiamonds]
    fgs = [outRun, fftZoom, fftQuad, barchartLogspace, bargraphHill,timeSignal, fftSignal, fftHill, barchart] #outRunZoom
    PVM = piVideoMusic(fgs,bgs,listFlag=args.l,videoInterface=args.v,audioInterface=args.a,fullscreen=args.w, verbose=args.verbose, thin=args.t) #
    while PVM.running:
    	PVM.processEvent()

if __name__=="__main__":
    # call the main function
    # main([timeSignal,meanFreq,meanDiamonds,timeBackground,fftBackground,peakDiamonds,barchart,spectrogram])

	main()
