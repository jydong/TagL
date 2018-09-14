# Jingyan Dong
# CS251
# display.py
# 2/23/17

import Tkinter as tk
import tkFont as tkf
import math
import random
import view
import data
import analysis
import numpy as np
import tkFileDialog
import os
import scipy.stats as stats
import subprocess
import types
import colorsys



# create a class to build and manage the display
class DisplayApp:

	def __init__(self, width, height):

		# create a tk object, which is the root window
		self.root = tk.Tk()

		# width and height of the window
		self.initDx = width
		self.initDy = height

		# set up the geometry for the window
		self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

		# set the title of the window
		self.root.title("Take a Good Look")

		# set the maximum size of the window for resizing
		self.root.maxsize( 1600, 900 )

		# setup the menus
		self.buildMenus()

		# build the controls
		self.buildControls()

		# build the Canvas
		self.buildCanvas()

		# bring the window to the front
		self.root.lift()

		# - do idle events here to get actual canvas size
		self.root.update_idletasks()

		# now we can ask the size of the canvas
		print self.canvas.winfo_geometry()

		# set up the key bindings
		self.setBindings()

		# set up the application state
		self.objects = [] # list of data objects that will be drawn in the canvas
		self.data = None # will hold the raw data someday.
		self.baseClick = None # used to keep track of mouse movement
		self.baseClick1 = None
		self.baseClick2 = None
		self.baseClick3 = None

		self.view = view.View()

		self.axes = np.matrix([[0,0,0,1],
							   [1,0,0,1],
							   [0,0,0,1],
							   [0,1,0,1],
							   [0,0,0,1],
							   [0,0,1,1]])

		self.x_axis = None
		self.y_axis = None
		self.z_axis = None
		self.lines = [self.x_axis, self.y_axis, self.z_axis]

		self.f = 1.0
		self.current = [None, None]

		self.buildAxes()

		self.RLs = []   # regression lines
		self.RLpts = None  # regression lines' endpoints
		self.RL_labels = [] # regression line labels
		self.pca_labels = [] #pca analysis labels
		self.cluster_labels = []  # cluster analysis labels

		self.analysis_history = {}
		self.pca_history = {}
		self.cluster_history = {}

		self.clusterData = None #store current cluster data

		self.header_source = ["Original Data","Original Data","Original Data"]
		self.header_raw_index = [None, None,None]




	def buildAxes(self):
		vtm = self.view.build()
		pts = (vtm * self.axes.T).T

		self.x_axis = self.canvas.create_line(pts[0,0], pts[0,1], pts[1,0], pts[1,1], fill="firebrick")
		self.y_axis = self.canvas.create_line(pts[2,0], pts[2,1], pts[3,0], pts[3,1], fill="midnight blue")
		self.z_axis = self.canvas.create_line(pts[4,0], pts[4,1], pts[5,0], pts[5,1], fill="dark olive green")

		self.x_label = self.canvas.create_text(self.canvas.coords(self.x_axis)[2], self.canvas.coords(self.x_axis)[3],
											   text = "X", fill="firebrick")
		self.y_label = self.canvas.create_text(self.canvas.coords(self.y_axis)[2], self.canvas.coords(self.y_axis)[3],
											   text = "Y", fill="midnight blue")
		self.z_label = self.canvas.create_text(self.canvas.coords(self.z_axis)[2], self.canvas.coords(self.z_axis)[3],
											   text = "Z", fill="dark olive green")



	def updateAxes(self):
		vtm = self.view.build()
		pts = (vtm * self.axes.T).T

		self.canvas.coords(self.x_axis, pts[0,0], pts[0,1], pts[1,0], pts[1,1])
		self.canvas.coords(self.y_axis, pts[2,0], pts[2,1], pts[3,0], pts[3,1])
		self.canvas.coords(self.z_axis, pts[4,0], pts[4,1], pts[5,0], pts[5,1])

		# extension 3 - update axes labels
		self.canvas.coords(self.x_label, self.canvas.coords(self.x_axis)[2], self.canvas.coords(self.x_axis)[3])
		self.canvas.coords(self.y_label, self.canvas.coords(self.y_axis)[2], self.canvas.coords(self.y_axis)[3])
		self.canvas.coords(self.z_label, self.canvas.coords(self.z_axis)[2], self.canvas.coords(self.z_axis)[3])


	def updatePoints(self):
		if self.objects == []:
			return

		vtm = self.view.build()
		pts = (vtm * self.data_to_plot.T).T

		for i in range(len(self.objects)):
			x = pts[i, 0]
			y = pts[i, 1]

			if self.ptSize == 4:
				size = 4
			else:
				size = 2 + 6* self.size_dim[i,0]

			if self.ptShape == "circle":
				self.canvas.coords(self.objects[i], x - size, y - size, x + size, y +  size)

			elif self.ptShape == "square":
				self.canvas.coords(self.objects[i], x -size, y -  size, x + size, y +size)

			elif self.ptShape == "triangle":
				self.canvas.coords(self.objects[i], x, y -size, x -size, y +size, x +size, y + size)

	def updateFits(self):
		if self.RLs == []:
			return

		vtm = self.view.build()
		pts = (vtm * self.RLpts.T).T

		for i in range(len(self.RLs)):
			self.canvas.coords(self.RLs[i], pts[i*2,0], pts[i*2,1], pts[i*2+1,0], pts[i*2+1,1])



	def buildMenus(self):

		# create a new menu
		menu = tk.Menu(self.root)

		# set the root menu to our new menu
		self.root.config(menu = menu)

		# create a variable to hold the individual menus
		menulist = []

		# create a file menu
		filemenu = tk.Menu( menu )
		menu.add_cascade( label = "File", menu = filemenu )
		menulist.append(filemenu)

		# create another menu for kicks
		cmdmenu = tk.Menu( menu )
		menu.add_cascade( label = "Command", menu = cmdmenu )
		menulist.append(cmdmenu)

		# menu text for the elements
		# the first sublist is the set of items for the file menu
		# the second sublist is the set of items for the option menu
		menutext = [ [ '-', '-', 'Quit \xE2\x8C\x98-Q','Clear \xE2\x8C\x98-N'],
					 [ 'Command 1', '-', '-', 'LinearRegression']]

		# menu callback functions (note that some are left blank,
		# so that you can add functions there if you want).
		# the first sublist is the set of callback functions for the file menu
		# the second sublist is the set of callback functions for the option menu
		menucmd = [ [None, None, self.handleQuit, self.clearData],
					[self.handleMenuCmd1, None, None, self.handleLinearRegression]]

		# build the menu elements and callbacks
		for i in range( len( menulist ) ):
			for j in range( len( menutext[i]) ):
				if menutext[i][j] != '-':
					menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
				else:
					menulist[i].add_separator()

	# create the canvas object
	def buildCanvas(self):
		self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy )
		self.canvas.pack( expand=tk.YES, fill=tk.BOTH )

		return

	# build a frame and put controls in it
	def buildControls(self):
		### Control ###
		# make a control frame on the right
		rightcntlframe = tk.Frame(self.root)
		rightcntlframe.pack(side=tk.RIGHT, padx=2, pady=2, fill=tk.Y)

		# make a separator frame
		sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
		sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

		# use a label to set the size of the right panel
		label = tk.Label( rightcntlframe, text="Control Panel", width=20 )
		label.pack( side=tk.TOP, pady=10 )


		# extension 2 - controls panning speed
		label = tk.Label( rightcntlframe, text="Panning Speed ", width=20 )
		label.pack(side=tk.TOP)
		self.panningOption = tk.StringVar( self.root )
		self.panningOption.set("1.0") #default
		panning = tk.OptionMenu( rightcntlframe, self.panningOption, "1.0", "0.5", "2.0") # can add a command to the menu
		panning.pack(side=tk.TOP)

		# extension 2 - controls scaling speed
		label = tk.Label( rightcntlframe, text="Scaling Speed ", width=20 )
		label.pack(side=tk.TOP)
		self.scalingOption = tk.StringVar( self.root )
		self.scalingOption.set("1.0") #default
		scaling = tk.OptionMenu( rightcntlframe, self.scalingOption, "1.0", "0.5", "2.0") # can add a command to the menu
		scaling.pack(side=tk.TOP)

		# extension 2 - controls rotation speed
		label = tk.Label( rightcntlframe, text="Rotation Speed ", width=20 )
		label.pack(side=tk.TOP)
		self.rotationOption = tk.StringVar( self.root )
		self.rotationOption.set("1.0") #default
		rotation = tk.OptionMenu( rightcntlframe, self.rotationOption, "1.0", "0.5", "2.0") # can add a command to the menu
		rotation.pack(side=tk.TOP)

		button1 = tk.Button( rightcntlframe, text="File", command=self.handleOpen )
		button1.pack(side=tk.TOP)  # default side is top

		self.file= tk.StringVar(self.root)
		location = tk.Label(rightcntlframe, textvariable=self.file, width=20)
		location.pack(side=tk.TOP)
		self.file.set("No File Selected")

		button2 = tk.Button( rightcntlframe, text="Plot Data", command=self.handlePlotData )
		button2.pack(side=tk.TOP)  # default side is top

		button3 = tk.Button( rightcntlframe, text="Simple Linear Regression", command=self.handleLinearRegression )
		button3.pack(side=tk.TOP)  # default side is top

		button6 = tk.Button(rightcntlframe, text="Multiple Linear Regression", command=self.handleMLR)
		button6.pack(side=tk.TOP)  # default side is top

		#  extension 8 - create set plane button
		self.planeOption = tk.StringVar( self.root )
		self.planeOption.set("XY") #default
		scaling = tk.OptionMenu( rightcntlframe, self.planeOption, "XY", "YZ", "XZ") # can add a command to the menu
		scaling.pack(side=tk.TOP)

		button4 = tk.Button(rightcntlframe, text="Set Plane", command=self.setPlane)
		button4.pack(side=tk.TOP)  # default side is top


		# extension 7- reset button
		button5 = tk.Button(rightcntlframe, text="Reset", command=self.reset)
		button5.pack(side=tk.TOP)  # default side is top

		# make a bottom frame
		bottomframe = tk.Frame(self.root)
		bottomframe.pack(side=tk.BOTTOM, padx=2, pady=2, fill=tk.Y)

		# make a separator frame
		sep = tk.Frame( self.root, height=2, width=self.initDx, bd=1, relief=tk.SUNKEN )
		sep.pack( side=tk.BOTTOM, padx = 2, pady = 2, fill=tk.X)

		self.location= tk.StringVar(self.root)
		location = tk.Label(bottomframe, textvariable=self.location, width=60)
		location.pack(side=tk.LEFT)
		self.location.set("None: None,	None: None,	 None: None")

		self.scaling= tk.StringVar(self.root)
		location = tk.Label(bottomframe, textvariable=self.scaling, width=10)
		location.pack(side=tk.LEFT)
		self.scaling.set("Scale: 100%")

		self.angle= tk.StringVar(self.root)
		location = tk.Label(bottomframe, textvariable=self.angle, width=30)
		location.pack(side=tk.LEFT)
		self.angle.set("Rotation Angle: X-Axis 0, Y-Axis 0")

		button6 = tk.Button(rightcntlframe, text="Save Image", command=self.save_image)
		button6.pack(side=tk.TOP)  # default side is top

		button7 = tk.Button(rightcntlframe, text="Save Regression Analysis", command=self.save_analysis)
		button7.pack(side=tk.TOP)  # default side is top

		button8 = tk.Button(rightcntlframe, text="Replot Regression Analysis", command=self.replot_analysis)
		button8.pack(side=tk.TOP)  # default side is top

		button9 = tk.Button(rightcntlframe, text="Perform PCA", command=self.handle_pca_analysis)
		button9.pack(side=tk.TOP)  # default side is top

		button10 = tk.Button(rightcntlframe, text="PCA History", command=self.pca_analysis_history)
		button10.pack(side=tk.TOP)  # default side is top

		label = tk.Label(rightcntlframe, text="Saved Clustering Analysis: ", width=25)
		label.pack(side=tk.TOP, pady=5)
		self.clusterOption = tk.StringVar(self.root)
		self.clusterOption.set("None")
		self.clusterMenu = tk.OptionMenu(rightcntlframe, self.clusterOption,"None")  # can add a command to the menu
		self.clusterMenu.pack(side=tk.TOP)

		button11 = tk.Button(rightcntlframe, text="Perform Clustering", command=self.handle_clustering)
		button11.pack(side=tk.TOP)  # default side is top

		button12 = tk.Button(rightcntlframe, text="Plot Clusters", command=self.plot_clusters)
		button12.pack(side=tk.TOP)  # default side is top


	def setBindings(self):
		# bind mouse motions to the canvas
		# button2(rotation): control + button1
		# button3(scaling): option + button1
		self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
		self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
		self.canvas.bind( '<Button-2>', self.handleMouseButton2 )
		self.canvas.bind( '<Control-Command-Button-1>', self.handleMouseButton3)
		self.canvas.bind( '<Option-Button-1>', self.handleMouseButton3)
		self.canvas.bind( '<Button-3>', self.handleMouseButton3)
		self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
		self.canvas.bind( '<B2-Motion>', self.handleMouseButton2Motion )
		self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )
		self.canvas.bind( '<B3-Motion>', self.handleMouseButton3Motion)
		self.canvas.bind( '<Option-B1-Motion>', self.handleMouseButton3Motion)

		# bind command sequences to the root window
		self.root.bind( '<Command-q>', self.handleQuit )
		self.root.bind( '<Command-n>', self.clearData )
		self.root.bind( '<Command-o>', self.handleOpen )

		# bind mouse motion with handleMotion method
		self.canvas.bind( '<Motion>',self.handleMotion)
		self.canvas.bind('<Configure>', self.handleResize)



	def handleQuit(self, event=None):
		print 'Terminating'
		self.root.destroy()



	def handleMenuCmd1(self):
		print 'handling menu command 1'


	def handleMouseButton1(self, event):
		self.baseClick1 = (event.x, event.y)


	def handleMouseButton2(self, event):
		self.baseClick2 = (event.x, event.y)
		self.original_view = self.view.clone()



	def handleMouseButton3(self, event):
		self.baseClick3 = (event.x, event.y)
		self.baseExtent = self.view.extent[:]


	# This is called if the first mouse button is being moved
	def handleMouseButton1Motion(self, event):
		# calculate the difference
		diff = (event.x - self.baseClick1[0], event.y - self.baseClick1[1])

		# divide by the screen size and multiply by the extents
		delta0 = float(diff[0]) / self.view.screen[0] * self.view.extent[0]
		delta1 = float(diff[1]) / self.view.screen[1] * self.view.extent[1]

		# apply panning speed
		delta0 = delta0 * float(self.panningOption.get())
		delta1 = delta1 * float(self.panningOption.get())

		self.view.vrp += delta0 * self.view.u + delta1* self.view.vup

		self.baseClick1 = (event.x, event.y)

		self.updateAxes()
		self.updatePoints()
		self.updateFits()

	# This is called if the second button of a real mouse has been pressed
	# and the mouse is moving. Or if the control key is held down while
	# a person moves their finger on the track pad.
	def handleMouseButton2Motion(self, event):
		w = min(self.view.screen)

		delta0 = (event.x- self.baseClick2[0]) / (0.5*w) * math.pi
		delta1 = -(event.y - self.baseClick2[1]) / (0.5*w) * math.pi

		# apply rotation speed
		delta0 = delta0 * float(self.rotationOption.get())
		delta1 = delta1 * float(self.rotationOption.get())

		self.view = self.original_view.clone()
		self.view.rotateVRC(-delta0, delta1)
		self.updateAxes()
		self.updatePoints()
		self.updateFits()

		# update rotation angles shown in the bottom frame (keep two decimal points)
		self.angle.set("Rotation Angle: X-Axis " + "%.2f" % delta1 + ", Y-Axis " + "%.2f" % delta0)



	# This is called if the third mouse button is being moved
	def handleMouseButton3Motion(self, event):
		# calculate the difference
		diff = event.y - self.baseClick3[1]

		# apply scaling speed
		diff = diff * float(self.scalingOption.get())

		# calculate the scale factor
		k = 1.0 / self.canvas.winfo_height()
		self.f = 1.0 + k * diff		 #store the factor as a field for plane switching
		self.f = max(min(self.f, 3.0), 0.1)


		self.view.extent = [self.baseExtent[0] * self.f,self.baseExtent[1] * self.f,
							self.baseExtent[2] * self.f]
		self.updateAxes()
		self.updatePoints()
		self.updateFits()

		# update scaling status in the bottom frame
		self.scaling.set("Scale: " + str(int(1.0/self.view.extent[0] * 100)) + "%")


	# display raw data in the bottom frame when the mouse is hovering over a data point
	def handleMotion(self,event):
		if self.objects == []:
			return

		# change the color back if the mouse is not longer on the point
		if self.current[0] != None:
			if self.canvas.coords(self.current[0])[0] > event.x or event.x >self.canvas.coords(self.current[0])[2]:
				self.canvas.itemconfig(self.current[0], fill = self.current[1])
			if self.canvas.coords(self.current[0])[1] > event.y or event.y >self.canvas.coords(self.current[0])[3]:
				self.canvas.itemconfig(self.current[0], fill = self.current[1])

		for i in range(len(self.objects)):
			obj = self.objects[i]
			if self.canvas.coords(obj)[0] <= event.x <= self.canvas.coords(obj)[2]:
				if self.canvas.coords(obj)[1] <= event.y <= self.canvas.coords(obj)[3]:
					if self.header_source[0] == "Original Data":
						raw_x = self.data.matrix_data[i, self.header_raw_index[0]]
					else:
						raw_x = self.pca.matrix_data[i, self.header_raw_index[0]]

					if self.header_source[1] == "Original Data":
						raw_y = self.data.matrix_data[i, self.header_raw_index[1]]
					else:
						raw_y = self.pca.matrix_data[i, self.header_raw_index[1]]


					if self.axisHeader[2] != None:
						if self.header_source[2] == "Original Data":
							raw_z = self.data.matrix_data[i, self.header_raw_index[2]]
						else:
							raw_z = self.pca.matrix_data[i, self.header_raw_index[2]]
					else:
						raw_z = "None"

					# display raw data information of the point
					self.location.set(self.axisHeader[0]+ ": " +str(raw_x)+ ", "
					+ self.axisHeader[1]+ ": " +str(raw_y)+ ", " +
					str(self.axisHeader[2])+ ": " +str(raw_z))

					# store original color
					if self.current[0] != obj:
						if self.current[0] != None:
							self.canvas.itemconfig(self.current[0], fill = self.current[1])
						self.current[0] = obj
						self.current[1] = self.canvas.itemcget(obj, "fill")

					# highlight the point
					self.canvas.itemconfig(obj, fill = "red")



	def handleOpen(self,event= None):
  		fn = tkFileDialog.askopenfilename( parent=self.root,title='Choose a data file', initialdir='.' )
  		# fn = "AustraliaCoast.csv"

		self.fn = fn   # for saving the current analysis
		self.data = data.Data(fn)
		self.file.set(os.path.basename(fn))

		self.resetForNewFile()

	def handlePlotData(self, event=None):
		if self.data is None:
			print "No file selected"
			return
		self.headers = self.data.matrix_headers
		self.buildPoints(self.handleChooseAxes())

	def handleChooseAxes(self):
		d = PlotDataDialog(self.root, self.headers)
		selected = d.result[:]

		for i in range(len(d.result)):
			if d.result[i] == None:
				if PlotDataDialog.selected[i] != None:
					selected[i] = PlotDataDialog.selected[i]

		self.headerSelected = selected[:-1]
		self.ptShape = selected[-1]

		PlotDataDialog.selected = selected[:]


		for i in range(len(self.headerSelected)):
			if self.headerSelected[i] == "None":
				self.headerSelected[i] = None

		self.axisHeader =  self.headerSelected[:3]

		print "selected headers: ", self.headerSelected
		print "selected shape: ", self.ptShape
		return self.headerSelected

	def buildPoints(self, headers):
		self.matrix_data = self.data.matrix_data #store numeric data
		self.normalized_data = analysis.normalize_columns_separately(self.data)

		if headers[0] != None :
			if headers[1] != None:
				self.clearCanvas()

				x_index = self.data.header2matrix[headers[0]]
				y_index = self.data.header2matrix[headers[1]]
				x_col = self.normalized_data[:,x_index]
				y_col = self.normalized_data[:,y_index]
				self.header_raw_index[0] = x_index
				self.header_raw_index[1] = y_index

				if headers[2] != None:
					z_index = self.data.header2matrix[headers[2]]
					z_col = self.normalized_data[:,z_index]
					self.header_raw_index[2] = z_index

					self.data_to_plot = self.normalized_data[:, [x_index,y_index,z_index]]

				else:
					z_col = np.zeros((len(self.normalized_data), 1))

					self.data_to_plot = self.normalized_data[:, [x_index,y_index]]
					self.data_to_plot = np.hstack((self.data_to_plot, z_col))

				homogeneous = np.ones((len(self.normalized_data), 1))
				self.data_to_plot = np.hstack((self.data_to_plot, homogeneous))


				if headers[3] == None:
					self.ptColor = "black"
				else:
					self.ptColor = "dimension"
					index = self.data.header2matrix[headers[3]]
					self.color_dim = self.normalized_data[:,index]

				if headers[4] == None:
					self.ptSize = 4
				else:
					self.ptSize = "dimension"
					index = self.data.header2matrix[headers[4]]
					self.size_dim = self.normalized_data[:,index]

				if self.ptShape == None:
					self.ptShape = "circle"

				# place the data
				vtm = self.view.build()
				pts = (vtm * self.data_to_plot.T).T

				for i in range(len(self.data_to_plot)):
					x = pts[i,0]
					y = pts[i,1]

					if self.ptColor == "black":
						color = "black"
					else:
						alpha = self.color_dim[i,0]
						alpha = round(alpha,2)
						color = ((1-alpha)*255, (1-alpha)*255, alpha*255)
						color = '#%02x%02x%02x' % color

					if self.ptSize == 4:
						size = 4
					else:
						size = 2 + 6* self.size_dim[i,0]

					if self.ptShape == "circle":
						pt = self.canvas.create_oval( x-size, y-size,x+size, y+size, fill=color, outline='')
						self.objects.append(pt)
					elif self.ptShape == "square":
						pt = self.canvas.create_rectangle( x-size, y-size, x+size, y+size,fill=color, outline='' )
						self.objects.append(pt)
					elif self.ptShape == "triangle":
						pt = self.canvas.create_polygon( x, y-size, x-size, y+size, x+size, y+size,	 fill=color, outline='')
						self.objects.append(pt)

				self.canvas.itemconfig(self.x_label, text="X: " + str(self.axisHeader[0]))
				self.canvas.itemconfig(self.y_label, text="Y: " + str(self.axisHeader[1]))
				self.canvas.itemconfig(self.z_label, text="Z: " + str(self.axisHeader[2]))

	def handleLinearRegression(self, event=None):
		if self.data is None:
			print "No file selected"
			return

		self.headers = self.data.matrix_headers
		d = LRDialog(self.root, self.headers).result
		if d == None:
			return

		selected = d[:]

		for i in range(len(d)):
			if d[i] == None:
				if LRDialog.selected[i] != None:
					selected[i] = LRDialog.selected[i]
				else:
					if i == 0:
						selected[i] = self.headers[0]
					elif i == 1:
						selected[i] = self.headers[1]

		self.headerSelected = selected[:-1]
		self.ptShape = selected[-1]
		LRDialog.selected = selected[:]

		for i in range(len(self.headerSelected)):
			if self.headerSelected[i] == "None":
				self.headerSelected[i] = None

		self.axisHeader = self.headerSelected[:2]
		self.axisHeader.append(None)

		print "axis headers: ", self.axisHeader
		print "selected headers: ", selected[:-1]
		print "selected shape: ", self.ptShape

		self.clearData()
		self.clearCanvas()
		self.buildLinearRegression(selected)

	def buildLinearRegression(self, headers):
		self.matrix_data = self.data.matrix_data # store raw numeric data
		self.normalized_data = analysis.normalize_columns_separately(self.data)

		x_index = self.data.header2matrix[headers[0]]
		y_index = self.data.header2matrix[headers[1]]
		x_col = self.normalized_data[:,x_index]
		y_col = self.normalized_data[:,y_index]
		self.header_raw_index[0] = x_index
		self.header_raw_index[1] = y_index


		z_col = np.zeros((len(self.normalized_data), 1))
		homogeneous = np.ones((len(self.normalized_data), 1))

		self.data_to_plot = self.normalized_data[:, [x_index,y_index]]
		self.data_to_plot = np.hstack((self.data_to_plot, z_col))
		self.data_to_plot = np.hstack((self.data_to_plot, homogeneous))


		if headers[2] == None:
			self.ptColor = "black"
		else:
			self.ptColor = "dimension"
			index = self.data.header2matrix[headers[2]]
			self.color_dim = self.normalized_data[:,index]

		if headers[3] == None:
			self.ptSize = 4
		else:
			self.ptSize = "dimension"
			index = self.data.header2matrix[headers[3]]
			self.size_dim = self.normalized_data[:,index]

		if self.ptShape == None:
			self.ptShape = "circle"

		# place the data
		vtm = self.view.build()
		pts = (vtm * self.data_to_plot.T).T

		for i in range(len(self.data_to_plot)):
			x = pts[i,0]
			y = pts[i,1]

			if self.ptColor == "black":
				color = "black"
			else:
				alpha = self.color_dim[i,0]
				alpha = round(alpha,2)
				color = ((1-alpha)*255, (1-alpha)*255, alpha*255)
				color = '#%02x%02x%02x'  % color

			if self.ptSize == 4:
				size = 4
			else:
				size = 2 + 6* self.size_dim[i,0]

			if self.ptShape == "circle":
				pt = self.canvas.create_oval( x-size, y-size,x+size, y+size, fill=color, outline='')
				self.objects.append(pt)
			elif self.ptShape == "square":
				pt = self.canvas.create_rectangle( x-size, y-size, x+size, y+size,fill=color, outline='' )
				self.objects.append(pt)
			elif self.ptShape == "triangle":
				pt = self.canvas.create_polygon( x, y-size, x-size, y+size, x+size, y+size,	 fill=color, outline='')
				self.objects.append(pt)

		self.canvas.itemconfig(self.x_label, text = "X: " + str(self.axisHeader[0]))
		self.canvas.itemconfig(self.y_label, text = "Y: " + str(self.axisHeader[1]))
		self.canvas.itemconfig(self.z_label, text = "Z: " + str(self.axisHeader[2]))


		# draw a regression line

		x = self.matrix_data[:,x_index].T.tolist()[0]
		y = self.matrix_data[:,y_index].T.tolist()[0]

		slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
		[xmin, xmax] = analysis.data_range(self.data, headers = headers[:2])[0]
		[ymin, ymax] = analysis.data_range(self.data, headers = headers[:2])[1]

		endpoint1 = [(xmin-xmin)/(xmax-xmin), ((xmin * slope + intercept) - ymin)/(ymax - ymin), 0, 1]
		endpoint2 = [(xmax-xmin)/(xmax-xmin), ((xmax * slope + intercept) - ymin)/(ymax - ymin), 0, 1]

		# normalize endpoints
		self.RLpts = np.matrix([endpoint1, endpoint2])
		pts = (vtm * self.RLpts.T).T

		linear_regression_line = self.canvas.create_line(pts[0,0], pts[0,1],pts[1,0], pts[1,1], fill="firebrick")
		self.RLs.append(linear_regression_line)

		# update labels on the canvas
		if self.RL_labels != []:
			self.RL_labels[0].config(text="Independent Varible(X): \n" + headers[0])
			self.RL_labels[1].config(text="Dependent Varible(Y): \n" + headers[1])
			self.RL_labels[2].config(text="Slope: " + str(round(slope,3)))
			self.RL_labels[3].config(text="Intercept: " + str(round(intercept,3)))
			self.RL_labels[4].config(text="R-value: " + str(round(r_value*r_value,3)))
		else:
			label1 = tk.Label(self.canvas, text="Independent Varible(X): \n" + headers[0], width=20)
			label1.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label2 = tk.Label(self.canvas, text="Dependent Varible(Y): \n" + headers[1], width=20)
			label2.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label3 = tk.Label( self.canvas, text="Slope: " + str(round(slope,3)) , width=20 )
			label3.pack( side=tk.TOP, anchor = tk.E, pady=5 )
			label4 = tk.Label( self.canvas, text="Intercept: " + str(round(intercept,3)) , width=20 )
			label4.pack( side=tk.TOP, anchor = tk.E, pady=5 )
			label5 = tk.Label( self.canvas, text="R-squared: " + str(round(r_value*r_value,3)) , width=20 )
			label5.pack( side=tk.TOP, anchor = tk.E, pady=5 )
			self.RL_labels = [label1, label2, label3,label4, label5]

	def handleMLR(self, event=None):
		if self.data is None:
			print "No file selected"
			return

		self.headers = self.data.matrix_headers
		d = MLRDialog(self.root, self.headers).result
		if d == None:
			return

		selected = d[:]

		for i in range(len(d)):
			if d[i] == None:
				if MLRDialog.selected[i] != None:
					selected[i] = MLRDialog.selected[i]
				else:
					if i == 0:
						selected[i] = [self.headers[0]]
					elif i == 1:
						selected[i] = self.headers[1]

		self.headerSelected = selected[:-1]

		self.ptShape = selected[-1]
		MLRDialog.selected = selected[:]

		for i in range(len(self.headerSelected)):
			if self.headerSelected[i] == "None":
				self.headerSelected[i] = None

		if len(self.headerSelected[0]) > 1:
			self.axisHeader = [self.headerSelected[0][0], self.headerSelected[1],self.headerSelected[0][1]]
		else:
			self.axisHeader = [self.headerSelected[0][0], self.headerSelected[1], None]

		print "axis headers: ", self.axisHeader
		print "selected headers: ", selected[:-1]
		print "selected shape: ", self.ptShape

		self.clearData()
		self.clearCanvas()
		self.reset()
		self.buildMLR(selected)

	def buildMLR(self, headers):
		self.matrix_data = self.data.matrix_data # store raw numeric data
		self.normalized_data = analysis.normalize_columns_separately(self.data)

		x_index = self.data.header2matrix[headers[0][0]]
		y_index = self.data.header2matrix[headers[1]]
		x_col = self.normalized_data[:,x_index]
		y_col = self.normalized_data[:,y_index]
		self.header_raw_index[0] = x_index
		self.header_raw_index[1] = y_index

		if len(headers[0]) > 1:
			z_index = self.data.header2matrix[headers[0][1]]
			z_col = self.normalized_data[:, z_index]
			self.header_raw_index[2] = z_index
		else:
			z_col = np.zeros((len(self.normalized_data), 1))
		homogeneous = np.ones((len(self.normalized_data), 1))

		self.data_to_plot = self.normalized_data[:, [x_index,y_index]]
		self.data_to_plot = np.hstack((self.data_to_plot, z_col))
		self.data_to_plot = np.hstack((self.data_to_plot, homogeneous))


		if headers[2] == None:
			self.ptColor = "black"
		else:
			self.ptColor = "dimension"
			index = self.data.header2matrix[headers[2]]
			self.color_dim = self.normalized_data[:,index]

		if headers[3] == None:
			self.ptSize = 4
		else:
			self.ptSize = "dimension"
			index = self.data.header2matrix[headers[3]]
			self.size_dim = self.normalized_data[:,index]

		if self.ptShape == None:
			self.ptShape = "circle"

		# place the data
		vtm = self.view.build()
		pts = (vtm * self.data_to_plot.T).T

		for i in range(len(self.data_to_plot)):
			x = pts[i,0]
			y = pts[i,1]

			if self.ptColor == "black":
				color = "black"
			else:
				alpha = self.color_dim[i,0]
				alpha = round(alpha,2)
				color = ((1-alpha)*255, (1-alpha)*255, alpha*255)
				color = '#%02x%02x%02x' % color

			if self.ptSize == 4:
				size = 4
			else:
				size = 2 + 6* self.size_dim[i,0]

			if self.ptShape == "circle":
				pt = self.canvas.create_oval( x-size, y-size,x+size, y+size, fill=color, outline='')
				self.objects.append(pt)
			elif self.ptShape == "square":
				pt = self.canvas.create_rectangle( x-size, y-size, x+size, y+size,fill=color, outline='' )
				self.objects.append(pt)
			elif self.ptShape == "triangle":
				pt = self.canvas.create_polygon( x, y-size, x-size, y+size, x+size, y+size,	 fill=color, outline='')
				self.objects.append(pt)

		self.canvas.itemconfig(self.x_label, text = "X: " + str(self.axisHeader[0]))
		self.canvas.itemconfig(self.y_label, text = "Y: " + str(self.axisHeader[1]))
		self.canvas.itemconfig(self.z_label, text = "Z: " + str(self.axisHeader[2]))


		# draw a regression line

		result = analysis.linear_regression(self.data, headers[0], headers[1])
		simple_result = analysis.simple_results(result)

		x = self.matrix_data[:,x_index].T.tolist()[0]
		y = self.matrix_data[:,y_index].T.tolist()[0]

		x_range = analysis.data_range(self.data, headers = headers[0])
		[xmin, xmax] = x_range[0]
		[ymin, ymax] = analysis.data_range(self.data, headers = headers[1:2])[0]
		xmins = []
		xmaxs = []
		for r in x_range:
			xmins.append(r[0])
			xmaxs.append(r[1])
		xmins.append(1)
		xmaxs.append(1)
		yhat_min = np.dot(np.matrix(xmins),result[0])
		yhat_max = np.dot(np.matrix(xmaxs),result[0])

		endpoint1 = [0, (yhat_min-ymin)/(ymax-ymin), 0, 1]
		if len(headers[0]) > 1:
			endpoint2 = [1, (yhat_max-ymin)/(ymax-ymin), 1, 1]
		else:
			endpoint2 = [1, (yhat_max-ymin)/(ymax-ymin), 0, 1]

		# normalize endpoints
		self.RLpts = np.matrix([endpoint1, endpoint2])
		pts = (vtm * self.RLpts.T).T

		linear_regression_line = self.canvas.create_line(pts[0,0], pts[0,1],pts[1,0], pts[1,1], fill="firebrick")
		self.RLs.append(linear_regression_line)

		# update labels on the canvas
		if self.RL_labels != []:
			self.RL_labels[0].config(text="Independent Varible(X): \n" + '\n'.join(headers[0]))
			self.RL_labels[1].config(text="Dependent Varible(Y): \n" + headers[1])
			self.RL_labels[2].config(text="Coefficients: \n" + "\n".join(map(str,simple_result[0])))
			self.RL_labels[3].config(text="SSE: " + str(result[1]))
			self.RL_labels[4].config(text="R2: " + str(result[2]))
			self.RL_labels[5].config(text="t: \n" + "\n".join(map(str,simple_result[3])))
			self.RL_labels[6].config(text="p: \n" + "\n".join(map(str,simple_result[4])))
		else:
			label1 = tk.Label(self.canvas, text="Independent Varible(X): \n" +'\n'.join(headers[0]), width=25)
			label1.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label2 = tk.Label(self.canvas, text="Dependent Varible(Y): \n" + headers[1], width=25)
			label2.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label3 = tk.Label( self.canvas, text="Coefficients: \n" + "\n".join(map(str,simple_result[0])) , width=25 )
			label3.pack( side=tk.TOP, anchor = tk.E, pady=5 )
			label4 = tk.Label( self.canvas, text="SSE: " + str(simple_result[1]) , width=25 )
			label4.pack( side=tk.TOP, anchor = tk.E, pady=5 )
			label5 = tk.Label( self.canvas, text="R2: " + str(simple_result[2]) , width=25 )
			label5.pack( side=tk.TOP, anchor = tk.E, pady=5 )
			label6 = tk.Label(self.canvas, text="t: \n" + "\n".join(map(str,simple_result[3])), width=25)
			label6.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label7 = tk.Label(self.canvas, text="p: \n" + "\n".join(map(str,simple_result[4])), width=25)
			label7.pack(side=tk.TOP, anchor=tk.E, pady=5)
			self.RL_labels = [label1, label2, label3, label4, label5, label6, label7]

	# save current canvas to a ps file
	def save_image(self):
		filename = SaveDialog(self.root, title='Save Image').result
		if filename == None:
			return

		if filename == "":
			filename = "image1.ps"
		else:
			filename += ".ps"
		self.canvas.postscript(file = filename,colormode="color")

		# process = subprocess.Popen(["ps2pdf",filename, "result.pdf"])
		# # os.remove("tmp.ps")

	# save linear regression analysis
	def save_analysis(self):
		if self.data is None:
			print "No file selected"
			return

		if self.headerSelected == []:
			return

		# key
		name = SaveDialog(self.root, title='Save Analysis').result
		if name == None:
			return
		if name == "":
			name = "analysis_1"

		# value:[filename, independent vars, dependent vars, color, size, shape]
		information = [self.fn, self.headerSelected[0],self.headerSelected[1],
					   self.headerSelected[2], self.headerSelected[3], self.ptShape]

		self.analysis_history[name] = information
		print "history: ", self.analysis_history

	# replot linear regression analysis
	def replot_analysis(self):
		name = AnalysisHistory(self.root, self.analysis_history).result
		if name == None:
			return
		information = self.analysis_history[name]
		fn = information[0]
		self.data = data.Data(fn)
		self.file.set(os.path.basename(fn))
		self.resetForNewFile()

		# simple linear regression
		if isinstance(information[1], types.StringTypes):
			LRDialog.selected = information[1:]
			print LRDialog.selected
			self.handleLinearRegression()
		# multiple linear regression
		else:
			MLRDialog.selected = information[1:]
			print MLRDialog.selected
			self.handleMLR()

	# enable the user to select headers for a PCA analysis
	def handle_pca_analysis(self):
		if self.data is None:
			print "No file selected"
			return

		self.headers = self.data.matrix_headers  #store original data headers
		self.pca_headers = PCA(self.root, self.headers).result # get pca headers
		print "selected PCA headers: ", self.pca_headers
		if self.pca_headers == None:
			return

		# enable the user to name a PCA analysis
		name = SaveDialog(self.root, title='Save PCA Analysis').result
		if name == None:
			return
		if name == "":
			name = "pca_analysis_1" #give a default name

		# store a pca analysis in a dictoinary
		# key: analysis name;
		# value: [data filename, a PCAData object]
		information = [self.fn, analysis.pca(self.data, self.pca_headers),self.headers]
		self.pca_history[name] = information
		print "pca analysis history: ", self.pca_history
		return name

	# plot pca data, takes in a list of data source and a list of headers
	def plot_pca(self,source,headers,cluster=0):
		# clear the canvas
		self.clearData()
		self.clearCanvas()
		self.reset()

		# automatically normalize data
		normalized_pdata = analysis.normalize_columns_separately(self.pca)
		normalized_odata = analysis.normalize_columns_separately(self.data)

		self.pca_headers = self.pca.get_data_headers()
		self.header_source = source[:3]

		if cluster == 0:
			self.ptColor = "dimension"
		else:
			self.ptColor = "cluster"
		self.ptSize = "dimension"
		self.ptShape = "circle"

		# generate data_to_plot matrix
		self.data_to_plot = np.empty([self.pca.get_num_rows(),0])
		for i in range(len(source[:2])):
			if source[i] == "PCA Data":
				col_ind = self.pca.evecs2index[headers[i]]
				self.data_to_plot = (np.hstack((self.data_to_plot, normalized_pdata[:,col_ind])))
			else:
				col_ind = self.data.header2matrix[headers[i]]
				self.data_to_plot = (np.hstack((self.data_to_plot, normalized_odata[:,col_ind])))
			self.header_raw_index[i] = col_ind

		if headers[2] != None:
			if source[2] == "PCA Data":
				col_ind = self.pca.evecs2index[headers[2]]
				self.data_to_plot = (np.hstack((self.data_to_plot, normalized_pdata[:, col_ind])))
			else:
				col_ind = self.data.header2matrix[headers[2]]
				self.data_to_plot = (np.hstack((self.data_to_plot, normalized_odata[:, col_ind])))
			self.header_raw_index[2] = col_ind

		else: # add a row of zeroes if headers for the z-axis is None
			z_col = np.zeros((self.pca.get_num_rows(), 1))
			self.data_to_plot = (np.hstack((self.data_to_plot, z_col)))


		# add a row of ones
		homogeneous = np.ones((self.pca.get_num_rows(), 1))
		self.data_to_plot = (np.hstack((self.data_to_plot, homogeneous)))

		# generate 2 matrices for color and size dimensions
		if source[3] == "PCA Data":
			col_ind = self.pca.evecs2index[headers[3]]
			self.color_dim = normalized_pdata[:, col_ind]
		else:
			col_ind = self.data.header2matrix[headers[3]]
			self.color_dim = normalized_odata[:, col_ind]

		if source[4] == "PCA Data":
			col_ind = self.pca.evecs2index[headers[4]]
			self.size_dim = normalized_pdata[:, col_ind]
		else:
			col_ind = self.data.header2matrix[headers[4]]
			self.size_dim = normalized_odata[:, col_ind]

		# place data
		vtm = self.view.build()
		pts = (vtm * self.data_to_plot.T).T

		for i in range(len(self.data_to_plot)):
			x = pts[i, 0]
			y = pts[i, 1]

			alpha = self.color_dim[i, 0]
			alpha = round(alpha, 2)
			color = ((1 - alpha) * 255, (1 - alpha) * 255, alpha * 255)
			color = '#%02x%02x%02x' % color

			size = 2 + 6 * self.size_dim[i, 0]

			pt = self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=color, outline='')
			self.objects.append(pt)

        	self.canvas.itemconfig(self.x_label, text=headers[0])
        	self.canvas.itemconfig(self.y_label, text=headers[1])
        	self.canvas.itemconfig(self.z_label, text=headers[2])

        	self.axisHeader = [headers[0], headers[1], headers[2]]

		# update labels on the canvas
		if self.pca_labels != []:
			self.pca_labels[0].config(text="X-Axis: " +headers[0])
			self.pca_labels[1].config(text="Y-Axis: " + headers[1])
			self.pca_labels[2].config(text="Z-Axis: " + str(headers[2]))
			self.pca_labels[3].config(text="Color: " + headers[3])
			self.pca_labels[4].config(text="Size: " + headers[4])
			self.pca_labels[5].config(text="PCA Headers: \n"+"\n".join(map(str,self.pca_headers)))

		else:
			label1 = tk.Label(self.canvas, text="X-Axis: " +headers[0], width=25)
			label1.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label2 = tk.Label(self.canvas, text="Y-Axis: " + headers[1], width=25)
			label2.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label3 = tk.Label(self.canvas, text="Z-Axis: " + str(headers[2]), width=25)
			label3.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label4 = tk.Label(self.canvas, text="Color: " + headers[3], width=25)
			label4.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label5 = tk.Label(self.canvas, text="Size: " + headers[4], width=25)
			label5.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label6 = tk.Label(self.canvas, text="PCA Headers: \n" +"\n".join(map(str,self.pca_headers)), width=25)
			label6.pack(side=tk.TOP, anchor=tk.E, pady=5)
			self.pca_labels = [label1, label2, label3, label4,label5, label6]

	# Plot, delete, view and export prior PCAs
	def pca_analysis_history(self):
		pca = PCAHistory(self.root, self.pca_history)
		selection = pca.result
		print "PCA plot:", selection

		# update history (in case analysis are deleted)
		self.pca_history = pca.history

		if selection == None:
			return
		if selection[0] == None: #return if no analysis is chosen
			return
		if selection[1] == None:
			return

		# return if any dimension other than Z-axis is not selected
		for i in range(5):
			if i != 2:
				if selection[2][i] == None:
					return

		information = self.pca_history[selection[0]] # extract filename
		fn = information[0]
		self.pca = information[1]
		self.data = data.Data(fn)
		self.file.set(os.path.basename(fn))
		self.resetForNewFile()
		self.plot_pca(selection[1], selection[2])  # plot data, pass in data source and headers


	# Perform clustering on selected data headers
	def handle_clustering(self):
		if self.data is None:
			print "No file selected"
			return

		self.data = data.Data(self.fn) #reset data

		self.headers = self.data.matrix_headers  # store original data headers
		results = ClusterDialog(self.root, self.headers).result  # get clustering headers
		if results == None:
			return

		for i in range(len(results)):
			if results[i] == None:
				if ClusterDialog.selected[i] != None:
					results[i] = ClusterDialog.selected[i]

		self.cluster_headers = results[0]
		self.cluster_size = results[1]
		self.cluster_color = results[2]
		self.cluster_metric = results[3]
		print "clustering selections: ", results
		if None in results:
			return

		self.original_data = data.Data(self.fn) # generate original data
		self.data = self.original_data # reset data to original data
		ClusterDialog.selected = results[:] #set selected values for ClusterDialog

		if results[0] == ["PC0","PC1","PC2"]:
			ClusterDialog.selected[0] = None
			pca_name = self.handle_pca_analysis()
			if pca_name == None:
				print "invalid PCA selections"
				return
			information = self.pca_history[pca_name]
			self.pca = information[1]
			self.data.add_column("PC0","numeric",self.pca.matrix_data[:,0].T.tolist()[0])
			self.data.add_column("PC1", "numeric", self.pca.matrix_data[:, 1].T.tolist()[0])
			self.data.add_column("PC2", "numeric", self.pca.matrix_data[:, 2].T.tolist()[0])

		clustering = analysis.kmeans(self.data, self.cluster_headers, self.cluster_size, metric=self.cluster_metric)
		addData = clustering[1].T.tolist()[0] #IDs
		self.data.add_column("ID", "numeric", addData) #add IDs

		# enable the user to name a clustering analysis
		name = SaveDialog(self.root, title='Save PCA Analysis').result
		if name == None:
			return
		if name == "":
			name = "clustering_analysis_1"  # give a default name
		#add to dictionary and option menu
		self.cluster_history[name] = self.data

		# handle cluster menu
		self.clusterOption.set(self.cluster_history.keys()[0])
		self.clusterMenu['menu'].delete(0,'end')
		for k in self.cluster_history.keys():
			self.clusterMenu['menu'].add_command(label=k,
												 command=tk._setit(self.clusterOption,k))


		# # add columns of means to data
		# self.clusterMeans = clustering[0].T.tolist()[0]
		# means = clustering[0].T.tolist()
		# for i in range(len(means)):
		# 	self.clusterData.add_column("Mean" + str(i), "numeric", np.matrix(means[i])

		self.data.write("clusterdata.csv")





	def plot_clusters(self):
		self.clusterData = self.cluster_history[self.clusterOption.get()]
		# self.clusterData = self.data
		print "selected cluster name: ", self.clusterData
		if self.clusterData is None:
			print "no clustering analysis selected"
			return
		self.clearCanvas()
		headers = ClusterPlotDialog(self.root,self.data.get_headers()).result
		if headers is None:
			return
		print headers
		if headers[0] is None or headers[1] is None:
			print "Not enough headers selected"
			return
		self.axisHeader = headers

		self.matrix_data = self.data.matrix_data  # store numeric data
		self.normalized_data = analysis.normalize_columns_separately(self.data)


		self.clearCanvas()

		x_index = self.data.header2matrix[headers[0]]
		y_index = self.data.header2matrix[headers[1]]
		x_col = self.normalized_data[:, x_index]
		y_col = self.normalized_data[:, y_index]
		self.header_raw_index[0] = x_index
		self.header_raw_index[1] = y_index

		if headers[2] != None:
			z_index = self.data.header2matrix[headers[2]]
			z_col = self.normalized_data[:, z_index]
			self.header_raw_index[2] = z_index
			self.data_to_plot = self.normalized_data[:, [x_index, y_index, z_index]]

		else:
			z_col = np.zeros((len(self.normalized_data), 1))
			self.data_to_plot = self.normalized_data[:, [x_index, y_index]]
			self.data_to_plot = np.hstack((self.data_to_plot, z_col))

		homogeneous = np.ones((len(self.normalized_data), 1))
		self.data_to_plot = np.hstack((self.data_to_plot, homogeneous))

		self.ptShape = "circle"
		self.ptColor = "cluster"
		self.ptSize = 4
		size = 4

		ids = self.data.get_data(["ID"])
		if self.cluster_color == "Preselected":
			N = self.cluster_size
			HSV_tuples = [(x * 1.0 / N,0.5, 1) for x in range(N)]
			colors = []
			for c in map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples):
				colors.append(tuple([255*x for x in c]))
			colors = ['#%02x%02x%02x' % c for c in colors]
		else:
			N = self.cluster_size
			rgb = [(1-x * 1.0 / N, 1-x * 1.0 / N, x * 1.0 / N) for x in range(N)]
			colors = []
			for c in rgb:
				colors.append(tuple([255*x for x in c]))
			colors = ['#%02x%02x%02x' % c for c in colors]

		# place the data
		vtm = self.view.build()
		pts = (vtm * self.data_to_plot.T).T

		for i in range(len(self.data_to_plot)):
			x = pts[i, 0]
			y = pts[i, 1]
			color = colors[int(ids[i,0])]

			pt = self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=color, outline='')
			self.objects.append(pt)

		self.canvas.itemconfig(self.x_label, text="X: " + str(self.axisHeader[0]))
		self.canvas.itemconfig(self.y_label, text="Y: " + str(self.axisHeader[1]))
		self.canvas.itemconfig(self.z_label, text="Z: " + str(self.axisHeader[2]))


		# update labels on the canvas
		if self.cluster_labels != []:
			self.cluster_labels[0].config(text="Cluster Headers: \n" + "\n".join(map(str, self.cluster_headers)))
			self.cluster_labels[1].config(text="Cluster Size: " + str(self.cluster_size))
			self.cluster_labels[2].config(text="Color Scheme: " + self.cluster_color)
			self.cluster_labels[3].config(text="Distance Metric: " + self.cluster_metric)
			self.cluster_labels[4].config(text="X-Axis: " + headers[0])
			self.cluster_labels[5].config(text="Y-Axis: " + headers[1])
			self.cluster_labels[6].config(text="Z-Axis: " + headers[2])
		else:
			label1 = tk.Label(self.canvas, text="Cluster Headers: \n" + "\n".join(map(str, self.cluster_headers)), width=25)
			label1.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label2 = tk.Label(self.canvas, text="Cluster Size: " + str(self.cluster_size), width=25)
			label2.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label3 = tk.Label(self.canvas, text="Color Scheme: " + self.cluster_color, width=25)
			label3.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label4 = tk.Label(self.canvas, text="Distance Metric: " + self.cluster_metric, width=25)
			label4.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label5 = tk.Label(self.canvas, text="X-Axis: " + headers[0], width=25)
			label5.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label6 = tk.Label(self.canvas, text="Y-Axis: " + headers[1], width=25)
			label6.pack(side=tk.TOP, anchor=tk.E, pady=5)
			label7 = tk.Label(self.canvas, text="Z-Axis: " + headers[2], width=25)
			label7.pack(side=tk.TOP, anchor=tk.E, pady=5)
			self.cluster_labels = [label1, label2, label3, label4,label5, label6,label7]




	# clear any existing regression lines, regression labels and data points
	def clearCanvas(self):
		for obj in self.RLs:
			self.canvas.delete(obj)
		for label in self.RL_labels:
			label.destroy()
		for label in self.pca_labels:
			label.destroy()
		for label in self.cluster_labels:
			label.destroy()
		self.RL_labels = []
		self.pca_labels = []
		self.cluster_labels = []
		self.RLs = []
		self.RLpts = None
		self.clearData()
		self.current = [None, None]

	# reset necessary variables before openning a new file
	def resetForNewFile(self):
		PlotDataDialog.selected = [None, None, None, None, None, None]
		LRDialog.selected = [None, None, None, None, None]
		MLRDialog.selected = [None, None, None, None, None]

		self.reset()
		self.clearCanvas()

		self.view = view.View()
		self.matrix_data = self.data.matrix_data  # store numeric data
		self.normalized_data = analysis.normalize_columns_separately(self.data)
		self.headerSelected = []
		self.headers = []
		self.ptShape = None
		self.ptColor = None
		self.ptSize = None
		self.data_to_plot = None
		self.size_dim = None
		self.color_dim = None
		self.canvas.itemconfig(self.x_label, text="X: ")
		self.canvas.itemconfig(self.y_label, text="Y: ")
		self.canvas.itemconfig(self.z_label, text="Z: ")
		self.location.set("None: None,	None: None,	 None: None")
		self.scaling.set("Scale: 100%")
		self.angle.set("Rotation Angle: X-Axis 0, Y-Axis 0")

		self.header_source = ["Original Data","Original Data","Original Data"]
		self.header_raw_index = [None, None,None]

	# clear all data points1
	def clearData(self, event=None):
		print "Clearing"

		for obj in self.objects:
			self.canvas.delete(obj)

		self.objects = []

	# reset screen
	def reset(self):
		screen0 = self.view.screen
		self.view = view.View()
		self.view.screen = screen0


		self.updateAxes()
		self.updatePoints()
		self.updateFits()

		# update plane
		self.planeOption.set("XY")
		self.setPlane()

		# reset labels
		self.scaling.set("Scale: 100%")
		self.angle.set("Rotation Angle: X-Axis 0, Y-Axis 0")

		print "reset"

	def setPlane(self):
		print "set to " + self.planeOption.get() + " plane"
		screen0 = self.view.screen
		self.view = view.View()
		self.view.screen = screen0

		if self.planeOption.get() == "XY":
			self.view.vrp = np.matrix([0.5, 0.5, 1])  # center of the view windowl; origin of view reference coordinates
			self.view.vpn = np.matrix([0, 0, -1])  # direction of viewing
			self.view.vup = np.matrix([0, 1, 0])  # view up vector
			self.view.u = np.matrix([-1, 0, 0])

		elif self.planeOption.get() == "YZ":
			self.view.vrp = np.matrix([0, 0.5, 0.5])  # center of the view windowl; origin of view reference coordinates
			self.view.vpn = np.matrix([1, 0, 0])  # direction of viewing
			self.view.vup = np.matrix([0, 1, 0])  # view up vector
			self.view.u = np.matrix([0, 0, -1])

		elif self.planeOption.get() == "XZ":
			self.view.vrp = np.matrix([0.5, 1, 0.5])  # center of the view windowl; origin of view reference coordinates
			self.view.vpn = np.matrix([0, -1, 0])  # direction of viewing
			self.view.vup = np.matrix([1, 0 , 0])  # view up vector
			self.view.u = np.matrix([0, 0, -1])

		# apply scaling
		self.view.extent = [self.view.extent[0] * self.f, self.view.extent[1] * self.f,
							self.view.extent[2] * self.f]

		# update label
		self.angle.set("Rotation Angle: X-Axis 0, Y-Axis 0")

		self.updateAxes()
		self.updatePoints()
		self.updateFits()

	def handleResize(self, event = None):
		w = self.canvas.winfo_width()
		h = self.canvas.winfo_height()
		shorter = min(w,h)
		if hasattr(self, 'view'):
			self.view.screen = [shorter-250, shorter-250]
			self.updateAxes()
			self.updatePoints()
			self.updateFits()

	def main(self):
		print 'Entering main loop'
		self.root.mainloop()



class Dialog(tk.Toplevel):

	def __init__(self, parent, title = None):

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent

		self.result = None

		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()

		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
								  parent.winfo_rooty()+50))

		self.initial_focus.focus_set()

		self.wait_window(self)

	#
	# construction hooks

	def body(self, master):
		# create dialog body.  return widget that should have
		# initial focus.  this method should be overridden
		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	#
	# standard button semantics

	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set() # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()

		self.cancel()

	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	#
	# command hooks

	def validate(self):

		return 1 # override

	def apply(self):

		pass # override


class PlotDataDialog(Dialog):
	# created a static class variable for selected distributions
	selected = [None,None,None,None,None,None]

	def __init__(self, parent, headers, title = None):
		self.headers = headers
		self.values = [None,None,None,None,None,None]
		Dialog.__init__(self, parent, title)
  		self.result = self.values[:]


	def body(self, master):
		self.title("Plot Data")

		label = tk.Label( master, text="X Axis", width=20 )
		label.grid( row=0, column= 1)
		listBox1 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox1.grid(row=1, column= 1)

		label = tk.Label( master, text="Y Axis", width=20 )
		label.grid( row=0, column= 2)
		listBox2 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox2.grid(row=1, column= 2)

		label = tk.Label( master, text="Z Axis", width=20 )
		label.grid( row=0, column= 3)
		listBox3 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox3.grid(row=1, column= 3)

		label = tk.Label( master, text="Color", width=20 )
		label.grid( row=0, column= 4)
		listBox4 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox4.grid(row=1, column= 4)

		label = tk.Label( master, text="Size", width=20 )
		label.grid( row=0, column= 5)
		listBox5 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox5.grid(row=1, column= 5)

		label = tk.Label( master, text="Shape", width=20 )
		label.grid( row=0, column= 6)
		listBox6 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox6.grid(row=1, column= 6)

		boxes = [listBox1, listBox2,listBox3,listBox4,listBox5,listBox6]
		colors = ["None"]
		sizes = ["None"]
		shapes = ["circle", "square","triangle"]
		for h in self.headers:
			colors.append(h)
			sizes.append(h)


		for box in boxes[:3]:
			box.insert(0, "None")
			for i in range(len(self.headers)):
				box.insert(i+1, self.headers[i])

		for i in range(len(colors)):
			listBox4.insert(i, colors[i])

		for i in range(len(sizes)):
			listBox5.insert(i, sizes[i])

		for i in range(len(shapes)):
			listBox6.insert(i, shapes[i])

		for i in range(len(boxes)):
			boxes[i].focus_set()
			boxes[i].bind("<<ListboxSelect>>",lambda event, index = i: self.select(event,index))

			if PlotDataDialog.selected[i] != None:
				if i < 3:
					if PlotDataDialog.selected[i] == "None":
						boxes[i].selection_set(0)
					else:
						boxes[i].selection_set(self.headers.index(PlotDataDialog.selected[i]) + 1)
				if i == 3:
					listBox4.selection_set(colors.index(PlotDataDialog.selected[i]))
				if i == 4:
					listBox5.selection_set(sizes.index(PlotDataDialog.selected[i]))
				if i == 5:
					listBox6.selection_set(shapes.index(PlotDataDialog.selected[i]))
			else:
				boxes[i].selection_set(0)

		self.boxes = boxes


	def select(self,event, index):
		caller = event.widget
		selection = caller.curselection()
		self.values[index]= caller.get(selection[0])

	def apply(self, event = None):
		self.result = self.values


class LRDialog(Dialog):
	# created a static class variable for selected distributions
	selected = [None,None,None,None,None]

	def __init__(self, parent, headers, title = None):
		self.headers = headers
		self.values = [None,None,None,None,None]
		Dialog.__init__(self, parent, title)


	def body(self, master):
		self.title("Simple Linear Regression")

		label = tk.Label( master, text="Independent Variable(X)", width=20 )
		label.grid( row=0, column= 1)
		listBox1 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox1.grid(row=1, column= 1)

		label = tk.Label( master, text="Dependent Variable(Y)", width=20 )
		label.grid( row=0, column= 2)
		listBox2 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox2.grid(row=1, column= 2)

		label = tk.Label( master, text="Point Color", width=20 )
		label.grid( row=0, column= 4)
		listBox3 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox3.grid(row=1, column= 4)

		label = tk.Label( master, text="Point Size", width=20 )
		label.grid( row=0, column= 5)
		listBox4 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox4.grid(row=1, column= 5)

		label = tk.Label( master, text="Point Shape", width=20 )
		label.grid( row=0, column= 6)
		listBox5 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox5.grid(row=1, column= 6)

		boxes = [listBox1, listBox2,listBox3,listBox4,listBox5]
		colors = ["None"]
		sizes = ["None"]
		shapes = ["circle", "square","triangle"]
		for h in self.headers:
			colors.append(h)
			sizes.append(h)

		for i in range(len(self.headers)):
			listBox1.insert(i, self.headers[i])
			listBox2.insert(i, self.headers[i])

		for i in range(len(colors)):
			listBox3.insert(i, colors[i])

		for i in range(len(sizes)):
			listBox4.insert(i, sizes[i])

		for i in range(len(shapes)):
			listBox5.insert(i, shapes[i])

		for i in range(len(boxes)):
			boxes[i].focus_set()
			boxes[i].bind("<<ListboxSelect>>",lambda event, index = i: self.select(event,index))

			if LRDialog.selected[i] != None:
				if i < 2:
					boxes[i].selection_set(self.headers.index(LRDialog.selected[i]))
				if i == 2:
					boxes[i].selection_set(colors.index(LRDialog.selected[i]))
				if i == 3:
					boxes[i].selection_set(sizes.index(LRDialog.selected[i]))
				if i == 4:
					boxes[i].selection_set(shapes.index(LRDialog.selected[i]))
			else:
				if i == 1:
					boxes[i].selection_set(1)
				else:
					boxes[i].selection_set(0)


	def select(self,event, index):
		caller = event.widget
		selection = caller.curselection()
		self.values[index]= caller.get(selection[0])


	def apply(self, event = None):
		self.result = self.values


class MLRDialog(Dialog):
	# created a static class variable for selected distributions
	selected = [None, None, None, None, None]

	def __init__(self, parent, headers, title=None):
		self.headers = headers
		self.values = [None, None, None, None, None]
		Dialog.__init__(self, parent, title)

	def body(self, master):
		self.title("Multiple Linear Regression")

		label = tk.Label(master, text="Independent Variable(X)", width=20)
		label.grid(row=0, column=1)
		listBox1 = tk.Listbox(master, selectmode=tk.MULTIPLE, exportselection=0)
		listBox1.grid(row=1, column=1)

		label = tk.Label(master, text="Dependent Variable(Y)", width=20)
		label.grid(row=0, column=2)
		listBox2 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		listBox2.grid(row=1, column=2)

		label = tk.Label(master, text="Point Color", width=20)
		label.grid(row=0, column=4)
		listBox3 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		listBox3.grid(row=1, column=4)

		label = tk.Label(master, text="Point Size", width=20)
		label.grid(row=0, column=5)
		listBox4 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		listBox4.grid(row=1, column=5)

		label = tk.Label(master, text="Point Shape", width=20)
		label.grid(row=0, column=6)
		listBox5 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		listBox5.grid(row=1, column=6)

		boxes = [listBox1, listBox2, listBox3, listBox4, listBox5]
		colors = ["None"]
		sizes = ["None"]
		shapes = ["circle", "square", "triangle"]
		for h in self.headers:
			colors.append(h)
			sizes.append(h)

		for i in range(len(self.headers)):
			listBox1.insert(i, self.headers[i])
			listBox2.insert(i, self.headers[i])

		for i in range(len(colors)):
			listBox3.insert(i, colors[i])

		for i in range(len(sizes)):
			listBox4.insert(i, sizes[i])

		for i in range(len(shapes)):
			listBox5.insert(i, shapes[i])

		for i in range(len(boxes)):
			boxes[i].focus_set()
			boxes[i].bind("<<ListboxSelect>>", lambda event, index=i: self.select(event, index))

			if MLRDialog.selected[i] != None:
				if i == 0:
					for selection in MLRDialog.selected[i]:
						boxes[i].selection_set(self.headers.index(selection))
				elif i == 1:
					boxes[i].selection_set(self.headers.index(MLRDialog.selected[i]))
				elif i == 2:
					boxes[i].selection_set(colors.index(MLRDialog.selected[i]))
				elif i == 3:
					boxes[i].selection_set(sizes.index(MLRDialog.selected[i]))
				elif i == 4:
					boxes[i].selection_set(shapes.index(MLRDialog.selected[i]))
			else:
				if i == 1:
					boxes[i].selection_set(1)
				else:
					boxes[i].selection_set(0)

	def select(self, event, index):
		caller = event.widget
		selection = caller.curselection()
		if index == 0:
			self.values[index] = []
			for item in selection:
				self.values[0].append(caller.get(item))
		else:
			self.values[index] = caller.get(selection[0])

	def apply(self, event=None):
		self.result = self.values
		if self.result[0] == []:
			self.result[0] = None

# Name a file
class SaveDialog(Dialog):
	def __init__(self, parent,title=None):
		self.result = None
		Dialog.__init__(self, parent, title)

	def body(self, master):
		tk.Label(master, text="Name:").grid(row=0)
		self.e1 =tk.Entry(master)
		self.e1.grid(row=0, column=1)

	def apply(self):
		self.result = self.e1.get()

# View and replot prior linear regressions
class AnalysisHistory(Dialog):
	def __init__(self, parent, history, title=None):
		self.value = None
		self.history = history
		Dialog.__init__(self, parent, title)

	def body(self, master):
		self.title("Plot Prior Analysis")
		label = tk.Label(master, text="Choose Analysis", width=20)
		label.grid(row=0, column=1)
		listBox1 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		listBox1.grid(row=1, column=1)

		for i in range(len(self.history.keys())):
			listBox1.insert(i, self.history.keys()[i])

		listBox1.focus_set()
		listBox1.bind("<<ListboxSelect>>", self.select)

	def select(self, event):
		caller = event.widget
		selection = caller.curselection()
		self.value = caller.get(selection[0])


	def apply(self):
		self.result = self.value

# Generate a PCA by selecting headers
class PCA(Dialog):
	selected = None;
	def __init__(self, parent, headers, title=None):
		self.headers = headers;
		self.value = None
		Dialog.__init__(self, parent, title)

	def body(self, master):
		self.title("PCA Analysis")
		label = tk.Label(master, text="PCA Analysis", width=20)
		label.grid(row=0, column=1)
		listBox1 = tk.Listbox(master, selectmode=tk.MULTIPLE, exportselection=0)
		listBox1.grid(row=1, column=1)

		for i in range(len(self.headers)):
			listBox1.insert(i, self.headers[i])

		listBox1.focus_set()
		listBox1.bind("<<ListboxSelect>>", self.select)

	def select(self, event):
		caller = event.widget
		selection = caller.curselection()
		self.value = []
		for item in selection:
			self.value.append(caller.get(item))


	def apply(self):
		self.result = self.value
		if self.result == []:
			self.result = None

# Plot, delete, view and export prior PCAs.
class PCAHistory(Dialog):
	def __init__(self, parent, history, title=None):
		self.result = [None,None,None,None]
		self.value = [None,None,None,None]
		self.history = history
		Dialog.__init__(self, parent, title)

	def body(self, master):
		self.title("PCA Analysis History")
		label = tk.Label(master, text="PCA Analysis", width=20)
		label.grid(row=0, column=1)
		self.listBox1 = tk.Listbox(master, selectmode=tk.SINGLE, exportselection=0)
		self.listBox1.grid(row=1, column=1)

		for i in range(len(self.history.keys())):
			self.listBox1.insert(i, self.history.keys()[i])

		self.listBox1.focus_set()
		self.listBox1.bind("<<ListboxSelect>>", self.select)

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="Plot PCA", width=10, command=self.plot_pca)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Delete", width=10, command=self.delete_selection)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="PCA Report", width=10, command=self.report)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Export Analysis", width=10, command=self.export)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	# delete current selected analysis from the history
	def delete_selection(self):
		if self.value[0] != None:
			self.listBox1.delete(self.selected_index)
			del self.history[self.value] #delete it from the history dictionary
			self.value[0] = None

	# view selected analysis in a dialog box
	def report(self):
		if self.value[0] != None:
			PCAEig(self, self.history[self.value[0]][1])

	# export selected analysis to a csv file
	def export(self):
		if self.value[0] != None:
			filename = SaveDialog(self,title = "Export PCA").result
			if filename == None:
				return
			if filename == "":
				filename = "PCA_1.csv"
			else:
				filename += ".csv"

			pca = self.history[self.value[0]][1]
			pca.export_analysis(filename)


	def plot_pca(self):
		if self.value[0] != None:
			source = PCADataSource(self).result
			if source != None:
				headers = PCADataPlot(self, source, self.history[self.value[0]][1].get_data_headers(),
									  self.history[self.value[0]][2]).result
				self.value[1] = source
				self.value[2] = headers[:-1]
				self.value[3] = headers[-1]

				self.ok()


	def select(self, event):
		caller = event.widget
		selection = caller.curselection()
		self.value[0] = caller.get(selection[0])
		self.selected_index = selection[0]


	def apply(self):
		self.result = self.value

# View eigenvalues and eigenvectors of a PCA
class PCAEig(Dialog):
	def __init__(self, parent, pca, title=None):
		self.pca = pca
		Dialog.__init__(self, parent, title)

	def body(self, master):
		self.title("View Information of a PCA Analysis")
		evals = self.pca.get_eigenvalues()
		evecs = self.pca.get_eigenvectors()
		headers = self.pca.get_data_headers()
		cum_percentage = self.pca.get_cum_percentage()

		# first row
		label = tk.Label(master, text="E-vec", width=10)
		label.grid(row=0, column=0)
		label = tk.Label(master, text="E-val", width=10)
		label.grid(row=0, column=1)
		label = tk.Label(master, text="Cumulative", width=10)
		label.grid(row=0, column=2)
		for i in range(len(headers)):
			label = tk.Label(master, text=headers[i], width=10)
			label.grid(row=0, column=i+3)

		# eigenvector index and values
		for i in range(len(headers)):
			if i < 10:
				t = "P0" + str(i)
			else:
				t = "P" + str(i)
			label = tk.Label(master, text= t, width=10)
			label.grid(row=i+1, column=0)

			for j in range(len(headers)):
				label = tk.Label(master, text=str(round(evecs[i,j],4)), width=10)
				label.grid(row=i+1, column=j+3)

		# cumulative values
			for j in range(len(headers)):
				label = tk.Label(master, text=str(cum_percentage[i]), width=10)
				label.grid(row=i+1, column=2)


		# eigenvalues
		for i in range(len(headers)):
			label = tk.Label(master, text= str(round(evals[i],4)), width=10)
			label.grid(row=i+1, column=1)



	def apply(self):
		return

# Choose data source for a PCA plot
class PCADataSource(Dialog):

	def __init__(self, parent, title = None):
		self.values = ['PCA Data','PCA Data','PCA Data','PCA Data','PCA Data']
		Dialog.__init__(self, parent, title)


	def body(self, master):
		self.title("Choose PCA Data Source")

		label = tk.Label( master, text="X-Axis", width=20 )
		label.grid( row=0, column= 0)
		listBox1 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox1.grid(row=1, column= 0)

		label = tk.Label( master, text="Y-Axis", width=20 )
		label.grid( row=0, column= 1)
		listBox2 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox2.grid(row=1, column= 1)

		label = tk.Label( master, text="Z-Axis", width=20 )
		label.grid( row=0, column= 2)
		listBox3 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox3.grid(row=1, column= 2)

		label = tk.Label( master, text="Color", width=20 )
		label.grid( row=0, column= 3)
		listBox4 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox4.grid(row=1, column= 3)

		label = tk.Label( master, text="Size", width=20 )
		label.grid( row=0, column= 4)
		listBox5 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox5.grid(row=1, column= 4)


		boxes = [listBox1, listBox2,listBox3,listBox4,listBox5]
		for i in range(len(boxes)):
			boxes[i].insert(0,'PCA Data')
			boxes[i].insert(1,'Original Data')
			boxes[i].focus_set()
			boxes[i].selection_set(0)
			boxes[i].bind("<<ListboxSelect>>",lambda event, index = i: self.select(event,index))

	def select(self,event, index):
		caller = event.widget
		selection = caller.curselection()
		self.values[index]= caller.get(selection[0])


	def apply(self, event = None):
		self.result = self.values

# Choose headers for a PCA plot
class PCADataPlot(Dialog):

	def __init__(self, parent, source, pheaders, oheaders, title = None):
		self.source = source
		self.pheaders = pheaders
		self.oheaders = oheaders
		self.values = [None,None,None,None,None,None]
		Dialog.__init__(self, parent, title)

	def body(self, master):
		self.title("Plot Data")

		label = tk.Label( master, text="X-Axis", width=20 )
		label.grid( row=0, column= 0)
		listBox1 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox1.grid(row=1, column= 0)

		label = tk.Label( master, text="Y-Axis", width=20 )
		label.grid( row=0, column= 1)
		listBox2 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox2.grid(row=1, column= 1)

		label = tk.Label( master, text="Z-Axis", width=20 )
		label.grid( row=0, column= 2)
		listBox3 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox3.grid(row=1, column= 2)

		label = tk.Label( master, text="Color", width=20 )
		label.grid( row=0, column= 3)
		listBox4 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox4.grid(row=1, column= 3)

		label = tk.Label( master, text="Size", width=20 )
		label.grid( row=0, column= 4)
		listBox5 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox5.grid(row=1, column= 4)

		pca_indices = []
		for i in range(len(self.pheaders)):
			if i < 10:
				pca_indices.append("P0" + str(i))
			else:
				pca_indices.append("P" + str(i))

		boxes = [listBox1, listBox2,listBox3,listBox4,listBox5]
		for i in range(len(boxes)):
			if self.source[i] == "PCA Data":
				for j in range(len(pca_indices)):
					boxes[i].insert(j, pca_indices[j])
			else:
				for j in range(len(self.oheaders)):
					boxes[i].insert(j, self.oheaders[j])

			boxes[i].focus_set()
			boxes[i].bind("<<ListboxSelect>>",lambda event, index = i: self.select(event,index))


	def select(self,event, index):
		caller = event.widget
		selection = caller.curselection()
		self.values[index]= caller.get(selection[0])


	def apply(self, event = None):
		self.result = self.values

# Choose headers and number of clusters for clustering
class ClusterDialog(Dialog):
	selected = [None, None, None, None]

	def __init__(self, parent, headers, title = None):
		self.headers = headers
		self.values = [None,None,None,None]
		Dialog.__init__(self, parent, title)


	def body(self, master):
		self.title("Clustering")

		label = tk.Label( master, text="Headers for Clustering", width=20 )
		label.grid( row=0, column= 0)
		listBox1 = tk.Listbox(master, selectmode = tk.MULTIPLE, exportselection = 0)
		listBox1.grid(row=1, column= 0)

		label = tk.Label( master, text="Number of Clusters", width=20 )
		label.grid( row=0, column= 1)
		self.size = tk.StringVar()
		self.entry = tk.Entry(master,textvariable=self.size)
		self.entry.grid(row=1, column= 1)

		label = tk.Label( master, text="Color Scheme", width=20 )
		label.grid( row=0, column= 2)
		listBox2 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox2.grid(row=1, column= 2)

		label = tk.Label( master, text="Distance Metric", width=20 )
		label.grid( row=0, column= 3)
		listBox3 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox3.grid(row=1, column= 3)

		self.cluster = tk.IntVar()
		self.cluster.set(0)
		c = tk.Checkbutton(master, text="Cluster on First 3 PCA Dimensions",
						   variable=self.cluster, command=self.cb)
		c.grid( row=2, column= 0)


		self.boxes = [listBox1, listBox2,listBox3]
		colors = ["Smooth","Preselected"]
		metrics = ["Euclidean","L1-Norm","Hamming","Correlation","Cosine"]

		for i in range(len(self.headers)):
			listBox1.insert(i, self.headers[i])

		for i in range(len(colors)):
			listBox2.insert(i, colors[i])

		for i in range(len(metrics)):
			listBox3.insert(i, metrics[i])

		for i in range(len(self.boxes)):
			self.boxes[i].focus_set()
			self.boxes[i].bind("<<ListboxSelect>>",lambda event, index = i: self.select(event,index))
			if ClusterDialog.selected[i] != None:
				if i == 0:
					for h in ClusterDialog.selected[i]:
						self.boxes[i].selection_set(self.headers.index(h))
				if i == 1:
					self.boxes[i].selection_set(colors.index(ClusterDialog.selected[i+1]))
				if i == 2:
					self.boxes[i].selection_set(metrics.index(ClusterDialog.selected[i+1]))
		if ClusterDialog.selected[1] != None:
			self.size.set(ClusterDialog.selected[1])


	def select(self,event, index):
		caller = event.widget
		selection = caller.curselection()
		if index == 0:
			self.values[index] = []
			for item in selection:
				self.values[0].append(caller.get(item))
		else:
			self.values[index+1] = caller.get(selection[0])

	def cb(self):
		if self.cluster.get() == 1:
			self.values[0] = ["PC0","PC1","PC2"]
			self.boxes[0].config(state=tk.DISABLED)
		else:
			self.boxes[0].config(state=tk.NORMAL)
			self.values[0] = []
			if self.boxes[0].curselection() != ():
				for s in self.boxes[0].curselection():
					self.values[0].append(self.boxes[0].get(s))

	def apply(self, event = None):
		if self.entry.get() != "":
			self.values[1] = int(self.entry.get())
		if self.values[0] == []:
			self.values[0] = None
		self.result = self.values

# Choose headers to visulize clusters
class ClusterPlotDialog(Dialog):

	def __init__(self, parent, headers, title = None):
		self.headers = headers
		self.values = [None,None,None]
		Dialog.__init__(self, parent, title)


	def body(self, master):
		self.title("Plot Clusters")

		label = tk.Label( master, text="X-Axis", width=20 )
		label.grid( row=0, column= 1)
		listBox1 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox1.grid(row=1, column= 1)

		label = tk.Label( master, text="Y-Axis", width=20 )
		label.grid( row=0, column= 2)
		listBox2 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox2.grid(row=1, column= 2)

		label = tk.Label( master, text="Z-Axis", width=20 )
		label.grid( row=0, column= 4)
		listBox3 = tk.Listbox(master, selectmode = tk.SINGLE, exportselection = 0)
		listBox3.grid(row=1, column= 4)

		boxes = [listBox1, listBox2,listBox3]

		for i in range(len(self.headers)):
			listBox1.insert(i, self.headers[i])
			listBox2.insert(i, self.headers[i])
			listBox3.insert(i, self.headers[i])

		for i in range(len(boxes)):
			boxes[i].focus_set()
			boxes[i].bind("<<ListboxSelect>>",lambda event, index = i: self.select(event,index))


	def select(self,event, index):
		caller = event.widget
		selection = caller.curselection()
		self.values[index]= caller.get(selection[0])


	def apply(self, event = None):

		self.result = self.values



if __name__ == "__main__":
	dapp = DisplayApp(1200, 675)
	dapp.main()


