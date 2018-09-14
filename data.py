# Jingyan Dong
# data.py
# CS251
# 2/14/17

import numpy as np
import csv
import datetime
import analysis


class Data:

	def __init__(self, filename = None):
		self.raw_headers = []
		self.raw_types = []
		self.raw_data = []
		self.header2raw = {}

		self.matrix_headers = []
		self.matrix_data = np.matrix([]) # matrix of numeric data
		self.header2matrix = {} # dictionary mapping header string to index of column in matrix data

		# extension 1
		self.enum_headers = []
		self.allEnum = [[]] #for buiding the dictonary, stores lists of enums in raw form
		self.enum = []
		self.data_with_enum = []
		self.matrix_enum = np.matrix([])
		self.matrix_data_with_enum =  np.matrix([])

		# extension 2
		self.date_headers = []
		self.date = []
		self.data_with_date = []
		self.matrix_date = np.matrix([])
		self.matrix_data_with_date =  np.matrix([])

		if filename != None:
			self.read(filename)

	def read(self, filename):
		fp = file(filename, 'rU')
		reader = csv.reader(fp, delimiter=',', quotechar='|')
		raw_headers_with_space = reader.next()
		raw_types_with_space = reader.next()

		#strip out leading and trailing spaces in raw headers and raw types
		for h in raw_headers_with_space:
			self.raw_headers.append(h.strip())
		for t in raw_types_with_space:
			self.raw_types.append(t.strip())


		numericMatrixList = []
		for row in reader:
			self.raw_data.append(row)

			numericList = []
			numericEnumList = []
			numericDateList = []
			enumList = []
			dateList = []

			enumType = 0
			for i in range(len(row)):
				if self.raw_types[i] == "numeric":
					numericList.append(float(row[i]))
					numericEnumList.append(float(row[i]))
					numericDateList.append(float(row[i]))

				elif self.raw_types[i] == "enum":
					if enumType != 0 and len(self.allEnum) <= enumType:
						self.allEnum.append([]) #create another list for a new type of enum
					row[i] = row[i].strip(" ")
					self.allEnum[enumType].append(row[i])
					numericEnumList.append(self.enumDict(enumType)[row[i]]) #append converted enum
					enumList.append(self.enumDict(enumType)[row[i]]) #append converted enum
					enumType += 1

				elif self.raw_types[i] == "date":
					row[i] = row[i].strip(" ")
					numericDateList.append(self.date_to_number(row[i]))
					dateList.append(self.date_to_number(row[i]))

			enumType = 0


			numericMatrixList.append(numericList)
			self.enum.append(enumList)
			self.data_with_enum.append(numericEnumList)

			self.date.append(dateList)
			self.data_with_date.append(numericDateList)

		self.matrix_data = np.matrix(numericMatrixList, np.float)
		self.matrix_data_with_enum = np.matrix(self.data_with_enum, np.float)
		self.matrix_enum = np.matrix(self.enum, np.float)

		self.matrix_date = np.matrix(self.date, np.float)
		self.matrix_data_with_date =  np.matrix(self.data_with_date, np.float)


		for i in range(len(self.raw_headers)):
	 		self.header2raw[self.raw_headers[i]] = i

		for header in self.raw_headers:
			if self.raw_types[self.header2raw[header]] == "numeric":
				self.matrix_headers.append(header)
			elif self.raw_types[self.header2raw[header]] == "enum":
				self.enum_headers.append(header)
			elif self.raw_types[self.header2raw[header]] == "date":
				self.date_headers.append(header)

		for i in range(len(self.matrix_headers)):
			self.header2matrix[self.matrix_headers[i]] = i



		fp.close()

	# extension1
	def enumDict(self, type):
		dict = {}
		count = 0
		for e in self.allEnum[type]:
			if e in dict:
				pass
			else:
				dict[e] = count
				count += 1
		return dict

	def printDict(self):
		for i in range(len(self.enum_headers)):
			print self.enumDict(i)


	# returns a list of all of the headers.
	def get_raw_headers(self):
		return self.raw_headers

	# returns a list of all of the types.
	def get_raw_types(self):
		return self.raw_types

	# returns the number of columns in the raw data set.
	def get_raw_num_columns(self):
		return len(self.raw_data[0])

	# returns the number of rows in the data set.
	def get_raw_num_rows(self):
		return len(self.raw_data)

	# returns a row of data (the type is list) given a row index (int).
	def get_raw_row(self, index):
		try:
			return self.raw_data[index]
		except:
			print "invalid index"


	# takes a row index (an int) and column header (a string) and returns the raw data at that location
	def get_raw_value(self, index, header):
		try:
			return self.raw_data[index][self.header2raw[header]]
		except:
			print "get_raw_value: invalid index or header"

	def get_headers(self):
		return self.matrix_headers

	def get_num_columns(self):
		return len(self.matrix_data[0])

	def get_num_rows(self):
		return len(self.matrix_data)

	# takes a row index and returns a row of numeric data
	def get_row(self, index):
		try:
			return self.matrix_data[index]
		except:
			print "invalid row index"

	# takes a row index (int) and column header (string) and returns the data in the numeric matrix.
	def get_value(self, index, header):
		try:
			return self.matrix_data[index, self.header2matrix[header]]
		except:
			print "get_value: invalid index or header"

	def get_data(self, headers=[]):
		if headers == []:
			return self.matrix_data

		indices = []
		for h in headers:
			indices.append(self.header2matrix[h])

		m = np.empty([self.get_num_rows(),0])
		for i in indices:
			m = np.hstack([m, self.matrix_data[:, i]])
		return m

	# extension 2
	def date_to_number(self, date):
		# replace other symbols with / to allow for more delimiters
		date = date.replace(' ', '/')
		date = date.replace('-', '/')

		date = date.split("/")
		d = datetime.date(int(date[0]),int(date[1]),int(date[2]))
		return d.toordinal()

	# extension
	def add_column(self, header, type, data):
		if len(data) != self.get_raw_num_rows():
			print "please choose data with row number " , self.get_raw_num_rows()
			return

		# add to raw data
		self.raw_headers.append(header)
		self.raw_types.append(type)
		for i in range(len(self.raw_data)):
			self.raw_data[i].append(data[i])
		#update dictonary
		self.header2raw[header] = len(self.raw_headers) -1

		# add to numeric data
		if type == "numeric":
			self.matrix_headers.append(header)
			# update dictonary
			self.header2matrix[header] = len(self.matrix_headers) -1
			# stack on the list as the bottom row and then transpose back
			toadd = np.matrix(data)
			self.matrix_data = np.hstack((self.matrix_data, toadd.T))

	def write(self,filename,headers=None):
		if headers is None:
			headers = self.get_headers()

		with open(filename, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow(headers)

			types = ['numeric']*len(headers)
			writer.writerow(types)

			data = self.get_data(headers).tolist()
			writer.writerows(data)


class PCAData(Data):
	def __init__(self,pheaders, pdata, evals, evecs, means,filename= None):
		Data.__init__(self, filename=None)
		self.evals = evals
		self.evecs = evecs
		self.means = means
		self.matrix_data = pdata
		self.matrix_headers = pheaders


		self.raw_headers = self.matrix_headers[:]
		for i in range(len(self.raw_headers)):
			self.raw_types.append("numeric")
		self.raw_data = self.matrix_data.tolist()

		self.evecs_indices = []
		for i in range(len(self.matrix_headers)):
			if i < 10:
				self.evecs_indices.append("P0" + str(i))
			else:
				self.evecs_indices.append("P" + str(i))
		self.raw_headers = self.evecs_indices

		for i in range(len(self.raw_headers)):
			self.header2raw[self.raw_headers[i]] = i
		for i in range(len(self.matrix_headers)):
			self.header2matrix[self.matrix_headers[i]] = i

		self.evecs2index = {}
		for i in range(len(self.evecs_indices)):
			self.evecs2index[self.evecs_indices[i]] = i


	# returns a copy of the eigenvalues as a single-row numpy matrix.
	def get_eigenvalues(self):
		return self.evals.copy()

	# returns a copy of the eigenvectors as a numpy matrix with the eigenvectors as rows.
	def get_eigenvectors(self):
		return self.evecs.copy()

	# returns the means for each column in the original data as a single row numpy matrix.
	def get_data_means(self):
		return self.means

	#  returns a copy of the list of the headers from the original data used to generate the projected data.
	def get_data_headers(self):
		return self.matrix_headers

	# return a list of cumulative percentages of eigenvalues
	def get_cum_percentage(self):
		evals = self.get_eigenvalues()
		evals_sum = float(np.sum(evals, axis=0))
		cum_evals = 0
		cum_percentage = []
		for i in range(len(self.get_data_headers())):
			cum_percentage.append((cum_evals + evals[i]) / evals_sum)
			cum_percentage[i] = round(cum_percentage[i], 4)
			cum_evals += evals[i]
		return cum_percentage

	def get_evecs_indices(self):
		return self.evecs_indices

	# export a PCA analysis to a csv file
	def export_analysis(self,filename):
		headers = ["Eigenvector","EigenValue", "Cummulative%","Mean"]

		for h in self.get_data_headers():
			headers.append(h)

		with open(filename, 'wb') as f:
			# round eigenvalues
			rounded_evals = []
			for val in self.get_eigenvalues():
				rounded_evals.append(round(val,4))

			# round means
			rounded_means = []
			for i in range(self.get_data_means().shape[1]):
				rounded_means.append(round(self.get_data_means()[0,i],4))

			list = [self.get_evecs_indices(),rounded_evals,self.get_cum_percentage(),rounded_means]
			# add rounded eigenvectors
			list = list + self.get_eigenvectors().T.round(4).tolist()

			writer = csv.writer(f)
			writer.writerow(headers)
			writer.writerows(zip(*list))



if __name__ == "__main__":
	# pca = analysis.pca(Data("AustraliaCoast.csv"),
	# 						 ['premin','premax','salmin','salmax',
	# 						  'minairtemp', 'maxairtemp','minsst',
	# 						  'maxsst','minsoilmoist','maxsoilmoist',
	# 						  'runoffnew'])
	# pca.export_analysis("PCA_1.csv")

	d = Data("AustraliaCoast.csv")
	d.write("testWrite.csv",['Longitude','Latitude','evapproxy','premin','premax','salmin'])


	# d = Data("testdata3.csv")
	# print "raw headers: ", d.get_raw_headers()
	# print "raw types: ", d.get_raw_types()
	# print "raw data: ", d.raw_data
	# print "raw row 0: ", d.get_raw_row(0)
	# print "raw row 20: ", d.get_raw_row(20)
	# print "raw row 0 thing 1: ", d.get_raw_value(0,"thing1")
	# print "header to row: ", d.header2raw
    #
	# print "numeric headers: ", d.get_headers()
	# print "header to column index: ", d.header2matrix
	# print "numeric row 0: ", d.get_row(0)
	# print "numeric row 0 thing 1: ", d.get_value(0, "thing1")
	# print "matrix_data: ", d.matrix_data
	# print "numeric data matrix with header thing1 and thing2, and row 0 and 1: ", d.get_data(["thing1","thing2"])

	# d = Data("colbyhours.csv")
	# d = Data("testdata2.csv")
	# print "-----eunm headers------"
	# print d.enum_headers , "\n"
	# print "-----enum dictonary-----"
	# print d.printDict() , "\n"
	# print "-----all enums before conversion-----"
	# print d.allEnum , "\n"
	# print "-----matrix with converted enum types-----"
	# print d.matrix_enum , "\n"
	# print "-----matrix with numeric and converted enum types-----"
	# print d.matrix_data_with_enum , "\n"


	# print "-----date headers------"
	# print d.date_headers , "\n"
	# print "-----matrix with converted date types-----"
	# print d.matrix_date , "\n"
	# print "-----matrix with numeric and converted date types-----"
	# print d.matrix_data_with_date , "\n"

	# print "------before adding a new column-------"
	# for list in d.raw_data:
	# 	print list
	# print "------after adding a new column-------"
	# # d.add_column("number", "numeric", [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
	# d.add_column("number", "numeric", [0, 1, 2])
	# for list in d.raw_data:
	# 	print list







