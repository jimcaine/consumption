##############################################################################################
# Jim Caine
# caine.jim@gmail.com
# Depaul University - CSC 672
# SAS Analytics Shootout
# June, 2015
##############################################################################################

##############################################################################################
# IMPORT DEPENDENCIES
##############################################################################################
import random
import datetime
import itertools
import os
import glob
import subprocess
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.grid_search import GridSearchCV
from sklearn import tree
import sklearn.feature_selection as FeatureSelection
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns # prettify plots
import pylab # for qqplot

# # get connection into R
# import rpy2.robjects as robjects
# import pandas.rpy.common as com
# print '------'

##############################################################################################
# DEFINE FILE PATHS AND CONSTANTS
##############################################################################################
DATA_PATH = './data/'
CHARTS_PATH = './charts/'
TABLES_PATH = './tables/'
ANIMATE_PATH = './animate/'
COLOR_SCHEME = ['#60A917', '#FA6800', '#1BA1E2', '#E51400',
				'#0050EF', '#76608A', '#6D8764', '#647687']

SECTORS = ['FOOD_SERVICE', 'HEALTH_CARE', 'K12_SCHOOLS', 'LODGING', 'OFFICE',
		   'RESIDENTIAL', 'STAND_ALONE_RETAIL', 'GROCERY']


sns.set_palette(COLOR_SCHEME)


##############################################################################################
# CREATE/MERGE DATASETS
##############################################################################################
class merge_datasets():
	def add_season(self, month_number):
	    if month_number in [1,2,3]:
	        return 'Winter'
	    elif month_number in [4,5,6]:
	        return 'Spring'
	    elif month_number in [7,8,9]:
	        return 'Summer'
	    elif month_number in [10,11,12]:
	        return 'Fall'

	def bin_hours(self, hour):
		if hour in [0,1,2,3]:
			return '0-3'
		elif hour in [4,5,6,7]:
			return '4-7'
		elif hour in [8,9,10,11]:
			return '8-11'
		elif hour in [12,13,14,15]:
			return '12-15'
		elif hour in [16,17,18,19]:
			return '16-19'
		elif hour in [20,21,22,23]:
			return '20-23'

	def bin_weekends(self, weekday):
		if weekday in ['Sat', 'Sun']:
			return 1
		else:
			return 0

	def bin_holidays(self, holiday_name):
		if holiday_name != 'No Holiday':
			return 1
		else:
			return 0

	def create_dummies(self, df, attr_list):
	    df = df.copy()
	    for a in attr_list:
	        dummies = pd.get_dummies(df[a])
	        colnames = []
	        for c in dummies.columns:
	            colnames.append(str(a) + '_' + str(c))
	        dummies.columns = colnames
	        df = pd.concat([df, dummies], axis=1)
	        df.drop([a], axis=1)
	    return df

	def create_dataset(self):
		# load data frames
		calendar_days_consumption = pd.DataFrame.from_csv(DATA_PATH + 'calendar_days_consumption.csv')
		powercity_consumption = pd.DataFrame.from_csv(DATA_PATH + 'powercity_consumption.csv')
		powercity_solarangle_consumption = pd.DataFrame.from_csv(DATA_PATH + 'powercity_solarangle_consumption.csv')
		powercity_weather_consumption = pd.DataFrame.from_csv(DATA_PATH + 'powercity_weather_consumption.csv')

		# merge dataframes
		df = calendar_days_consumption.merge(powercity_consumption, how='inner', on=['Month','Day'])
		df = df.merge(powercity_solarangle_consumption, how='inner', on=['Month', 'Day', 'Hour'])
		df = df.merge(powercity_weather_consumption, how='inner', on=['Month', 'Day', 'Hour'])

		# add in population by age group
		powercity_population = pd.DataFrame.from_csv(DATA_PATH + 'powercity_population.csv')
		powercity_population = powercity_population.drop(['CITY'], axis=1).sum()
		powercity_population = powercity_population / powercity_population.Total
		powercity_population = powercity_population.drop(['Total'])
		for index, value in powercity_population.iteritems():
		    colname = 'Population' + str(index)
		    df[colname] = [value]*df.shape[0]

		# add in square footage by age group
		sector_use_matrix = pd.DataFrame.from_csv(DATA_PATH + 'sector_use_matrix.csv')
		sector_dictionary = {'FOOD_SERVICE': 'Food Service',
		                     'GROCERY': 'Grocery',
		                     'HEALTH_CARE': 'Health Care',
		                     'K12_SCHOOLS': 'K-12',
		                     'LODGING': 'Lodging',
		                     'OFFICE': 'Office',
		                     'RESIDENTIAL': 'Residential',
		                     'STAND_ALONE_RETAIL': 'Stand-alone Retail'}
		sector_use_matrix_inprogress = []
		for index, value in df.iterrows():
		    sector_use_list = []
		    sector = value.Sector
		    sector = sector_dictionary[sector]
		    sqft_series = sector_use_matrix.loc[sector]
		    for index, value in sqft_series.iteritems():
		        sector_use_list.append(value)
		    sector_use_matrix_inprogress.append(sector_use_list)
		sector_use_matrix = pd.DataFrame(sector_use_matrix_inprogress,
		                                 columns = ['SQFT_less5', 'SQFT_5to18', 'SQFT_18to25', 'SQFT25to65','SQFT65plus'])
		df = pd.concat([df, sector_use_matrix], axis=1)

		# add in derivative variables - sqft per person
		df['sqft_per_person'] = df['Population<5'] * df['SQFT_less5'] + \
		                        df['Population5>18'] * df['SQFT_5to18'] + \
		                        df['Population18>25'] * df['SQFT_18to25'] + \
		                        df['Population25>65'] * df['SQFT25to65'] + \
		                        df['Population65+'] * df['SQFT65plus']
			
		# change data types
		df.Month = df.Month.astype(int)
		df.Day = df.Day.astype(int)
		df.School_Day = df.School_Day.astype(object)
		df.Hour = df.Hour.astype(int)

		# fill in Holiday nulls
		df.HolidayName = df.HolidayName.replace(np.nan, 'No Holiday')
		
		# decrement the hour by 1 (start with 0)
		df['Hour'] = df['Hour'].apply(lambda x: x-1)
		
		# add DateTime column
		df['DateTime'] = df[['Month', 'Day', 'Hour']].apply(lambda x: datetime.datetime(2012, x['Month'], x['Day'], x['Hour']), axis=1)
		df['Date'] = df['DateTime'].astype('datetime64[ns]').map(lambda t: t.date())
		df = df.drop(['Day_of_week', 'Year_x', 'Year_y'], axis=1)

		# add in season variable
		df['season'] = df['Month'].apply(lambda x: self.add_season(x))
		df = self.create_dummies(df, ['season'])

		# bin variables
		df['hourbin'] = df['Hour'].apply(lambda x: self.bin_hours(x))
		df['weekend'] = df['Weekdays'].apply(lambda x: self.bin_weekends(x))
		df['holiday'] = df['HolidayName'].apply(lambda x: self.bin_holidays(x))

		# transform response variable
		df['log_consumption'] = np.log(df.Electricity_KW_SQFT)

		# normalize continuous variables
		continuous_variables_list = ['Solar_Elevation', 'Cloud_Cover_Fraction', 'Dew_Point',
									 'Humidity_Fraction', 'Temperature', 'Visibility',
									 'sqft_per_person']
		for v in continuous_variables_list:
			minmax_colname = v + '_minmax'
			zscore_colname = v + '_zscore'
			df[minmax_colname] = (df[v] - df[v].min()) / (df[v].max() - df[v].min())
			df[zscore_colname] = (df[v] - df[v].mean()) / df[v].std()

		# create dummy variables
		df = self.create_dummies(df, ['Month','Weekdays','HolidayName','Hour','Sector','hourbin'])

		# write datasets to csv
		df.to_csv(DATA_PATH + 'consumption.csv')
		for s in df.Sector.unique():
			df_sector = df[df.Sector == s]
			df_sector.to_csv(DATA_PATH + 'consumption_%s' % s)

		return df




##############################################################################################
# LOAD DATA
##############################################################################################
class data_manipulator(object):
	def __init__(self):
		self.fullframe = pd.DataFrame.from_csv(DATA_PATH + 'consumption.csv')
		self.fullframe = self.fullframe.rename(columns = {'Electricity_KW_SQFT': 'consumption'})
		self.df = self.fullframe[['consumption']]
		self.add_date_feature()



	##########################################################################################
	# ADD ROLLING MEAN
	##########################################################################################
	def add_rolling_mean(self, feature, ndays):
		rolling_mean = pd.rolling_mean(self.df[feature], window=192*ndays, min_periods=1)
		rolling_mean = pd.DataFrame(rolling_mean)
		rolling_mean.columns = ['%s_rolling_mean_%sday' % (feature, str(ndays))]
		self.df = self.df.join(rolling_mean, how='outer')

	def add_all_rolling_means(self):
		features = ['Cloud_Cover_Fraction', 'Dew_Point', 'Humidity_Fraction',
					'Precipitable_Water', 'Temperature', 'Visibility']
		for f in features:
			for f2 in self.df.columns:
				if f in f2:
					self.add_rolling_mean(f2, 1)



	##########################################################################################
	# ADD HOURLY DERIVATIVE
	##########################################################################################
	def add_delta_feature(self, feature):
		# add in month, day, and hour to aggregate sectors (temp independent of sector)
		if 'Month' not in self.df.columns:
			self.add_any_feature('Month', object)
		if 'Day' not in self.df.columns:
			self.add_any_feature('Day', object)
		if 'Hour' not in self.df.columns:
			self.add_any_feature('Hour', object)

		df_by_day = self.df.groupby(['Month', 'Day', 'Hour']).mean()
		df_by_day = df_by_day[[feature]]
		month_levels = df_by_day.index.get_level_values('Month')
		day_levels = df_by_day.index.get_level_values('Day')
		hour_levels = df_by_day.index.get_level_values('Hour')
		delta_feature = [np.nan] # first row (no change)
		for i in range(1, len(month_levels)):
			delta = df_by_day.loc[(month_levels[i-1], day_levels[i-1], hour_levels[i-1])][feature] - \
				df_by_day.loc[(month_levels[i], day_levels[i], hour_levels[i])][feature]
			delta_feature.append(delta)
		colname = 'delta_' + feature
		df_by_day[colname] = delta_feature
		df_by_day['Month'] = month_levels
		df_by_day['Day'] = day_levels
		df_by_day['Hour'] = hour_levels
		try:
			df_by_day[colname].loc[(1,1)] = df_by_day[colname].loc[(1,2)]
		except:
			pass
		df_by_day = df_by_day.drop([feature], axis=1)
		self.df = self.df.merge(df_by_day, how='outer', on=['Month', 'Day', 'Hour'])


	def add_derivative_variable(self, feature_name):
		features = self.fullframe[[feature_name]]
		col_name = 'delta_' + feature_name
    	# features[col_name] = features[feature_name].apply(lambda x: )

	def add_all_delta_features(self):
		features = ['Cloud_Cover_Fraction', 'Dew_Point', 'Humidity_Fraction',
					'Precipitable_Water', 'Temperature', 'Visibility']
		for f in features:
			for f2 in self.df.columns:
				if f in f2:
					self.add_delta_feature(f2)


	##########################################################################################
	# ADD ORIGINAL DATASET
	##########################################################################################
	def add_categorical_features(self):
		features = self.fullframe[['Sector', 'Month', 'Weekdays', 'School_Day',
								   'HolidayName', 'Hour']]
		features = features.astype(object)
		self.df = pd.concat([self.df, features], axis=1)

	def add_continuous_features(self):
		features = self.fullframe[['Solar_Elevation', 'Cloud_Cover_Fraction', 'Dew_Point',
								   'Precipitable_Water','Humidity_Fraction', 'Temperature', 
								   'Visibility', 'sqft_per_person']]
		features = features.astype(float)
		self.df = pd.concat([self.df, features], axis=1)


	##########################################################################################
	# ADD NORMALIZED / STANDARD SPREADSHEET FEATURES
	##########################################################################################
	def add_minmax_continuous_features(self):
		features = self.fullframe[[c for c in self.fullframe.columns if 'minmax' in c]]
		features = features.astype(float)
		self.df = pd.concat([self.df, features], axis=1)

	def add_zscore_continuous_features(self):
		features = self.fullframe[[c for c in self.fullframe.columns if 'zscore' in c]]
		features = features.astype(float)
		self.df = pd.concat([self.df, features], axis=1)

	def add_dummy_features(self):
		match_strings = ['HolidayName_', 'Hour_', 'Sector_', 'Weekdays_']
		for ms in match_strings:
			features = self.fullframe[[c for c in self.fullframe.columns if ms in c]]
			features = features.astype(int)
			self.df = pd.concat([self.df, features], axis=1)


	##########################################################################################
	# ADD DATE
	##########################################################################################
	def add_date_feature(self):
		features = self.fullframe[['DateTime']]
		features = features.astype('datetime64[ns]')
		features['Date'] = features['DateTime'].map(lambda t: t.date())
		features = features.drop(['DateTime'], axis=1)
		self.df = pd.concat([self.df, features], axis=1)

	def add_datetime_feature(self):
		features = self.fullframe[['DateTime']]
		features = features.astype('datetime64[ns]')
		self.df = pd.concat([self.df, features], axis=1)


	##########################################################################################
	# ADD SMOOTHING
	##########################################################################################
	def add_bin_features(self):
		# hour bins
		features = self.fullframe[['hourbin_0-3', 'hourbin_4-7', 'hourbin_8-11',
								   'hourbin_12-15', 'hourbin_16-19', 'hourbin_20-23']]
		features = features.astype(int)
		self.df = pd.concat([self.df, features], axis=1)

		# add season dummies
		features = self.fullframe[['season_Winter', 'season_Fall', 'season_Summer', 'season_Spring']]
		features = features.astype(int)
		self.df = pd.concat([self.df, features], axis=1)


	##########################################################################################
	# ADD PCA
	##########################################################################################
	def add_interaction_features(self, pca_perc=0.8, return_pca_terms=False):
		# create a list of features to use in interactions
		columns = list(self.df.columns.values)

		# create dataframe consisting of the product of all combinations of terms
		df_interaction_terms = pd.DataFrame()
		combinations = itertools.combinations(columns, 2)
		for i in combinations:
			if (i[0] in ('Date', 'consumption', 'DateTime')) or \
					(i[1] in ('Date', 'consumption', 'DateTime')):
				pass
			else:
				colname = 'interaction_' + str(i[0]) + '_' + str(i[1])
				df_interaction_terms[colname] = self.df[i[0]] * self.df[i[1]]

		# pca
		pca = PCA()
		pca.fit(df_interaction_terms)
		df_interaction_terms_pca = pca.transform(df_interaction_terms)

		if return_pca_terms == True:
			return df_interaction_terms_pca
		else:
			# plot scree plot
			fig = plt.figure()
			ax = fig.add_subplot(111)
			n_components_range = range(1, df_interaction_terms.shape[1]+1)
			ax.plot(n_components_range, pca.explained_variance_ratio_)
			ax.set_ylabel('Explained Variance')
			ax.set_xlabel('Component #')
			ax.set_title('Screeplot for PCA on Interaction Terms')
			ax.set_xlim(0,200)
			plt.savefig(CHARTS_PATH + 'pca_screeplot_interaction_terms.png')

			# select components explained X% of the variance
			num_components = 0
			explained_variance = 0
			for v in pca.explained_variance_ratio_:
				num_components += 1
				explained_variance += v
				if explained_variance > pca_perc:
					break
			print 'Keeping %s components' % str(num_components)
			df_interaction_terms_pca = df_interaction_terms_pca[:,:num_components]
			df_interaction_terms_pca = pd.DataFrame(df_interaction_terms_pca)
			print df_interaction_terms_pca.shape, self.df.shape
			self.df = pd.concat([self.df, pd.DataFrame(df_interaction_terms_pca)], axis=1)


	##########################################################################################
	# OTHER
	##########################################################################################
	def add_kde_features(self, num_vectors=100):
		kde = KernelDensity()
		kde.fit(self.df)
		df_kde = pd.DataFrame(kde.sample(num_vectors))
		print df_kde
		df_kde.columns = self.df.columns
		df_kde_start_index = self.df.index.values[-1] + 1
		df_kde_index = range(df_kde_start_index, df_kde_start_index+df_kde.shape[0])
		df_kde.index = df_kde_index
		self.df = pd.concat([self.df, df_kde])

	def add_polynomials(self, degree=2):
		for f in self.df.columns:
			try:
				if self.df[f].dtype == float:
					if f in ['consumption']:
						pass
					else:
						feature_name = f + '_poly' + str(degree)
						self.df[feature_name] = self.df[f]**degree
			except:
				'Error in adding polynomials, possible duplicate column? %s' % str(f)


	##########################################################################################
	# WORK WITH DATA FRAME
	##########################################################################################
	def clear_current_frame(self):
		self.df = self.fullframe[['Electricity_KW_SQFT']]

	def set_to_new_frame(self, new_frame):
		self.df = new_frame

	def add_any_feature(self, feature_name, feature_type):
		features = self.fullframe[[feature_name]]
		features = features.astype(feature_type)
		self.df = pd.concat([self.df, features], axis=1)


	##########################################################################################
	# MAIN (LOAD DATA)
	##########################################################################################
	def load_data(self,
				  from_file=False,
				  normalization='zscore',
				  dummies=True,
				  bin_features=False,
				  rolling_means=False,
				  delta_features=False,
				  save_to_csv=False):
		if from_file != False:
			try:
				self.df = pd.DataFrame.from_csv(from_file)
			except:
				print 'Bad path!  Data did not load!'
		else:
			self.__init__()

			# add continuous features
			if normalization == 'zscore':
				self.add_zscore_continuous_features()
			elif normalization == 'minmax':
				self.add_minmax_continuous_features()
			elif normalization == 'none':
				self.add_continuous_features()
			else:
				print 'No continuous features were added!'
				pass

			# add categorical features
			if dummies == True:
				self.add_dummy_features()

			if bin_features == True:
				self.add_bin_features()


			# add rolling means
			if rolling_means == True:
				self.add_all_rolling_means()

			# add derivative features
			if delta_features == True:
				self.add_all_delta_features()

			# save to csv
			if save_to_csv == True:
				pass

			# return the dataframe
			return self.df




##############################################################################################
# DATA EXPLORATION
##############################################################################################
class data_explorer():
	def descriptive_statistics(self):
		descriptive_statistics = self.df.describe()
		descriptive_statistics.to_csv(TABLES_PATH + 'descriptive_statistics.csv')


	def plot_scatter_continuous_features(self):
		dm = data_manipulator()
		dm.add_continuous_features()
		dm.add_any_feature('Month', object)
		dm.add_any_feature('Day', object)
		df_plot = dm.df
		df_plot.drop('Date', axis=1, inplace=True)
		df_plot = df_plot.groupby(by=['Month','Day']).mean()
		print df_plot.columns

		for c in df_plot.columns:
		    if c == 'consumption':
		        pass
		    else:
		    	print c
		        fig = plt.figure(figsize=(18,9))
		        ax = fig.add_subplot(111)
		        df_plot.plot(kind='scatter', x=c, y='consumption', ax=ax)
		        ax.set_ylim(0.002,.0042)
		        ax.set_title('Scatterplot of Consumption And %s (Mean For Hour & Day)' % c)
	        	plt.savefig(CHARTS_PATH + 'scatter_%s_consumption.png' % c)


	def plot_plot_electricity_over_calendar_year(self):
		dm = data_manipulator()
		dm.add_any_feature('Sector', object)

		fig = plt.figure(figsize=(18,9))
		ax = fig.add_subplot(111)
		color_counter = 0
		for s in SECTORS:
			print s
			df_plot = dm.df[dm.df.Sector == s]
			df_plot.drop('Sector', axis=1, inplace=True)
			df_plot = df_plot.groupby('Date').mean()
			print df_plot.index.shape
			print df_plot.consumption.shape
			ax.scatter(x=df_plot.index, y=df_plot.consumption,
					c=COLOR_SCHEME[color_counter], label=s)
			color_counter += 1

		df_plot = dm.df.groupby('Date').mean()
		ax.plot(df_plot.index, df_plot.consumption,
				linewidth=5,
				color='orange',
				label='All Sectors')

		ax.set_xlabel('Day Of Year')
		ax.set_ylabel('Electricity KW SQFT')
		ax.set_title('Consumption over Calendar Year')
		ax.set_ylim([0,0.012])
		ax.legend(loc='best')
		plt.savefig(CHARTS_PATH + 'plot_consumption_over_calendar_year.png')
		plt.clf()


	def plot_histogram_consumption(self):
		# retrieve dataset
		dm = data_manipulator()
		dm.add_any_feature('Sector', object)
		df_plot = dm.df

		# open plot
		fig = plt.figure(figsize=(18,9))

		#### GRAPH 1
		ax = fig.add_subplot(121)
		# plot consumption all sectors
		y, binEdges = np.histogram(df_plot.consumption, bins=100)
		y = y / float(np.sum(y))
		bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
		ax.plot(bincenters,y,'-',color='orange', linewidth=2.5, label='All Sectors')

		# plot for each sector
		color_counter = 0
		for s in SECTORS:
			df_plot_sector = df_plot[df_plot.Sector == s]
			y, binEdges = np.histogram(df_plot_sector.consumption, bins=100)
			y = y / float(np.sum(y))
			bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
			ax.plot(bincenters,y,'-',
					color=COLOR_SCHEME[color_counter],
					label=s)
			color_counter += 1

		# pretty up graph
		ax.set_xlabel('Mean Consumption For Bin')
		ax.set_ylabel('Frequency (%)')
		ax.set_title('Distribution Of Consumption (Histogram Curve)')
		ax.legend(loc='best')

		#### GRAPH 2
		ax = fig.add_subplot(122)
		# plot consumption all sectors
		y, binEdges = np.histogram(df_plot.consumption, bins=100)
		y = y / float(np.sum(y))
		bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
		ax.plot(bincenters,y,'-',color='orange', linewidth=5, label='All Sectors')

		# plot for each sector
		color_counter = 0
		for s in SECTORS:
			df_plot_sector = df_plot[df_plot.Sector == s]
			y, binEdges = np.histogram(df_plot_sector.consumption, bins=100)
			y = y / float(np.sum(y))
			bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
			ax.plot(bincenters,y,'-',
					color=COLOR_SCHEME[color_counter],
					label=s)
			color_counter += 1

		# pretty up graph
		ax.set_xlabel('Mean Consumption For Bin')
		ax.set_ylabel('Frequency (%)')
		ax.set_title('Distribution Of Consumption (Histogram Curve) - ZOOM')
		ax.set_xlim([0,0.002])
		ax.set_ylim([0,0.15])
		ax.legend(loc='best')


		plt.savefig(CHARTS_PATH + 'hist_consumption.png')
		plt.clf()


	def plot_histogram_of_continuous_features(self):
		dm = data_manipulator()
		dm.add_continuous_features()
		df_plot = dm.df
		df_plot.drop(['consumption', 'Date'], axis=1, inplace=True)

		for c in df_plot.columns:
			print c
			df_plot_feature = df_plot[[c]]

			# plot
			fig = plt.figure()
			ax = fig.add_subplot(111)
			df_plot_feature.plot(kind='hist', bins=25)
			ax.set_title('Histogram of %s' % c)
			plt.savefig(CHARTS_PATH + 'hist_%s.png' % c)
			plt.clf()




		# retrieve dataset
		# dm = data_manipulator()
		# dm.add_continuous_features()
		# df_plot = dm.df
		# df_plot.drop(['consumption', 'Date', 'Visibility', 'Solar_Elevation'], axis=1)

		# # open plot
		# fig = plt.figure(figsize=(27,9))

		# # plot
		# plot_counter = 0
		# for c in df_plot.columns:
		# 	plot_counter += 1
		# 	y, binEdges = np.histogram(df_plot[c], bins=25)
		# 	bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
		# 	subplot_number = 230 + plot_counter
		# 	ax = fig.add_subplot(subplot_number)
		# 	ax.hist(bincenters, y)


		# # save fig
		# plt.show()
		# plt.clf()

		# ax1 = fig.add_subplot()

		#### GRAPH 1	



##############################################################################################
# MODEL ANALYIZER
##############################################################################################
class model_analyzer():
	def __init__(self, df, m, xtrain=None, ytrain=None, xtest=None, ytest=None):
		self.df = df
		self.model_name = m[0]
		self.clf = m[1]
		try:
			self.xtrain = xtrain
			self.ytrain = ytrain
			self.xtest = xtest
			self.ytest = ytest
		except:
			pass

	def set_model(self, model):
		self.model_name = model[0]
		self.clf = model[1]

	def partition_dataset(self, train_perc=0.8, override_train=False, override_test=False):
		# split train and test
		train_size = int(self.df.shape[0] * train_perc)
		train_ind = random.sample(self.df.index,train_size)
		if type(override_train) != bool:
			self.train = override_train
		else:
			self.train = self.df.loc[train_ind,:]
		if type(override_test) != bool:
			self.test = override_test
		else:
			self.test = self.df.drop(train_ind)

		# split x and y
		self.xtrain = self.train.drop(['consumption'], axis=1)
		self.ytrain = self.train.consumption
		self.xtest = self.test.drop(['consumption'], axis=1)
		self.ytest = self.test.consumption
		self.xtrain = self.xtrain.drop(['Date'], axis=1)
		self.xtest = self.xtest.drop(['Date'], axis=1)
		self.x = pd.concat([self.xtrain, self.xtest], axis=0)
		self.y = pd.concat([self.ytrain, self.ytest], axis=0)

	def optimize_model(self, params):
		# perform grid search to find the optimized model
		grid_search = GridSearchCV(estimator=self.clf,
									   cv=10,
									   scoring='r2',
									   param_grid=params)

		grid_search.fit(self.x, self.y)

		# assign the optimized model the current model
		self.clf = grid_search.best_estimator_

		# print the results to a dataframe
		list_configs = []
		list_validation_scores = []
		for config in grid_search.grid_scores_:
			list_configs.append(config[0])
			list_validation_scores.append(config[1])
		self.df_configs = pd.DataFrame(list_configs)
		self.df_configs['validation_score'] = list_validation_scores
		self.df_configs.to_csv(TABLES_PATH + 'optimize_%s.csv' % self.model_name)

		# print results
		print 'The model has been optimized...\n%s\n\n' % str(self.clf)
		print '-'*50

	def fit_model(self):
		# fit the model and make predictions
		print 'model....'
		print self.clf
		print self.xtrain.columns
		self.xtrain.to_csv(TABLES_PATH + 'x_train_%s.csv' % self.model_name)
		self.ytrain.to_csv(TABLES_PATH + 'y_train_%s.csv' % self.model_name)
		self.m = self.clf.fit(self.xtrain, self.ytrain)
		self.predtrain = self.m.predict(self.xtrain)
		self.predtest = self.m.predict(self.xtest)
		self.errorstrain = self.ytrain - self.predtrain
		self.errorstest = self.ytest - self.predtest

		# create data frame for analysis (train)
		dm = data_manipulator()
		dm.set_to_new_frame(self.xtrain)
		self.df_errorstrain = dm.df
		self.df_errorstrain['pred'] = self.predtrain
		self.df_errorstrain['actual'] = self.ytrain
		self.df_errorstrain['error'] = self.errorstrain
		self.df_errorstrain['squared_error'] = self.df_errorstrain['error'].apply(lambda x: x*x)

		# create data frame for analysis (test)
		dm = data_manipulator()
		dm.set_to_new_frame(self.xtest)
		self.df_errorstest = dm.df
		self.df_errorstest['pred'] = self.predtest
		self.df_errorstest['actual'] = self.ytest
		self.df_errorstest['error'] = self.errorstest
		self.df_errorstest['squared_error'] = self.df_errorstest['error'].apply(lambda x: x*x)

	def evaluate(self):
		results_list = []

		# initialize evaluation data frames
		df_eval_train = self.df_errorstrain
		df_eval_test = self.df_errorstest

		# calculate metrics and print results for all sectors
		mae_train = metrics.mean_absolute_error(df_eval_train.actual, df_eval_train.pred)
		mse_train = metrics.mean_squared_error(df_eval_train.actual, df_eval_train.pred)
		nmse_train = (mse_train / (np.sum(df_eval_train.actual**2) / self.xtrain.shape[0]))*100
		rmse_train = np.sqrt(mse_train)
		r_squared_train = metrics.r2_score(df_eval_train.actual, df_eval_train.pred)*100

		mae_test = metrics.mean_absolute_error(df_eval_test.actual, df_eval_test.pred)
		mse_test = metrics.mean_squared_error(df_eval_test.actual, df_eval_test.pred)
		nmse_test = (mse_train / (np.sum(df_eval_test.actual**2) / self.xtrain.shape[0]))*100
		rmse_test = np.sqrt(mse_test)
		r_squared_test = metrics.r2_score(df_eval_test.actual, df_eval_test.pred)*100

		# turn the results into df
		results_list.append([mae_train,
							 mae_test,
							 mse_train,
							 mse_test,
							 nmse_train,
							 nmse_test,
							 rmse_train,
							 rmse_test,
							 r_squared_train,
							 r_squared_test])
		df_results = pd.DataFrame(results_list)
		df_results.columns = ['mae_train',
							 'mae_test',
							 'mse_train',
							 'mse_test',
							 'nmse_train',
							 'nmse_test',
							 'rmse_train',
							 'rmse_test',
							 'r_squared_train',
							 'r_squared_test']
		print 'RESULTS: %s' % self.model_name
		print df_results
		self.evaluation_metrics = df_results

		df_results.to_csv(TABLES_PATH + 'results_%s.csv' % self.model_name)

	def evaluate_over_feature(self, feature, feature_type):
		results_list = []		

		# create evaluation frames
		df_eval_train = self.df_errorstrain
		dm = data_manipulator()
		dm.set_to_new_frame(df_eval_train)
		dm.add_any_feature(feature, feature_type)
		df_eval_train = dm.df.dropna()

		df_eval_test = self.df_errorstest
		dm = data_manipulator()
		dm.set_to_new_frame(df_eval_test)
		dm.add_any_feature(feature, feature_type)
		df_eval_test = dm.df.dropna()

		# calculate metrics and print results per sector
		for s in df_eval_train[feature].unique():
			df_eval_train_s = df_eval_train[df_eval_train[feature] == s]
			df_eval_test_s = df_eval_test[df_eval_test[feature] == s]

			# calculate evaluation metrics
			mae_train = metrics.mean_absolute_error(df_eval_train_s.actual, df_eval_train_s.pred)
			mse_train = metrics.mean_squared_error(df_eval_train_s.actual, df_eval_train_s.pred)
			nmse_train = (mse_train / (np.mean(df_eval_train_s.actual**2)))*100
			rmse_train = np.sqrt(mse_train)
			r_squared_train = metrics.r2_score(df_eval_train_s.actual, df_eval_train_s.pred)*100

			mae_test = metrics.mean_absolute_error(df_eval_test_s.actual, df_eval_test_s.pred)
			mse_test = metrics.mean_squared_error(df_eval_test_s.actual, df_eval_test_s.pred)
			nmse_test = (mse_test / (np.mean(df_eval_test_s.actual**2)))*100
			rmse_test = np.sqrt(mse_test)
			r_squared_test = metrics.r2_score(df_eval_test_s.actual, df_eval_test_s.pred)*100

			results_list.append([s,
								 mae_train,
								 mae_test,
								 mse_train,
								 mse_test,
								 nmse_train,
								 nmse_test,
								 rmse_train,
								 rmse_test,
								 r_squared_train,
								 r_squared_test])

		df_results = pd.DataFrame(results_list)
		df_results.columns = [feature,
							 'mae_train',
							 'mae_test',
							 'mse_train',
							 'mse_test',
							 'nmse_train',
							 'nmse_test',
							 'rmse_train',
							 'rmse_test',
							 'r_squared_train',
							 'r_squared_test']

		df_results.to_csv(TABLES_PATH + 'results_%s_over_%s.csv' % (self.model_name, feature))

		# plot results
		df_results.index = df_results[feature]
		df_results = df_results.drop([feature], axis=1)
		df_results = df_results[['nmse_train', 'nmse_test']]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		df_results.plot(kind='bar', ax=ax)
		ax.set_title('Model Performance Over %s - %s' % (feature, self.model_name))
		ax.set_xlabel('%s' % feature)
		ax.set_ylabel('Normalized Mean Squared Error (%)')
		plt.xticks(rotation=40)
		plt.subplots_adjust(bottom=.3)
		plt.savefig(CHARTS_PATH + '%s_evaluation_over_%s' % (self.model_name, feature))

	def feature_importance(self):
		fi = self.m.feature_importances_
		attr = self.xtrain.drop(['pred', 'actual', 'error', 'squared_error'], axis=1).columns.values
		df_fi = pd.DataFrame(list(fi), attr)
		df_fi.to_csv(TABLES_PATH + 'feature_importance_%s' % self.model_name)

	# plots
	def plot_optimization_two_features(self, f1, f2):
		if f1 not in self.df_configs.columns:
			print 'Error: no %s in parameters!' % f1
			return
		if f2 not in self.df_configs.columns:
			print 'Error: no %s in parameters!' % f2
			return

		fig = plt.figure()
		ax = fig.add_subplot(111)
		for f2_value in self.df_configs[f2].unique():
			df_plot = self.df_configs[[f1, f2, 'validation_score']]
			df_plot = df_plot[df_plot[f2] == f2_value]
			df_plot = df_plot.groupby(by=[f1]).mean()
			ax.plot(df_plot.index, df_plot.validation_score,
					label='%s = %s' % (f2, f2_value),
					marker='s')

		ax.set_title('Optimal Performance - %s & %s - %s' % (f1, f2, self.model_name))
		ax.set_xlabel(f1)
		ax.set_ylabel('5 Fold Cross Validation Score - Rsquared')
		plt.legend(loc='best')
		plt.savefig(CHARTS_PATH + 'optimize_%s_%s_%s.png' % (self.model_name, f1, f2))
		plt.clf()	

	def plot_actual_vs_pred(self):
		# add sector to df_errorstest
		dm = data_manipulator()
		dm.set_to_new_frame(self.df_errorstest)
		dm.add_any_feature('Sector', object)
		df_errorstest = dm.df
		df_errorstest = df_errorstest.dropna()


		fig = plt.figure()
		ax = fig.add_subplot(111)
		color_counter = 0
		for s in SECTORS:
			df_plot = df_errorstest[df_errorstest.Sector == s]
			ax.scatter(x=df_plot['pred'], y=df_plot['actual'],
					   c=COLOR_SCHEME[color_counter], label=s)
			color_counter += 1
		ax.set_title('Actual vs. Predicted')
		ax.set_xlabel('Predicted Consumption')
		ax.set_ylabel('Actual Consumption')
		ax.set_xlim(left=0, right=0.02)
		ax.set_ylim([0, 0.02])
		ax.legend(loc='best')
		ax.legend(bbox_to_anchor=(1.1, 1.05))
		plt.savefig(CHARTS_PATH + '%s_actual_vs_pred.png' % self.model_name)
		plt.clf()

	def plot_residual_vs_actual(self):
		# add sector to df_errorstest
		dm = data_manipulator()
		dm.set_to_new_frame(self.df_errorstest)
		dm.add_any_feature('Sector', object)
		df_errorstest = dm.df
		df_errorstest = df_errorstest.dropna()

		# create plot
		fig = plt.figure()
		ax = fig.add_subplot(111)
		color_counter = 0
		for s in SECTORS:
			df_plot = df_errorstest[df_errorstest.Sector == s]
			ax.scatter(x=df_plot['actual'], y=df_plot['error'],
					   c=COLOR_SCHEME[color_counter], label=s)
			color_counter += 1
		ax.set_title('Residuals vs. Actual')
		ax.set_xlabel('Actual Consumption')
		ax.set_ylabel('Residual')
		ax.set_xlim(left=0, right=0.02)
		ax.set_ylim([-0.004, 0.004])
		ax.legend(loc='best')
		ax.legend(bbox_to_anchor=(1.1, 1.05))
		plt.savefig(CHARTS_PATH + '%s_residual_vs_actual.png' % self.model_name)
		plt.clf()

	def plot_residual_vs_pred(self):
		# add sector to df_errorstest
		dm = data_manipulator()
		dm.set_to_new_frame(self.df_errorstest)
		dm.add_any_feature('Sector', object)
		df_errorstest = dm.df
		df_errorstest = df_errorstest.dropna()

		# create plot
		fig = plt.figure()
		ax = fig.add_subplot(111)
		color_counter = 0
		for s in SECTORS:
			df_plot = df_errorstest[df_errorstest.Sector == s]
			ax.scatter(x=df_plot['pred'], y=df_plot['error'],
					   c=COLOR_SCHEME[color_counter], label=s)
			color_counter += 1
		ax.set_title('Residuals vs. Predicted')
		ax.set_xlabel('Predicted Consumption')
		ax.set_ylabel('Residual')
		ax.set_xlim(left=0, right=0.02)
		ax.set_ylim([-0.004, 0.004])
		ax.legend(loc='best')
		ax.legend(bbox_to_anchor=(1.1, 1.05))
		plt.savefig(CHARTS_PATH + '%s_residual_vs_pred.png' % self.model_name)
		plt.clf()

	def plot_qq(self):
		fig = plt.figure()
		errors = self.df_errorstest['error']
		stats.probplot(errors, dist='norm', plot=pylab)
		plt.title('Normal QQ Plot of Errors')
		plt.savefig(CHARTS_PATH + '%s_plot_qq.png' % self.model_name)
		plt.clf()

	# animation
	def animate(self, param, plot, iteration_range=range(1,25)):
		# define param, iteration range,
		for i in iteration_range:
			# change the model
			if param == 'max_depth':
				self.clf.set_params(max_depth=i)
			elif param == 'n_estimators':
				self.clf.set_params(n_estimators=i)
			else:
				print 'Bad param name!'
				return

			# fit and evaluate model
			self.model_name = self.model_name.split('_')[0] + '_animate_%s_%s' % (param, str(i))
			self.fit_model()
			self.evaluate()

			# plot
			if plot == 'residual_vs_actual':
				self.plot_residual_vs_actual()
			elif plot == 'residual_vs_pred':
				self.plot_residual_vs_pred()
			elif plot == 'actual_vs_pred':
				self.plot_actual_vs_pred()
			elif plot == 'qq':
				self.plot_qq()

		# turn into a movie
		search_string = 'charts/' + self.model_name.split('_')[0] + \
			'_animate_%s_%%01d_%s.png' % (param, plot)
		movie_name = 'movies/' + '%s_%s_%s.mp4' % (self.model_name.split('_')[0], param, plot)
		subprocess.call('ffmpeg -f image2 -r 4 -i %s -vcodec mpeg4 -y %s' % (search_string, movie_name), shell=True)

		# move image files to movie directory
		identifier_string = '%s_animate_%s*' % (self.model_name.split('_')[0], param)
		charts_files = glob.glob('charts/' + identifier_string)
		subprocess.call('mkdir movies/%s' % self.model_name.split('_')[0], shell=True)
		for f in charts_files:
			subprocess.call('mv %s movies/%s/%s' % (f, self.model_name.split('_')[0], f.split('charts/')[1]), shell=True)
		
		# trash tables that got created
		tables_files = glob.glob('tables/results_' + identifier_string)		
		for f in tables_files:
			os.remove(f)			

		# stitch 4 images togther



##############################################################################################
# DRIVER
##############################################################################################
def test_multiple_datasets(model):
	# initialize data maniuplator object to create datasets 
	dm = data_manipulator()

	# initialize a list to keep performance metrics
	results_list = []

	# zscore
	model_name = model_name0 + '_datasettest_zscore'
	model = (model_name, model[1])
	df_zscore = dm.load_data(normalization='zscore')
	ma = model_analyzer(df_zscore, model)
	ma.partition_dataset()
	ma.fit_model()
	ma.evaluate()
	print ma.evaluation_metrics
	print ma.evaluation_metrics.values

	# # minmax
	# model_name = model_name0 + '_datasettest_minmax'
	# model = (model_name, model[1])
	# df_zscore = dm.load_data(normalization='minmax')
	# ma = model_analyzer(df_zscore, model)
	# ma.partition_dataset()
	# ma.fit_model()
	# ma.evaluate()


	# df_minmax = dm.load_data(normalization='minmax')
	# df_bins = dm.load_data(bin_features=True)
	# df_rolling_means = dm.load_data(rolling_means=True)
	# df_deltas = dm.load_data(delta_features=True)
	# df_all = dm.load_data(bin_features=True, rolling_means=True, delta_features=True)



def run_model(df, model, optimized=False, plots=False, animate=False):
	# initialize model analyzer class
	ma = model_analyzer(df, model)

	# partition the dataset
	ma.partition_dataset()

	# optimize model
	if optimized == True:
		print '#'*50
		print 'Optimizing model...'
		# define parameter grid
		if model[0] == 'DecisionTreeRegressor':
			params = {'max_depth': [5,10,20,50],
					  'min_samples_leaf': [1,50,500],
					  'max_features':[0.5,0.8,'auto']}
		elif model[0] == 'RandomForestRegressor':
			params = {'n_estimators': [2,5,10,25],
					  'max_features': [0.8, 0.9, None]}
		elif model[0] == 'AdaBoostRegressor':
			params = {'n_estimators': [5,10,25],
					  'learning_rate': [0.5, 2, 3, 5]}
		else:
			print 'Bad model name!'
			print model[0]
			return

		# optimize model
		ma.optimize_model(params)

		# plot performance over two attributes
		if plots == True:
			if model[0] == 'DecisionTreeRegressor':
				ma.plot_optimization_two_features('max_depth', 'max_features')
				ma.plot_optimization_two_features('max_depth', 'min_samples_leaf')
			elif model[0] == 'RandomForestRegressor':
				ma.plot_optimization_two_features('n_estimators', 'max_depth')
				ma.plot_optimization_two_features('n_estimators', 'max_features')
			elif model[0] == 'AdaBoostRegressor':
				ma.plot_optimization_two_features('n_estimators', 'learning_rate')
		print '#'*50
		print '\n'

	# fit and evaluate model
	ma.fit_model()
	ma.evaluate()
	try:
		ma.feature_importance()
	except:
		pass
	print ma.evaluation_metrics

	# create plots
	if plots == True:
		ma.evaluate_over_feature('Sector', object)
		ma.plot_qq()
		ma.plot_residual_vs_pred()
		ma.plot_actual_vs_pred()
		ma.plot_residual_vs_actual()

	# create animations
	if animate == True:
		if model[0] in ('DecisionTreeRegressor', 'RandomForestRegressor', 'DTStack', 'RFStack'):
			ma.animate(param='max_depth', plot='residual_vs_actual')
		if model[0] in ('RandomForestRegressor', 'AdaBoostRegressor', 'RFStack', 'ABStack'):
			ma.animate(param='n_estimators', plot='residual_vs_actual', iteration_range=range(1,26))



def get_stack_frame(df, base_models):
	# initialize model analyzer object
	ma = model_analyzer(df, base_models[0])
	
	# partition dataset and grab train set
	ma.partition_dataset()
	df_stack_train = ma.train
	df_stack_test = ma.test

	# get the predictions for each base model and append to df
	for m in base_models:
		ma.set_model(m)
		ma.fit_model()
		colname = m[0] + '_pred'
		df_stack_train[colname] = ma.predtrain
		df_stack_test[colname] = ma.predtest

	# return train and test sets
	return df_stack_train, df_stack_test
	


def explore_data_driver():
	de = data_explorer()
	de.plot_histogram_of_continuous_features()


def main():
	# load datasets
	dm = data_manipulator()
	df = dm.load_data()

	######################################################################
	# DECISION TREE
	######################################################################	
	# define model
	m_dt = ('DecisionTreeRegressor', DecisionTreeRegressor(max_depth=25))

	# run model
	run_model(df, m_dt, optimized=True, plots=False, animate=False)

	# # # test multiple datasets
	# # test_multiple_datasets(m_dt)


	# ######################################################################
	# # RANDOM FOREST
	# ######################################################################	
	# define the model
	# m_rf = ('RandomForestRegressor', RandomForestRegressor())

	# # # run the model
	# # run_model(df, m_rf, optimized=True, plots=True, animate=False)


	# # ######################################################################
	# # # ADABOOST
	# # ######################################################################
	# # define model
	# m_ab = ('AdaBoostRegressor', AdaBoostRegressor(base_estimator=m_dt[1]))

	# # run model
	# run_model(df, m_ab, optimized=True, plots=True, animate=False)

	# # print optimization


	# ######################################################################
	# # STACKING
	# ######################################################################
	# # create stacked dataframe
	# models = [m_dt, m_rf, m_ab]
	# df_stack_train, df_stack_test = get_stack_frame(df, base_models=models)

	# # define models
	# m_dt_stack = ('DTStack', DecisionTreeRegressor(max_depth=25))
	# m_rf_stack = ('RFStack', RandomForestRegressor(n_estimators=25))
	# m_rf_stack = ('ABStack', AdaBoostRegressor(base_estimator=m_dt[1]))

	# # run model - dt
	# ma = model_analyzer(df_stack_train, m_dt_stack)
	# ma.partition_dataset(override_train=df_stack_train, override_test=df_stack_test)
	# ma.fit_model()
	# ma.evaluate()
	# # ma.animate(param='n_estimators', plot='residual_vs_actual', iteration_range=range(1,26))
	# ma.feature_importance()
	# print ma.evaluation_metrics

	# # run model - rf
	# ma = model_analyzer(df_stack_train, m_rf_stack)
	# ma.partition_dataset(override_train=df_stack_train, override_test=df_stack_test)
	# ma.fit_model()
	# ma.evaluate()
	# # ma.animate(param='n_estimators', plot='residual_vs_actual', iteration_range=range(1,26))
	# ma.feature_importance()
	# print ma.evaluation_metrics

	# # run model - ab
	# ma = model_analyzer(df_stack_train, m_rf_stack)
	# ma.partition_dataset(override_train=df_stack_train, override_test=df_stack_test)
	# ma.fit_model()
	# ma.evaluate()
	# # ma.animate(param='n_estimators', plot='residual_vs_actual', iteration_range=range(1,26))
	# ma.feature_importance()
	# print ma.evaluation_metrics




##############################################################################################
# TEST
##############################################################################################
class test():
	def __init__(self):
		pass

	def partition_data(self):
		train_perc = 0.8
		dm = data_manipulator()
		dm.load_data()
		train_size = int(dm.df.shape[0] * train_perc)
		train_ind = random.sample(dm.df.index, train_size)
		train = dm.df.loc[train_ind]
		test = dm.df.drop(train_ind)
		test_x = test.drop('consumption', axis=1)
		test_y = test.consumption
		train.to_csv(DATA_PATH + 'test_partition/train.csv')
		test_x.to_csv(DATA_PATH + 'test_partition/test_x.csv')
		test_y.to_csv(DATA_PATH + 'test_partition/test_y.csv')

	def load_data(self):
		self.train = pd.DataFrame.from_csv(DATA_PATH + 'test_partition/train.csv')
		# self.test = pd.DataFrame.from_csv(DATA_PATH + 'test_partition/test.csv')
		self.x_train = self.train.drop(['consumption', 'Date'], axis=1)
		self.y_train = self.train.consumption
		self.x_test = pd.DataFrame.from_csv(DATA_PATH + 'test_partition/test_x.csv')
		self.x_test = self.x_test.drop(['Date'], axis=1)
	
	def _get_stack_frame(self):
		df_stack_train = self.train
		df_stack_test = self.x_test.copy()

		# get the predictions for each base model and append to df
		for model in self.base_models:
			m = model[1]
			m.fit(self.x_train, self.y_train)
			pred_train = m.predict(self.x_train)
			pred_test = m.predict(self.x_test)
			colname = model[0] + '_pred'
			df_stack_train[colname] = pred_train
			df_stack_test[colname] = pred_test

		# return train and test sets
		return df_stack_train, df_stack_test
		

	def fit_model(self, model):
		m = model[1]
		m.fit(self.x_train, self.y_train)
		pred = m.predict(self.x_test)
		actual = pd.Series.from_csv(DATA_PATH + 'test_partition/test_y.csv')

		# calculate metrics and print results for all sectors
		mae = metrics.mean_absolute_error(actual, pred)
		mse = metrics.mean_squared_error(actual, pred)
		nmse = (mse / (np.sum(actual**2) / actual.shape[0]))*100
		rmse = np.sqrt(mse)
		r_squared = metrics.r2_score(actual, pred)*100
	
		return [mae, mse, nmse, rmse, r_squared]

	def fit_stack(self):
		m_dt = ('DecisionTreeRegressor', DecisionTreeRegressor(max_depth=25))
		m_rf = ('RandomForestRegressor', RandomForestRegressor())
		m_ab = ('AdaBoostRegressor', AdaBoostRegressor(base_estimator=m_dt[1]))
		self.base_models = [m_ab]
		df_stack_train, df_stack_test = self._get_stack_frame()

		stack_model = AdaBoostRegressor(base_estimator=m_dt[1])
		stack_model.fit(df_stack_train.drop(['consumption', 'Date'], axis=1), df_stack_train.consumption)
		pred_train = stack_model.predict(df_stack_train.drop(['consumption', 'Date'], axis=1))
		actual_train = df_stack_train.consumption
		pred_test = stack_model.predict(df_stack_test)
		actual_test = pd.Series.from_csv(DATA_PATH + 'test_partition/test_y.csv')

		# calculate metrics and print results for all sectors
		# train
		mae_train = metrics.mean_absolute_error(actual_train, pred_train)
		mse_train = metrics.mean_squared_error(actual_train, pred_train)
		nmse_train = (mse_train / (np.sum(actual_train**2) / actual_train.shape[0]))*100
		rmse_train = np.sqrt(mse_train)
		r_squared_train = metrics.r2_score(actual_train, pred_train)*100

		# test
		mae_test = metrics.mean_absolute_error(actual_test, pred_test)
		mse_test = metrics.mean_squared_error(actual_test, pred_test)
		nmse_test = (mse_test / (np.sum(actual_test**2) / actual_test.shape[0]))*100
		rmse_test = np.sqrt(mse_test)
		r_squared_test = metrics.r2_score(actual_test, pred_test)*100

		return [mae_train, mae_test,
				mse_train, mse_test,
				nmse_train, nmse_test,
				rmse_train, rmse_test,
				r_squared_train, r_squared_test]

	def mean_confidence_interval(self, data, confidence=0.95):
	    a = 1.0*np.array(data)
	    n = len(a)
	    m, se = np.mean(a), stats.sem(a)
	    h = se * stats.t._ppf((1+confidence)/2., n-1)
	    return [m-h, m, m+h]


	def main(self, n_iterations=1):
		# define models (stack model definied in fit_stack method)
		m_dt = ('DecisionTreeRegressor', DecisionTreeRegressor(max_depth=25))
		m_rf = ('RandomForestRegressor', RandomForestRegressor())
		m_ab = ('AdaBoostRegressor', AdaBoostRegressor(base_estimator=m_dt[1]))

		results_dt = []
		results_rf = []
		results_ab = []
		results_stack = []
		for i in range(n_iterations):
			# run all models
			results_dt.append(self.fit_model(m_dt))
			results_rf.append(self.fit_model(m_rf))
			results_ab.append(self.fit_model(m_ab))
			results_stack.append(self.fit_stack())

		# turn the results into a dataframe
		df_results_dt = pd.DataFrame(results_dt)
		df_results_dt.columns = ['mae_dt_train', 'mae_dt_test',
								 'mse_dt_train', 'mse_dt_test',
								 'nmse_dt_train', 'nmse_dt_test',
								 'rmse_dt_train', 'rmse_dt_test',
								 'r_squared_dt_train', 'r_squared_dt_test']
		df_results_rf = pd.DataFrame(results_rf)
		df_results_rf.columns = ['mae_rf_train', 'mae_rf_test',
								 'mse_rf_train', 'mse_rf_test',
								 'nmse_rf_train', 'nmse_rf_test',
								 'rmse_rf_train', 'rmse_rf_test',
								 'r_squared_rf_train', 'r_squared_rf_test']
		df_results_ab = pd.DataFrame(results_ab)
		df_results_ab.columns = ['mae_ab_train', 'mae_ab_test',
								 'mse_ab_train', 'mse_ab_test',
								 'nmse_ab_train', 'nmse_ab_test',
								 'rmse_ab_train', 'rmse_ab_test',
								 'r_squared_ab_train', 'r_squared_ab_test']
		df_results_stack = pd.DataFrame(results_stack)
		df_results_stack.columns = ['mae_stack_train', 'mae_stack_test',
								 	'mse_stack_train', 'mse_stack_test',
								 	'nmse_stack_train', 'nmse_stack_test',
								 	'rmse_stack_train', 'rmse_stack_test',
								 	'r_squared_stack_train', 'r_squared_stack_test']

		# merge into single dataframe
		df_results_all = pd.concat([df_results_dt, df_results_rf, df_results_ab, df_results_stack], axis=1)

		# write the results out to csv
		df_results_all.to_csv(TABLES_PATH + 'test_results_all.csv')

		# find the confidence interval
		confidence_intervals = {}
		for c in df_results_all.columns:
			ci = self.mean_confidence_interval(df_results_all[c], confidence=0.95)
			confidence_intervals[c] = ci
		print confidence_intervals
		df_confidence_intervals = pd.DataFrame.from_dict(confidence_intervals)
		df_confidence_intervals.index = ['lower_bound', 'mean', 'upper_bound']
		print df_confidence_intervals
		df_confidence_intervals.reindex_axis(sorted(df_confidence_intervals.columns), axis=1)
		print df_confidence_intervals
		df_confidence_intervals.to_csv(TABLES_PATH + 'confidence_intervals.csv')




def run_test():
	t = test()
	t.partition_data()
	t.load_data()
	t.main(n_iterations=10)

# explore_data_driver()
main()
# run_test()

# # to do
# create chart of adaboost at different learn rates
# create feature importance for different models
# test stack with different adaboost learn rates
# finish introduction
# edit the data preprocessing section
# throw in charts to data exploration section
# write up methodology section
	# decision tree
	# random forest
	# adaboost
	# stacking
# write results and analysis
	# table with the results (5 fold cv, 80%/20% confidence intervals)
	# discussion of decision tree performance
		# discussion of optimization
		# analysis of residuals
		# plot of decision tree at different max depths
	# random forest
		# optimization
		# analysis of residuals
		# plot as number of estimators is increased
	# plot of adaboost with different estimators
# host link to videos



