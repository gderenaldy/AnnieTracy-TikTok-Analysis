import sys
print(sys.executable)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

plt.style.use('seaborn')

pd.set_option('display.float_format', '{:.2f}'.format)

intro_container = st.container()
tiktok_container = st.container()

with intro_container:

	st.title("Annie Tracy TikTok EDA") 
	st.subheader("Performing EDA on Annie Tracy's TikTok data for potential insigths to improve digital strategy")
	st.divider()

with tiktok_container:
	
	st.markdown("### TikTok Active Followers And Social Analytics")
	tab1, tab2, tab3 = st.tabs(["Active Followers", "Social Analytics", "Predictive Model"])

	with tab1: 
		
		#aggregate active followers by date and hour
		
		tk_data = pd.read_csv('Data/Follower activity.csv')
		tk_data['Date'] = pd.to_datetime(tk_data['Date'])
		tk_data['Hour'] = tk_data['Hour'].astype('int') 
		tk_data['Active followers'] = tk_data['Active followers'].astype('int')
		
		tk_data_aggregated_date = tk_data.groupby(['Date']).mean().reset_index() #find the mean active followers per day
		tk_data_aggregated_hour = tk_data.groupby(['Date','Hour']).mean().reset_index()	#find the mean active followers per hour

		#EDA for Active followers 

		fig, (ax1, ax2) = plt.subplots(1,2) #sets up 'fig' canvas and 'ax1', 'ax2' whcih are the graphs withing the canvas. 
		fig.set_facecolor("white")

		ax1.bar(tk_data_aggregated_date['Date'], tk_data_aggregated_date['Active followers'], width=0.7, edgecolor= "black", color = "green")
		ax1.set_title("Mean Active Followers by Day", fontdict = { "fontname" : "arial" , "fontsize" : 12 }).set_color("black")
		ax1.set_xlabel(' Date', fontdict = { "fontname" : "arial" , "fontsize" : 12 }).set_color("black")
		ax1.set_ylabel('Average', fontdict = { "fontname" : "arial" , "fontsize" : 12 }).set_color("black")
		ax1.set_ybound(lower = 0, upper = 30000)
		ax1.tick_params(axis = 'x', rotation = 45, color = "white")
		
		ax2.bar(tk_data_aggregated_hour['Hour'], tk_data_aggregated_hour['Active followers'], width=0.7, edgecolor= "black", color = "blue")
		ax2.set_title("Mean Active Followers by Hour", fontdict = { "fontname" : "arial" , "fontsize" : 12 }).set_color("black")
		ax2.set_xlabel('Hour', fontdict = { "fontname" : "arial" , "fontsize" : 12 }).set_color("black")
		ax2.set_ylabel('Average', fontdict = { "fontname" : "arial" , "fontsize" : 12 }).set_color("black")
		ax2.set_ybound(lower = 0, upper = 40000)
		ax2.tick_params(axis='x', rotation=45, color = "white")

		plt.tight_layout() #makes sure there is no overlapping between graphs
		st.pyplot(fig) #displays the canvas(fig) in web browser
		
		st.markdown("#### **Week of 08/30 - 09/05**")
		st.markdown('''Annie's active followers on Tiktok appear to be slighlty more active during the week and less so on the weekends.
					In addition, there is a noticeable dip in activity between the hours of 6 am and 6 pm. This may be due to the fact that people are busy working''')
	
	with tab2:
		
		#Checking for correlation between variables in the social dataset

		tk_social_data = pd.read_csv('Data/Annie Tracy_TikTok_Social_Analytics.csv')
		tk_social_data_cleaned = tk_social_data.drop(tk_social_data.columns[0], axis =1) #remove date coloumn
		tk_social_data_cleaned = tk_social_data_cleaned.drop(range(0,7), axis =0).reset_index(drop = True) #remove first 6 rows because data is erroneous 

		fig2, ax3 = plt.subplots(figsize=(8, 10))
		fig2.set_facecolor("white")

		sns.heatmap(tk_social_data_cleaned.corr(), ax = ax3, cmap="YlGnBu", linewidths=.5, annot = True)
		plt.title('Heatmap of TikTok Social Analytics Data', fontsize=16)
		
		plt.tight_layout()
		st.pyplot(fig2)

		st.markdown("### **Week of 08/23 - 08/29**")
		st.markdown("### Positive Correlations Insights: ")
		st.markdown('''It appears that there are no negative correlations between the variables in the dataset. All variables are positively correlated to some degree. 
		however, the slighlty less strong correlation with *Comments*, compared to *Likes* and *Shares*, might suggest that users are more likely to engage 
		with the Annie's TikToks through likes and shares more so than through comments. 
		In addition, there are **High** levels of multicolllinearity between variables. 
			''')
		st.divider()

		#boxplot to see outliers for each variable: 'Video Views', 'Profile Views', 'Likes', 'Comments', 'Shares', 'Unique Viewers'
		# -----boxplot-----

		fig3, (ax4 , ax5 , ax6 , ax7 , ax8) = plt.subplots(1,5, figsize=(15, 10))
		fig3.set_facecolor("white")
		
		sns.boxplot(y=tk_social_data_cleaned['Video Views'], ax=ax4)
		sns.boxplot(y=tk_social_data_cleaned['Profile Views'], ax=ax5)
		sns.boxplot(y=tk_social_data_cleaned['Likes'], ax=ax6)
		sns.boxplot(y=tk_social_data_cleaned['Comments'], ax=ax7)
		sns.boxplot(y=tk_social_data_cleaned['Shares'], ax=ax8)

		ax4.set_title('Video Views')
		ax5.set_title('Profile Views')
		ax6.set_title('Likes')
		ax7.set_title('Comments')
		ax8.set_title('Shares')
		
		for ax in [ax4, ax5, ax6, ax7, ax8]:
			ax.ticklabel_format(style='plain', axis='y')

		fig3.suptitle('Boxplot for Outliers', fontsize=20)
		
		plt.tight_layout()
		st.pyplot(fig3)

		#filters dataset for "outliers"
		tk_social_data_filtered = tk_social_data[(tk_social_data['Video Views'] > 35000) & 
																(tk_social_data['Profile Views'] >= 10000) & 
																(tk_social_data['Likes'] > 42000) &
																(tk_social_data['Comments'] >= 1400) & 
																(tk_social_data['Shares'] > 2500)]

		st.write(tk_social_data_filtered)

		st.markdown('### **Viral Content**')

		st.markdown('''By looking at the boxplots, we can filter the dataset for TikTok entries that could be considered ***viral***. 
					This could be due to the content itself, the timing of the post, or any current trends or viral challenges.
					This  may be useful for future reference if one wants to replicate similar content strategies for virality.''')

		#filtering out the 'viral' posts to find the mean of each feature:

		viral_indices = [6, 7, 8, 9, 11, 21, 22]

		tk_social_data_not_viral = tk_social_data.drop(viral_indices)

		cl1, cl2, cl3, cl4, cl5, cl6 = st.columns (6)

		with cl1:
			st.markdown('##### *Video Views*: ')
			st.text('84,743')

		with cl2:
			st.markdown('##### *Profile Views*: ')
			st.text('2,901')

		with cl3: 
			st.markdown('##### *Likes*: ')
			st.text('10,129')

		with cl4:
			st.markdown('##### *Comments*: ')
			st.text('275')

		with cl5:
			st.markdown('##### *Shares*: ')
			st.text('516')

		with cl6:
			st.markdown('##### *Unique Views*: ')
			st.text('51,082')

		st.markdown("These are the average values for each feature in the dataset. Could be useful to compare these to the ***viral*** posts above.")

		st.divider()

		#creating titktok-related social features for engagemet analysis

		st.markdown('### **Social Features for Viral Posts**')

		# proportion of interactions (likes, comments, shares) per video view
		tk_social_data_filtered['Engagement Rate'] = (tk_social_data_filtered['Likes'] + tk_social_data_filtered['Comments'] + tk_social_data_filtered['Shares']) / tk_social_data_filtered['Video Views']
		
		#Profile Viewer Rate (The ratio of profile views to video views)
		tk_social_data_filtered['Profile View Rate'] = ( tk_social_data_filtered['Profile Views'] / tk_social_data_filtered['Video Views'])

		#Unique Viewer Rate(The fraction of unique viewers per video view)

		tk_social_data_filtered['Unique Viewer Rate'] = ( tk_social_data_filtered['Unique Viewers'] / tk_social_data_filtered['Video Views'])

		# Like-to-Comment Ratio (viewers'ratio of likes to comments, highlighting whether viewers prefer to express appreciation (like) or engage in discussions (comment).
		# Adding 1 to the denominator to avoid division by zero

		tk_social_data_filtered['Like-to-Comment Ratio'] = (tk_social_data_filtered['Likes'] / (tk_social_data_filtered['Comments'] + 1))

		# Share-to-Like Ratio(viewers' propensity to share content they like)
		# Adding 1 to the denominator to avoid division by zero

		tk_social_data_filtered['Share-to-Like Ratio'] = tk_social_data_filtered['Shares'] / (tk_social_data_filtered['Likes'] + 1)

		st.write(tk_social_data_filtered.loc[:, 'Engagement Rate' :'Share-to-Like Ratio'])

		#compare engagement rate of viral posts to the engagement rate of non-viral ones
		
		tk_social_data_not_viral['Engagement Rate'] = (tk_social_data_not_viral['Likes'] + tk_social_data_not_viral['Comments'] + tk_social_data_not_viral['Shares']) / tk_social_data_not_viral['Video Views']
		
		mean_engage_viral = tk_social_data_filtered['Engagement Rate'].mean() * 100

		st.markdown(f'The Average Engagement Rate for Viral Posts is: {mean_engage_viral: .2f} %')

		mean_engage_not_viral = tk_social_data_not_viral['Engagement Rate'].mean() * 100
		
		st.markdown(f'The Average Engagement Rate for Non-Viral Posts is: {mean_engage_not_viral: .2f} %')

		st.markdown('''The 64 percent increase in engagement rate for viral posts is a positive finding as 
				#**viral** content with lower engagement rates than regular may suggest a broad yet non-targeted reach, and / or audience mismatch.''')

		st.divider()

	with tab3:

		#using the filtered dataset (not including viral posts), let's predict engagement rate based on all features provided

		tk_social_data_not_viral_cleaned = tk_social_data_not_viral.iloc[6:, 1:] #take off first 6 rows and first 'Date' column

		st.markdown('Post z-score normalization. **Profile Views** as outcome variable')

		#normalize features

		tk_social_data_not_viral_cleaned_noEngagement = tk_social_data_not_viral_cleaned.drop(columns = ['Profile Views', 'Engagement Rate'])

		def zscore_standardize(df):
			z_norm_tk_social_data_not_viral_cleaned = tk_social_data_not_viral_cleaned_noEngagement.copy()
			for i in z_norm_tk_social_data_not_viral_cleaned.columns:
				mean_val = z_norm_tk_social_data_not_viral_cleaned[i].mean()
				std_val = z_norm_tk_social_data_not_viral_cleaned[i].std()
				z_norm_tk_social_data_not_viral_cleaned[i] = (tk_social_data_not_viral_cleaned[i] - mean_val) / std_val
			return z_norm_tk_social_data_not_viral_cleaned

		z_normalized_social_data = zscore_standardize(tk_social_data_not_viral_cleaned_noEngagement)

		#merge on indexes
		z_RegressionSocialData = pd.merge(z_normalized_social_data, tk_social_data_not_viral_cleaned['Profile Views'], left_index = True, right_index = True)

		#z_RegressionSocialData.to_csv('z_RegressionSocialData.csv', index = False)

		#z_normalized_social_data.to_csv('z_normalized_social_data.csv', index=False)
		
		st.write(z_normalized_social_data)

		#build model

		tk_data = pd.read_csv('Data/z_RegressionSocialData.csv')
		x = tk_data.drop('Profile Views', axis =1)
		y = tk_data['Profile Views']
		x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 123)

		# There's no need to reshape x_train for multiple regression 
		x_train_p = np.array(x_train)

		# Convert y_train to a 1D array
		y_train_p = np.array(y_train).ravel()

		lasso_cv = LassoCV() #ls is the container for parameter m and c in y=mX+c

		lasso_cv.fit(x_train , y_train) #training part to get m and c

		c = lasso_cv.intercept_

		m = lasso_cv.coef_

		st.text('Use Lasso regression due to high levels of multicolllinearity between predictor variables.')

		lasso_cv.alpha_.round(2)

		st.text(f'Optimal Alpha:{lasso_cv.alpha_: .2f}')
		
		lasso_cv_coefficients = lasso_cv.coef_

		lasso_cv_coefficients.round(2)

		st.text(f' Intercept : {c: .0f}')

		y_pred_test = m*x_test + c #our model

		fig4 , (ax9, ax12) = plt.subplots(1,2,figsize=(15, 10), sharey = True)

		#random_indices = np.random.choice(len(x_test), 10, replace=False)
		#x_samples = x_test.iloc[random_indices]
		#y_pred_samples = y_pred_test.iloc[random_indices]

		#new x_samples dataframe with 5 columns and 10 random samples for each from the og x_test dataset

		ax9.scatter(x_test['Video Views'], y_test)
		ax9.scatter(x_test['Video Views'], y_pred_test['Video Views'], color = 'red')
		ax9.set_ylabel('Profile Views (test)')
		ax9.set_xlabel('10 Video Views Samples (test)')

		ax12.scatter(x_test['Shares'], y_test)
		ax12.scatter(x_test['Shares'], y_pred_test['Shares'], color = 'red')
		ax12.set_ylabel('Profile Views (test)')
		ax12.set_xlabel('10 Shares Samples (test)')

		plt.tight_layout()
		st.pyplot(fig4)

		#rmse_VideoViews = (mean_squared_error(y_test, y_pred_test['Video Views'])**0.5).round(0)

		#st.write(rmse_VideoViews)

		st.text('Standard deviations from original data:')
		st.write(tk_social_data_not_viral_cleaned[['Video Views','Shares']].std().round(0))

		st.markdown('### Optimizing Digital Strategy with Our Model')
		st.markdown('''Our analysis reveals key insights into driving profile views. 
					Central to this is the influence of two metrics: **Video Views** and **Shares**. 
					These factors have considerable positive impact on Annie's TikTok profile visibility. Notably, 
					For every standard deviation increase in **Video Views** (90,477), there's an average augmentation of 11,549 in **Profile views**. 
					This translates to a ratio of 7.83 **Video Views** to a single **Profile View**.''')

with st.sidebar: 

	uploaded_file = st.sidebar.file_uploader("Please, choose a file") 

	if uploaded_file:  #if a file is uploaded, then ...	
		st.success("file uploaded succesfully")
		
		st.markdown("### Descriptive Statistics")
		df = pd.read_csv(uploaded_file)
		st.write(df.describe())
		st.markdown("### Null Values")
		st.write(df.isnull().sum())

		st.markdown("### Dataset Top 5 Rows")
		st.write(df.head())

	st.divider()



