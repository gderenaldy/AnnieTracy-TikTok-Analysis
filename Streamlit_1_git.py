import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split

plt.style.use('seaborn')

intro_container = st.container()
tiktok_container = st.container()

with intro_container:

	st.title("Annie Tracy TikTok EDA") 
	st.subheader("Performing EDA on Annie Tracy's TikTok data for potential insigths on digital strategy")
	st.divider()

with tiktok_container:
	
	st.markdown("### TikTok Active Followers And Social Analytics")
	tab1, tab2, tab3 = st.tabs(["Active Followers", "Social Analytics", "Predictive Model"])

	with tab1: 
		
		#aggregate active followers by date and hour
		@st.cache_data
		def load_data(Follower_activity):
			tk_data = pd.read_csv('Data/Follower activity.csv')
			return tk_data
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

		@st.cache_data
		def load_data_2(Annie_Tracy_TikTok_Social_Analytics):
			tk_social_data = pd.read_csv('Data/Annie Tracy_TikTok_Social_Analytics.csv')
			return tk_social_data
		tk_social_data = pd.read_csv('Data/Annie Tracy_TikTok_Social_Analytics.csv')

		@st.cache_data
		def load_data_3(New_RegressionData_upto_11_05):
			New_RegressionData = pd.read_csv('Data/New_RegressionData_upto_11.05.csv')
			return New_RegressionData 
		New_RegressionData = pd.read_csv('Data/New_RegressionData_upto_11.05.csv')


		tk_social_data = pd.concat([tk_social_data , New_RegressionData])
		tk_social_data_cleaned = tk_social_data.drop(tk_social_data.columns[0], axis =1) #remove date coloumn
		tk_social_data_cleaned = tk_social_data_cleaned.drop(range(0,7), axis =0).reset_index(drop = True) #remove first 6 rows because data is erroneous 

		fig2, ax3 = plt.subplots(figsize=(8, 10))
		fig2.set_facecolor("white")

		sns.heatmap(tk_social_data_cleaned.corr(), ax = ax3, cmap="YlGnBu", linewidths=.5, annot = True)
		plt.title('Heatmap of TikTok Social Analytics Data', fontsize=16)
		
		plt.tight_layout()
		st.pyplot(fig2)

		st.markdown("### **From: 07/10 - 11/05 (No 09/01 - 09/06)**")
		st.markdown("### Positive Correlations Insights: ")
		st.markdown('''It appears that there are no negative correlations between the variables in the dataset. All variables are positively correlated to some degree. 
					however, the slighlty less strong correlation with *Comments*, compared to *Likes* and *Shares*, might suggest that users are more likely to engage 
					with the Annie's TikToks through likes and shares more so than through comments. 
					In addition, there are **High** levels of multicolllinearity between variables. 
					''')
		st.divider()

		#boxplot to see outliers for each variable: 'Video Views', 'Profile Views', 'Likes', 'Comments', 'Shares'
		# -----boxplot-----

		fig3, (ax4 , ax5 , ax6 , ax7 , ax8) = plt.subplots(1,5, figsize=(20, 15))
		fig3.set_facecolor("white")
		
		sns.boxplot(y=tk_social_data_cleaned['Video Views'], ax=ax4, orient='Y', showmeans=True)
		sns.boxplot(y=tk_social_data_cleaned['Profile Views'], ax=ax5, orient='Y', showmeans=True)
		sns.boxplot(y=tk_social_data_cleaned['Likes'], ax=ax6, orient='Y', showmeans=True)
		sns.boxplot(y=tk_social_data_cleaned['Comments'], ax=ax7, orient='Y', showmeans=True)
		sns.boxplot(y=tk_social_data_cleaned['Shares'], ax=ax8, orient='Y', showmeans=True)

		ax4.set_title('Video Views')
		ax5.set_title('Profile Views')
		ax6.set_title('Likes')
		ax7.set_title('Comments')
		ax8.set_title('Shares')
		
		for ax in [ax4, ax5, ax6, ax7, ax8]:
			ax.ticklabel_format(style='plain', axis='y')
			ax.set_yscale('log')

		fig3.suptitle('Boxplot for "Viral" Days', fontsize=20)
		
		plt.tight_layout()
		st.pyplot(fig3)

		# create a dict of dicts with the column names as the keyword for each dict of statistics
		#stats = dict(tk_social_data_cleaned.columns, boxplot_stat(tk_social_data_cleaned))

		#st.write(stats)

		#filters dataset for "outliers"
		tk_social_data_filtered = tk_social_data[(tk_social_data['Video Views'] > 25000) & 
																(tk_social_data['Profile Views'] >= 8000) & 
																(tk_social_data['Likes'] > 30000) &
																(tk_social_data['Comments'] >= 700) & 
																(tk_social_data['Shares'] > 1000)]

		st.write(tk_social_data_filtered)

		st.markdown('### **Viral Content**')

		st.markdown('''By using boxplots, we can filter the dataset for those days when Annie's profile went ***viral***.
					Specifically, TikTok posts "Empire State of Mind (3.4M views) & "Original B&W" (1.6M views)" are behind her profile going viral the week of 7/10. 
					2 weeks later, TikTok's algorithm picked these posts up making Annie's profiile go "viral" once again on the week of 7/24.
					''')

		#filtering out the 'viral' posts to find the mean of each feature:

		viral_indices = [6, 7, 8, 9, 10, 11, 20 , 21, 22, 23, 24]

		tk_social_data_not_viral = tk_social_data.drop(viral_indices).reset_index()

		tk_social_data_not_viral_VideoViews = tk_social_data_not_viral['Video Views']
		tk_social_data_not_viral_ProfileViews = tk_social_data_not_viral['Profile Views']
		tk_social_data_not_viral_Likes = tk_social_data_not_viral['Likes']
		tk_social_data_not_viral_Comments = tk_social_data_not_viral['Comments']
		tk_social_data_not_viral_Shares = tk_social_data_not_viral['Shares']
		tk_social_data_not_viral_UniqueViewers = tk_social_data_not_viral['Unique Viewers']

		cl1, cl2, cl3, cl4, cl5, cl6 = st.columns (6)

		with cl1:
			st.markdown('##### *Video Views*: ')
			st.text(f'{tk_social_data_not_viral_VideoViews.mean(): .0f}')

		with cl2:
			st.markdown('##### *Profile Views*: ')
			st.text(f'{tk_social_data_not_viral_ProfileViews.mean(): .0f}')

		with cl3: 
			st.markdown('##### *Likes*: ')
			st.text(f'{tk_social_data_not_viral_Likes.mean(): .0f}')

		with cl4:
			st.markdown('##### *Comments*: ')
			st.text(f'{tk_social_data_not_viral_Comments.mean(): .0f}')

		with cl5:
			st.markdown('##### *Shares*: ')
			st.text(f'{tk_social_data_not_viral_Shares.mean(): .0f}')

		with cl6:
			st.markdown('##### *Unique Views*: ')
			st.text(f'{tk_social_data_not_viral_UniqueViewers.mean(): .0f}')

		st.markdown('''These are the average values for each feature in the dataset on a daily basis for Annie's TikTok profile. 
					Could be useful to compare these to the ***viral*** days above.''')

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

		st.write(tk_social_data_filtered.loc[:,'Engagement Rate':'Share-to-Like Ratio'])

		#compare engagement rate of viral posts to the engagement rate of non-viral ones
		
		tk_social_data_not_viral['Engagement Rate'] = (tk_social_data_not_viral['Likes'] + tk_social_data_not_viral['Comments'] + tk_social_data_not_viral['Shares']) / tk_social_data_not_viral['Video Views']
		
		mean_engage_viral = tk_social_data_filtered['Engagement Rate'].mean() * 100

		st.markdown(f'The Average Engagement Rate for Viral Posts is: {mean_engage_viral: .2f} %')

		mean_engage_not_viral = tk_social_data_not_viral['Engagement Rate'].mean() * 100
		
		st.markdown(f'The Average Engagement Rate for Non-Viral Posts is: {mean_engage_not_viral: .2f} %')

		st.markdown(f'''The {(mean_engage_viral - mean_engage_not_viral) / mean_engage_not_viral * 100: .2f} % increase in engagement rate for viral posts is a positive finding as 
					**viral** content with lower engagement rates than regular may suggest a broad yet non-targeted reach, and / or audience mismatch.''')

		st.divider()

	with tab3:

		#using the filtered dataset (not including viral posts), let's predict engagement rate based on all features provided
		tk_social_data_not_viral_cleaned = tk_social_data_not_viral.iloc[6:, 2:] #take off first 6 rows and first 'Date' column
	
		st.markdown('z-score normalization. **Profile Views** as outcome variable')

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

		z_RegressionSocialData.to_csv('z_RegressionSocialData.csv', index = False)
		
		st.write(z_RegressionSocialData)

		st.subheader('Lasso Regression')
		st.markdown('Leveraing *Lasso* regression due to high levels of multicollinearity between predictor variables.')

		#load data

		@st.cache_data
		def load_data_4(z_RegressionSocialData):
			tk_data_regression = pd.read_csv('Data/z_RegressionSocialData.csv')
			return tk_data_regression
		tk_data_regression = pd.read_csv('Data/z_RegressionSocialData.csv')

		#split testing and training sets

		x = tk_data_regression.drop('Profile Views', axis =1)
		y = tk_data_regression['Profile Views']
		x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 123)

		# There's no need to reshape x_train for multiple regression 
		x_train_p = np.array(x_train)

		# Convert y_train to a 1D array
		y_train_p = np.array(y_train).ravel()

		formula = r'''
		$$
		\text{Profile Views} = 2020 + (1545 \times \text{Video Views}) + (545 \times \text{Likes}) + (252 \times \text{Comments}) + (84 \times \text{Shares}) + (98 \times \text{Unique Viewers}) + \epsilon
		$$
		'''
		st.markdown(formula)

		with st.expander('Code'):

			st.text(

				'''def compute_cost_lasso(x_train, y_train, w, b, lambda_): 
					m = x_train.shape[0]  # Number of training examples
					cost_sum = 0
					for i in range(m): 
						y_pred = np.dot(x_train.iloc[i], w) + b
						cost = (y_pred - y_train.iloc[i]) ** 2  
						cost_sum += cost
						total_cost = (1 / (2 * m)) * cost_sum + (lambda_ / (2 * m)) * np.sum(np.abs(w))
					return total_cost

				def compute_gradient_lasso(x_train, y_train, w, b, lambda_):
					m = x_train.shape[0]
					dj_dbdj_dw = np.zeros(w.shape)
					dj_db = 0
					for i in range(m):  
						y_pred = np.dot(x_train.iloc[i], w) + b
						dj_dw_i = (y_pred - y_train.iloc[i]) * x_train.iloc[i].values.reshape(-1,1)
						dj_db_i = y_pred - y_train.iloc[i]
						dj_db += dj_db_i 
						dj_dw += dj_dw_i 
					dj_dw = dj_dw / m + lambda_ * np.sign(w) / m
					dj_db = dj_db / m
					return dj_dw, dj_db

				def gradient_descent_lasso(x_train, y_train, w_init, b_init, alpha, iterations, lambda_, compute_cost_lasso, compute_gradient_lasso):
					b = b_init 
					w = w_init
					for i in range(iterations): 
						dj_dw, dj_db = compute_gradient_lasso(x_train, y_train, w, b, lambda_)
						b = b - alpha * dj_db
						w = w - alpha * dj_dw
					return w, b

				w_init = np.zeros((5, 1))
				b_init = 0

				lambda_ = 10
				iterations = 10000
				tmp_alpha = 0.001
				w_final, b_final = gradient_descent_lasso(x_train, y_train, w_init, b_init, tmp_alpha, iterations, lambda_, compute_cost_lasso, compute_gradient_lasso)
				print(w_final.round(0), b_final.round(0))

				def compute_rmse(y_test, y_pred):
					rmse = np.mean((y_test - y_pred) ** 2) ** 0.5
					return rmse

				y_pred = np.dot(x_test, w_final) + b_final
				y_pred = y_pred.ravel()
				print (y_pred.round(2))
				rmse = compute_rmse(y_test, y_pred)
				print (rmse.round(0))''')
			
		st.divider()

		st.markdown('### Optimizing Digital Strategy')
		
		st.markdown('''The analysis reveals key insights into driving **profile views** by focusing mostly on raising total **Video Views** for Annie's TikTok account. 
					This variable has considerable positive impact on Annie's TikTok profile visibility.''') 

		st.markdown('''Notably, for every standard deviation increase in Video Views (90,477), there is an average increase of 1545 Profile Views. 
					This translates to a ratio of approximately 58.56 Video Views for a single Profile View. 
					In addition, one could focus on expanding engagement as well by boosting Likes and Comments with marginal overall returns.''')

		st.markdown('##### **ATTENTION**: ')
		st.markdown('''Due to the predictors being higly correlated to eachoter, results may be misleading. 
					This is also supported by the residulats **NOT** being normally distributed.''')

		st.image('Image/Histogram of Residuals.png')

#with st.sidebar: 

	#uploaded_file = st.sidebar.file_uploader("Please, choose a file") 

	#if uploaded_file:  #if a file is uploaded, then ...	
		#st.success("file uploaded succesfully")
		
		#st.markdown("### Descriptive Statistics")
		#df = pd.read_csv(uploaded_file)
		#st.write(df.describe())
		#st.markdown("### Null Values")
		#st.write(df.isnull().sum())

		#st.markdown("### Dataset Top 5 Rows")
		#st.write(df.head())

	#st.divider()



