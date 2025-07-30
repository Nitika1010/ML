import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pickle

# Load and preprocess data
@st.cache_data
def load_data_and_encode():
    df_original = pd.read_csv('Global YouTube Statistics.csv', encoding='latin1')

    # Data cleaning for original df (for filtering)
    df_original = df_original.dropna(subset=['Country', 'category', 'subscribers', 'uploads', 'video views'])

    # Convert earnings columns to numeric
    earnings_cols = ['lowest_monthly_earnings', 'highest_monthly_earnings', 
                    'lowest_yearly_earnings', 'highest_yearly_earnings']
    for col in earnings_cols:
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')

    # Calculate average earnings
    df_original['avg_monthly_earnings'] = (df_original['lowest_monthly_earnings'].fillna(0) + 
                                         df_original['highest_monthly_earnings'].fillna(0)) / 2
    df_original['avg_yearly_earnings'] = (df_original['lowest_yearly_earnings'].fillna(0) + 
                                         df_original['highest_yearly_earnings'].fillna(0)) / 2

    # Fix channel age calculation
    # First ensure all date components are numeric
    for col in ['created_year', 'created_month', 'created_date']:
        df_original[col] = pd.to_numeric(df_original[col], errors='coerce')
    
    # Handle missing/incorrect dates
    current_year = datetime.now().year
    df_original['created_year'] = df_original['created_year'].fillna(current_year).clip(lower=2005, upper=current_year)
    df_original['created_month'] = df_original['created_month'].fillna(1).clip(1, 12)
    df_original['created_date'] = df_original['created_date'].fillna(1).clip(1, 31)
    
    # Create datetime objects safely
    df_original['created_date_dt'] = pd.to_datetime(
        df_original['created_year'].astype(int).astype(str) + '-' +
        df_original['created_month'].astype(int).astype(str) + '-' +
        df_original['created_date'].astype(int).astype(str),
        errors='coerce'
    )
    
    # Calculate age in years properly (only for valid dates)
    valid_dates_mask = df_original['created_date_dt'].notna()
    df_original.loc[valid_dates_mask, 'age_years'] = (
        (datetime.now() - df_original.loc[valid_dates_mask, 'created_date_dt']).dt.days / 365.25
    )
    df_original['age_years'] = df_original['age_years'].fillna(0).clip(0)  # Ensure no negative ages
    
    # Calculate uploads per year safely
    df_original['uploads_per_year'] = np.where(
        df_original['age_years'] > 0,
        df_original['uploads'] / df_original['age_years'],
        np.nan  # Use NaN instead of 0 for invalid cases
    )

    # Store unique countries and categories for filter dropdowns
    all_countries = df_original['Country'].unique()
    all_categories = df_original['category'].unique()

    # Create a copy for encoding for the model
    df_encoded = df_original.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['Country', 'category'], drop_first=True, dtype=int)

    return df_original, df_encoded, all_countries, all_categories

df_original, df_encoded, all_countries, all_categories = load_data_and_encode()

# Streamlit app
st.title('Global YouTube Statistics Analysis')
st.write("""
This interactive dashboard analyzes global YouTube statistics, visualizes key metrics,
and predicts channel earnings using machine learning.
""")

# Sidebar filters
st.sidebar.header('Filters')
selected_countries = st.sidebar.multiselect(
    'Select Countries',
    all_countries,
    default=['United States', 'India', 'Brazil', 'United Kingdom']
)
selected_categories = st.sidebar.multiselect(
    'Select Categories',
    all_categories,
    default=['Music', 'Entertainment', 'Education', 'Gaming']
)
min_subscribers = st.sidebar.slider(
    'Minimum Subscribers (millions)',
    min_value=0,
    max_value=300,
    value=20
)

# Filter data using the original dataframe
filtered_df = df_original[
    (df_original['Country'].isin(selected_countries)) &
    (df_original['category'].isin(selected_categories)) &
    (df_original['subscribers'] >= min_subscribers * 1e6)
]

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Visualizations",
    "Category Analysis",
    "Country Analysis",
    "Earnings Prediction"
])

with tab1:
    st.header("Key Visualizations")

    # Visualization 1: Subscribers vs Video Views by Country
    st.subheader("1. Subscribers vs Video Views by Country")
    if not filtered_df.empty:
        fig1 = px.scatter(
            filtered_df,
            x='subscribers',
            y='video views',
            color='Country',
            hover_name='Youtuber',
            log_x=True,
            log_y=True,
            size='uploads',
            size_max=30,
            title='Subscribers vs Video Views (Log Scale)',
            labels={
                'subscribers': 'Subscribers',
                'video views': 'Video Views'
            },
            template='plotly_white'
        )
        fig1.update_layout(
            hovermode="closest",
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20
        )
        st.plotly_chart(fig1)
    else:
        st.info("No data available for the selected filters to display this chart.")

    # Visualization 2: Earnings Distribution by Category
    st.subheader("2. Earnings Distribution by Category")
    if not filtered_df.empty:
        fig2 = px.box(
            filtered_df,
            x='category',
            y='avg_yearly_earnings',
            color='category',
            log_y=True,
            title='Yearly Earnings Distribution by Category (Log Scale)',
            labels={
                'category': 'Category',
                'avg_yearly_earnings': 'Average Yearly Earnings (USD)'
            },
            template='plotly_white'
        )
        fig2.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20
        )
        st.plotly_chart(fig2)
    else:
        st.info("No data available for the selected filters to display this chart.")

    # Visualization 3: Channel Age vs Subscribers
    st.subheader("3. Channel Age vs Subscribers")
    if not filtered_df.empty:
        # Filter out channels with invalid ages
        age_filtered_df = filtered_df[(filtered_df['age_years'] > 0) & (filtered_df['age_years'].notna())]
        
        if not age_filtered_df.empty:
            fig3 = px.scatter(
                age_filtered_df,
                x='age_years',
                y='subscribers',
                color='Country',
                trendline="lowess",
                hover_name='Youtuber',
                title='Channel Age vs Subscribers',
                labels={
                    'age_years': 'Channel Age (Years)',
                    'subscribers': 'Subscribers'
                },
                template='plotly_white'
            )
            fig3.update_layout(
                hovermode="closest",
                margin=dict(l=20, r=20, t=50, b=20),
                title_font_size=20
            )
            st.plotly_chart(fig3)
        else:
            st.info("No channels with valid age data available for the selected filters.")
    else:
        st.info("No data available for the selected filters to display this chart.")

    # Visualization 4: Upload Frequency vs Earnings
    st.subheader("4. Upload Frequency vs Earnings")
    if not filtered_df.empty:
        upload_freq_df = filtered_df[(filtered_df['uploads_per_year'] > 0) & 
                                   (filtered_df['uploads_per_year'].notna())]
        
        if not upload_freq_df.empty:
            fig4 = px.scatter(
                upload_freq_df,
                x='uploads_per_year',
                y='avg_yearly_earnings',
                color='category',
                log_x=True,
                log_y=True,
                trendline="ols",
                hover_name='Youtuber',
                title='Upload Frequency vs Yearly Earnings (Log Scale)',
                labels={
                    'uploads_per_year': 'Uploads per Year',
                    'avg_yearly_earnings': 'Average Yearly Earnings (USD)'
                },
                template='plotly_white'
            )
            fig4.update_layout(
                hovermode="closest",
                margin=dict(l=20, r=20, t=50, b=20),
                title_font_size=20
            )
            st.plotly_chart(fig4)
        else:
            st.info("No channels with valid upload frequency data available.")
    else:
        st.info("No data available for the selected filters to display this chart.")

    # Visualization 5: Correlation Heatmap
    st.subheader("5. Feature Correlation Heatmap")
    # Select numerical features and drop NA
    numeric_features = ['subscribers', 'video views', 'uploads', 'avg_yearly_earnings', 'age_years']
    numeric_df = df_original[numeric_features].dropna()
    
    # Filter out invalid ages
    numeric_df = numeric_df[numeric_df['age_years'] > 0]
    
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        # Check for constant columns
        cols_to_include = [col for col in numeric_df.columns if numeric_df[col].nunique() > 1]
        
        if len(cols_to_include) > 1:
            corr = numeric_df[cols_to_include].corr()
            
            fig5 = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title='Feature Correlation Heatmap',
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                template='plotly_white'
            )
            fig5.update_layout(
                height=600,
                margin=dict(l=20, r=20, t=50, b=20),
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                title_font_size=20
            )
            fig5.update_xaxes(tickangle=45)
            st.plotly_chart(fig5)
        else:
            st.warning("Not enough varying numeric features to display correlation heatmap.")
    else:
        st.warning("Not enough numeric data for correlation analysis.")

    # Visualization 6: Top Channels by Subscribers
    st.subheader("6. Top Channels by Subscribers")
    if not filtered_df.empty:
        top_channels = filtered_df.nlargest(10, 'subscribers')
        fig6 = px.bar(
            top_channels,
            x='Youtuber',
            y='subscribers',
            color='Country',
            title='Top 10 Channels by Subscribers',
            labels={
                'subscribers': 'Subscribers (millions)',
                'Youtuber': 'Channel Name'
            },
            template='plotly_white'
        )
        fig6.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20,
            xaxis_tickangle=-45
        )
        fig6.update_yaxes(tickprefix="", tickformat=".2s")
        st.plotly_chart(fig6)
    else:
        st.info("No data available for the selected filters to display this chart.")

    # Visualization 7: Earnings vs Video Views with Regression
    st.subheader("7. Earnings vs Video Views with Regression")
    if not filtered_df.empty:
        fig7 = px.scatter(
            filtered_df,
            x='video views',
            y='avg_yearly_earnings',
            color='category',
            trendline="ols",
            log_x=True,
            log_y=True,
            hover_name='Youtuber',
            title='Video Views vs Yearly Earnings with Regression (Log Scale)',
            labels={
                'video views': 'Total Video Views',
                'avg_yearly_earnings': 'Average Yearly Earnings (USD)'
            },
            template='plotly_white'
        )
        fig7.update_layout(
            hovermode="closest",
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20
        )
        st.plotly_chart(fig7)
    else:
        st.info("No data available for the selected filters to display this chart.")

with tab2:
    st.header("Category Analysis")

    # Visualization 8: Category Distribution
    st.subheader("8. Category Distribution")
    category_counts = df_original['category'].value_counts().head(10)
    if not category_counts.empty:
        fig8 = px.pie(
            category_counts,
            names=category_counts.index,
            values=category_counts.values,
            title='Top 10 YouTube Category Distribution',
            template='plotly_white'
        )
        fig8.update_traces(textposition='inside', textinfo='percent+label')
        fig8.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20
        )
        st.plotly_chart(fig8)
    else:
        st.info("No category data available to display this chart.")

    # Visualization 9: Category vs Average Earnings
    st.subheader("9. Category vs Average Earnings")
    if not filtered_df.empty:
        avg_earnings = filtered_df.groupby('category')['avg_yearly_earnings'].mean().sort_values(ascending=False)
        fig9 = px.bar(
            avg_earnings,
            labels={'index': 'Category', 'value': 'Average Yearly Earnings (USD)'},
            color=avg_earnings.values,
            title='Average Yearly Earnings by Category',
            template='plotly_white'
        )
        fig9.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20,
            xaxis_tickangle=-45
        )
        fig9.update_yaxes(tickprefix="$", tickformat=".2s")
        st.plotly_chart(fig9)
    else:
        st.info("No data available for the selected filters to display this chart.")

    # Visualization 10: Category Growth Over Time
    st.subheader("10. Category Growth Over Time")
    if not filtered_df.empty:
        # Filter out invalid dates
        valid_dates_df = filtered_df[filtered_df['created_date_dt'].notna()]
        
        if not valid_dates_df.empty:
            # Group by year and category
            valid_dates_df['year'] = valid_dates_df['created_date_dt'].dt.year
            category_growth = valid_dates_df.groupby(['year', 'category']).size().reset_index(name='count')
            
            fig10 = px.line(
                category_growth,
                x='year',
                y='count',
                color='category',
                title='Category Growth Over Time',
                labels={
                    'year': 'Year',
                    'count': 'Number of Channels Created',
                    'category': 'Category'
                },
                template='plotly_white'
            )
            fig10.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                title_font_size=20
            )
            st.plotly_chart(fig10)
        else:
            st.info("No valid date data available to display this chart.")
    else:
        st.info("No data available for the selected filters to display this chart.")

with tab3:
    st.header("Country Analysis")

    # Visualization 11: Top Countries by Average Subscribers
    st.subheader("11. Top Countries by Average Subscribers")
    avg_subs = df_original.groupby('Country')['subscribers'].mean().sort_values(ascending=False).head(10)
    if not avg_subs.empty:
        fig11 = px.bar(
            avg_subs,
            labels={'index': 'Country', 'value': 'Average Subscribers'},
            color=avg_subs.values,
            title='Top 10 Countries by Average Subscribers',
            template='plotly_white'
        )
        fig11.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20
        )
        fig11.update_yaxes(tickprefix="", tickformat=".2s")
        st.plotly_chart(fig11)
    else:
        st.info("No country data available to display this chart.")

    # Visualization 12: Country vs Earnings with Regression
    st.subheader("12. Country vs Earnings with Regression")
    if not filtered_df.empty:
        fig12 = px.box(
            filtered_df,
            x='Country',
            y='avg_yearly_earnings',
            color='Country',
            log_y=True,
            title='Yearly Earnings by Country (Log Scale) with Regression',
            labels={
                'Country': 'Country',
                'avg_yearly_earnings': 'Average Yearly Earnings (USD)'
            },
            template='plotly_white'
        )
        
        # Add regression line
        for country in filtered_df['Country'].unique():
            country_df = filtered_df[filtered_df['Country'] == country]
            if len(country_df) > 1:
                fig12.add_trace(
                    go.Scatter(
                        x=[country, country],
                        y=[country_df['avg_yearly_earnings'].min(), 
                           country_df['avg_yearly_earnings'].max()],
                        mode='lines',
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=False
                    )
                )
        
        fig12.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig12)
    else:
        st.info("No data available for the selected filters to display this chart.")

    # Visualization 13: Geographic Distribution of Channels
    st.subheader("13. Geographic Distribution of Channels")
    if not filtered_df.empty:
        # Group by country and count channels
        country_counts = filtered_df['Country'].value_counts().reset_index()
        country_counts.columns = ['Country', 'count']
        
        # Get latitude and longitude from the original data
        geo_df = filtered_df[['Country', 'Latitude', 'Longitude']].drop_duplicates()
        country_counts = country_counts.merge(geo_df, on='Country', how='left')
        
        if not country_counts.empty and 'Latitude' in country_counts.columns and 'Longitude' in country_counts.columns:
            fig13 = px.scatter_geo(
                country_counts,
                lat='Latitude',
                lon='Longitude',
                size='count',
                color='Country',
                hover_name='Country',
                title='Geographic Distribution of YouTube Channels',
                template='plotly_white'
            )
            fig13.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                title_font_size=20
            )
            st.plotly_chart(fig13)
        else:
            st.info("Geographic data not available for the selected filters.")
    else:
        st.info("No data available for the selected filters to display this chart.")

with tab4:
    st.header("Earnings Prediction with Linear Regression")

    # Prepare data for modeling using the encoded dataframe
    model_df = df_encoded.dropna(subset=[
        'subscribers',
        'video views',
        'uploads',
        'avg_yearly_earnings',
        'age_years'
    ]).copy()

    # Define features (X) and target (y)
    numerical_features = ['subscribers', 'video views', 'uploads', 'age_years']
    country_cols = [col for col in model_df.columns if col.startswith('Country_')]
    category_cols = [col for col in model_df.columns if col.startswith('category_')]

    features = numerical_features + country_cols + category_cols
    features = [f for f in features if f in model_df.columns]

    if not model_df.empty and len(features) > 0:
        X = model_df[features]
        y = model_df['avg_yearly_earnings']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save model
        try:
            with open('linear_regression_model.pkl', 'wb') as file:
                pickle.dump(model, file)
            st.success("Model successfully trained and saved.")
        except Exception as e:
            st.error(f"Error saving model: {e}")

        # Model evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Model R-squared: {r2:.2f}")
        st.write(f"Mean Squared Error: {mse:.2e}")

        # Actual vs Predicted plot
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        }).sample(min(500, len(y_test)), random_state=42)

        fig = px.scatter(
            results_df,
            x='Actual',
            y='Predicted',
            trendline="lowess",
            title='Actual vs Predicted Earnings',
            template='plotly_white'
        )
        fig.add_shape(
            type="line",
            x0=min(y_test),
            y0=min(y_test),
            x1=max(y_test),
            y1=max(y_test),
            line=dict(color="Red", width=2, dash="dot")
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20
        )
        st.plotly_chart(fig)

        # Feature Importance
        st.subheader("Feature Importance")
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False).head(20)
        
        fig_coef = px.bar(
            coef_df,
            x='Feature',
            y='Coefficient',
            color='Coefficient',
            title='Top 20 Feature Coefficients',
            template='plotly_white'
        )
        fig_coef.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            title_font_size=20,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_coef)

# Add some insights
st.sidebar.header("Key Insights")
st.sidebar.write("""
- Music and Entertainment dominate YouTube channels
- India and US have the most top channels
- Channel age and upload frequency impact earnings
- Subscribers and views are highly correlated
- Upload frequency shows positive correlation with earnings
- Geographic distribution shows concentration in certain countries
- Linear regression can effectively predict earnings based on channel metrics
""")