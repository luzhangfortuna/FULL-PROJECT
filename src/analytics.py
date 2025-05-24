import pandas as pd
from datetime import datetime

class M&AAnalytics:
    def __init__(self, df):
        self.df = df
        self._prepare_data()
    
    def _prepare_data(self):
        # Clean and prepare data
        self.df['Acquisition Price Numeric'] = pd.to_numeric(
            self.df['Acquisition Price'].replace('-', '0').str.replace(',', ''), 
            errors='coerce'
        )
        self.df['Year-Month'] = pd.to_datetime(
            self.df['Acquisition Year'].astype(str) + '-' + 
            self.df['Acquisition Month'].astype(str), 
            format='%Y-%b', 
            errors='coerce'
        )
    
    def get_company_acquisition_history(self, company_name):
        """Get all acquisitions by a specific company"""
        return self.df[self.df['Parent Company'].str.contains(company_name, case=False, na=False)]
    
    def get_top_acquirers(self, n=10):
        """Get top N companies by number of acquisitions"""
        return self.df['Parent Company'].value_counts().head(n)
    
    def get_largest_deals(self, n=10):
        """Get top N largest deals by value"""
        return self.df.nlargest(n, 'Acquisition Price Numeric')[
            ['Parent Company', 'Acquired Company', 'Acquisition Price', 'Acquisition Year']
        ]
    
    def get_acquisition_trends(self):
        """Analyze acquisition trends over time"""
        yearly_stats = self.df.groupby('Acquisition Year').agg({
            'ID': 'count',
            'Acquisition Price Numeric': ['sum', 'mean', 'median']
        })
        return yearly_stats
    
    def get_sector_analysis(self):
        """Analyze acquisitions by business sector"""
        return self.df['Business'].value_counts().head(20)
