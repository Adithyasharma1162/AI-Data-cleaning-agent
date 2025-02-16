import pandas as pd
import argparse
from datetime import datetime, timedelta
import json
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class SchemaAnalyzer:
    """Handles dynamic column matching using LLM"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect_merge_keys(self, df1, df2):
        """Identify matching columns between two dataframes using LLM"""
        df1_info = self._get_column_info(df1)
        df2_info = self._get_column_info(df2)
        
        prompt = f"""Analyze these two datasets to find matching columns for merging:
        
        Dataset 1 (Bookings):
        {json.dumps(df1_info, indent=2)}
        
        Dataset 2 (Airlines):
        {json.dumps(df2_info, indent=2)}
        
        Identify which columns should be used for joining. Return JSON format:
        {{"df1_key": "column_name", "df2_key": "column_name", "reason": "..."}}"""
        
        response = self.llm([
            SystemMessage(content="You are a data schema expert. Respond with valid JSON only."),
            HumanMessage(content=prompt)
        ])
        
        result = json.loads(response.content)
        self._validate_keys(result, df1, df2)
        return result['df1_key'], result['df2_key']
    
    def _get_column_info(self, df):
        """Create column metadata with samples and statistics"""
        return {
            col: {
                "dtype": str(df[col].dtype),
                "sample_values": df[col].dropna().head(3).tolist(),
                "unique_count": df[col].nunique(),
                "null_count": df[col].isnull().sum()
            }
            for col in df.columns
        }
    
    def _validate_keys(self, result, df1, df2):
        """Ensure suggested keys exist in dataframes"""
        if result['df1_key'] not in df1.columns:
            raise ValueError(f"Invalid key for DF1: {result['df1_key']}")
        if result['df2_key'] not in df2.columns:
            raise ValueError(f"Invalid key for DF2: {result['df2_key']}")

class EnhancedFlightDataProcessor(FlightDataProcessor):
    """Extended with dynamic key detection"""
    
    def __init__(self, booking_path, airline_path, llm):
        super().__init__(booking_path, airline_path)
        self.schema_analyzer = SchemaAnalyzer(llm)
    
    def merge_data(self):
        """Dynamic merge based on detected keys"""
        booking_key, airline_key = self.schema_analyzer.detect_merge_keys(
            self.booking_df, 
            self.airline_df
        )
        
        merged = pd.merge(
            self.booking_df,
            self.airline_df,
            left_on=booking_key,
            right_on=airline_key,
            how='left',
            suffixes=('_booking', '_airline')
        )
        
        # Post-merge cleanup
        if booking_key != airline_key:
            merged = merged.drop(columns=[airline_key])
        merged = self._remove_duplicate_columns(merged)
        
        return merged
    
    def _remove_duplicate_columns(self, df):
        """Handle columns that might have been duplicated during merge"""
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [
                f"{dup}_booking" if i == 0 else f"{dup}_airline" 
                for i in range(sum(cols == dup))
            ]
        df.columns = cols
        return df


class FlightDataProcessor:
    """Handles data loading, merging, and preprocessing"""
    
    def __init__(self, booking_path, airline_path):
        self.booking_df = pd.read_csv(booking_path)
        self.airline_df = pd.read_csv(airline_path)
        
    def merge_data(self):
        """Merge booking data with airline mappings"""
        merged = pd.merge(
            self.booking_df,
            self.airline_df,
            on='airline_id',
            how='left'
        ).drop(columns=['airline_id'])
        return merged

class DataEnhancer:
    """Uses LLM to enhance data quality through automated cleaning"""
    
    def __init__(self, llm):
        self.llm = llm
        self.column_mapping = {}
        
    def _get_llm_response(self, prompt):
        """Helper method to get structured LLM response"""
        response = self.llm([
            SystemMessage(content="You are a data cleaning expert. Respond with valid JSON only."),
            HumanMessage(content=prompt)
        ])
        return json.loads(response.content)
    
    def standardize_columns(self, df):
        """Normalize column names using LLM suggestions"""
        prompt = f"""Analyze these columns: {list(df.columns)}. Suggest improved names following these rules:
        1. Use title case
        2. Remove special characters
        3. Make them descriptive
        4. Keep dates in 'YYYY-MM-DD' format
        Return JSON: {{"original": "improved"}}"""
        
        self.column_mapping = self._get_llm_response(prompt)
        return df.rename(columns=self.column_mapping)
    
    def clean_categorical(self, series):
        """Standardize categorical values using LLM"""
        samples = series.dropna().unique().tolist()[:10]
        prompt = f"""Standardize these {series.name} values: {samples}. 
        Group similar values and return JSON: {{"original": "standardized"}}"""
        
        mapping = self._get_llm_response(prompt)
        return series.map(mapping).fillna(series)
    
    def handle_missing_values(self, series):
        """Impute missing values using LLM-guided strategy"""
        null_count = series.isnull().sum()
        dtype = series.dtype
        samples = series.dropna().sample(min(5, len(series))).tolist()
        
        prompt = f"""Column '{series.name}' ({dtype}) has {null_count} missing values. 
        Sample values: {samples}. Suggest imputation method. Return JSON format:
        {{"strategy": "drop|mean|median|mode|custom", "value": "...", "reason": "..."}}"""
        
        solution = self._get_llm_response(prompt)
        
        if solution['strategy'] == 'drop':
            return series.dropna()
        elif solution['strategy'] == 'mean':
            return series.fillna(series.mean())
        elif solution['strategy'] == 'median':
            return series.fillna(series.median())
        elif solution['strategy'] == 'mode':
            return series.fillna(series.mode()[0])
        else:
            return series.fillna(solution.get('value', 'Unknown'))
    
    def enhance_dataset(self, df):
        """Full cleaning pipeline"""
        df = self.standardize_columns(df)
        
        # Clean categorical columns
        for col in df.select_dtypes(include='object'):
            df[col] = self.clean_categorical(df[col])
        
        # Handle missing values
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = self.handle_missing_values(df[col])
                
        return df

class FlightAnalyzer:
    """Performs business analysis on cleaned data"""
    
    def __init__(self, cleaned_df):
        self.df = cleaned_df
    
    def top_airlines(self, n=5):
        return self.df['Airline'].value_counts().nlargest(n).to_dict()
    
    def popular_destinations(self, n=3):
        return self.df['Destination'].value_counts().nlargest(n).to_dict()
    
    def cancellation_patterns(self):
        cancellations = self.df.groupby(['Airline', 'CancellationDate']).size()
        return cancellations[cancellations > 0].sort_values(ascending=False).to_dict()
    
    def occupancy_analysis(self):
        self.df['OccupancyRate'] = self.df['SeatsBooked'] / self.df['TotalSeats']
        return {
            'most_popular': self.df.loc[self.df['OccupancyRate'].idxmax()].to_dict(),
            'least_popular': self.df.loc[self.df['OccupancyRate'].idxmin()].to_dict()
        }

class CommandLineInterface:
    """Provides user interaction through command line"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.actions = {
            '1': ('Top Airlines', self.analyzer.top_airlines),
            '2': ('Popular Destinations', self.analyzer.popular_destinations),
            '3': ('Cancellation Patterns', self.analyzer.cancellation_patterns),
            '4': ('Occupancy Analysis', self.analyzer.occupancy_analysis)
        }
    
    def display_menu(self):
        print("\nFlight Data Analysis:")
        for key, (label, _) in self.actions.items():
            print(f"{key}. {label}")
        print("Q. Quit")
    
    def run(self):
        while True:
            self.display_menu()
            choice = input("Select analysis: ").strip().upper()
            
            if choice == 'Q':
                print("Exiting...")
                break
                
            if choice in self.actions:
                result = self.actions[choice][1]()
                print("\nResults:")
                print(json.dumps(result, indent=2))
            else:
                print("Invalid choice. Please try again.")

# Update main function to use enhanced processor
def main(booking_path, airline_path, openai_key):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_key)
    
    processor = EnhancedFlightDataProcessor(booking_path, airline_path, llm)
    raw_data = processor.merge_data()
    
    enhancer = DataEnhancer(llm)
    cleaned_data = enhancer.enhance_dataset(raw_data)
    
    analyzer = FlightAnalyzer(cleaned_data)
    cli = CommandLineInterface(analyzer)
    cli.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flight Data Analysis System")
    parser.add_argument("--bookings", required=True, help="Path to bookings CSV")
    parser.add_argument("--airlines", required=True, help="Path to airlines CSV")
    parser.add_argument("--openai-key", required=True, help="OpenAI API key")
    
    args = parser.parse_args()
    
    main(
        booking_path=args.bookings,
        airline_path=args.airlines,
        openai_key=args.openai_key
    )
