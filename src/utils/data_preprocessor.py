"""
Data preprocessing utility for lottery data
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import os

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess lottery data for training"""
    
    def __init__(self):
        self.required_columns = ['numbers', 'bonus_ball']  # Removed date requirement
    
    def load_and_validate_csv(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Load CSV file and validate format"""
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            
            # Check required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate and clean data
            errors = []
            cleaned_df = df.copy()
            
            # Process each row
            for idx, row in df.iterrows():
                row_errors = self._validate_row(row, idx + 1)
                errors.extend(row_errors)
                
                # Clean numbers format
                if 'numbers' in row and pd.notna(row['numbers']):
                    numbers_str = str(row['numbers'])
                    try:
                        # Handle different number formats
                        if ',' in numbers_str:
                            numbers = [int(x.strip()) for x in numbers_str.split(',')]
                        else:
                            # Try to split by spaces or other delimiters
                            numbers = [int(x.strip()) for x in numbers_str.split() if x.strip().isdigit()]
                        
                        if len(numbers) != 6:
                            errors.append(f"Row {idx + 1}: Expected 6 numbers, got {len(numbers)}")
                        else:
                            # Validate number range (1-49)
                            invalid_numbers = [n for n in numbers if n < 1 or n > 49]
                            if invalid_numbers:
                                errors.append(f"Row {idx + 1}: Invalid numbers {invalid_numbers} (must be 1-49)")
                            
                            # Check for duplicates
                            if len(set(numbers)) != 6:
                                errors.append(f"Row {idx + 1}: Duplicate numbers found")
                            
                            # Sort numbers and store as string
                            cleaned_df.at[idx, 'numbers'] = ','.join(map(str, sorted(numbers)))
                    
                    except ValueError as e:
                        errors.append(f"Row {idx + 1}: Invalid number format '{numbers_str}'")
                
                # Clean bonus ball
                if 'bonus_ball' in row and pd.notna(row['bonus_ball']):
                    try:
                        bonus = int(row['bonus_ball'])
                        if bonus < 1 or bonus > 49:
                            errors.append(f"Row {idx + 1}: Invalid bonus ball {bonus} (must be 1-49)")
                        cleaned_df.at[idx, 'bonus_ball'] = bonus
                    except ValueError:
                        errors.append(f"Row {idx + 1}: Invalid bonus ball format '{row['bonus_ball']}'")
            
            # Add sequential dates if no date column exists
            if 'date' not in cleaned_df.columns:
                logger.info("No date column found, adding sequential dates starting from 01/10/97...")
                start_date = datetime(1997, 1, 10)  # Wednesday
                dates = []
                for i in range(len(cleaned_df)):
                    # Alternate between Wednesday and Saturday
                    if i % 2 == 0:
                        # Wednesday (add 0 days)
                        draw_date = start_date + timedelta(days=i * 3)
                    else:
                        # Saturday (add 3 days from Wednesday)
                        draw_date = start_date + timedelta(days=(i * 3) + 3)
                    dates.append(draw_date.strftime('%Y-%m-%d'))
                
                cleaned_df['date'] = dates
                logger.info(f"Added sequential dates from {dates[0]} to {dates[-1]}")
            
            # Sort by date (oldest first)
            if 'date' in cleaned_df.columns:
                cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
                cleaned_df = cleaned_df.sort_values('date').reset_index(drop=True)
                logger.info(f"Sorted data chronologically from {cleaned_df['date'].min()} to {cleaned_df['date'].max()}")
            
            return cleaned_df, errors
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return pd.DataFrame(), [f"Failed to load file: {str(e)}"]
    
    def _validate_row(self, row: pd.Series, row_num: int) -> List[str]:
        """Validate a single row of data"""
        errors = []
        
        # Check for missing values
        for col in self.required_columns:
            if col not in row or pd.isna(row[col]):
                errors.append(f"Row {row_num}: Missing {col}")
        
        return errors
    
    def save_cleaned_data(self, df: pd.DataFrame, output_path: str):
        """Save cleaned data to CSV"""
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved cleaned data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving cleaned data: {e}")
    
    def create_sample_data(self, output_path: str = "data/cleaned_sample.csv"):
        """Create a sample of properly formatted data"""
        sample_data = {
            'numbers': [
                '3,12,25,31,38,45', '7,15,22,29,36,44', '2,11,19,27,35,43',
                '5,13,21,28,37,42', '1,9,17,24,33,41', '4,10,18,26,34,40',
                '6,14,20,30,39,46', '8,16,23,32,38,47', '2,11,19,25,35,43',
                '5,12,21,29,37,44'
            ],
            'bonus_ball': [7, 12, 8, 15, 6, 11, 9, 13, 7, 10]
        }
        
        df = pd.DataFrame(sample_data)
        self.save_cleaned_data(df, output_path)
        return df

def main():
    """Main function to preprocess data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess lottery data')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', default='data/cleaned_data.csv', help='Output file path')
    parser.add_argument('--create-sample', action='store_true', help='Create sample data instead')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    
    if args.create_sample:
        print("Creating sample data...")
        preprocessor.create_sample_data(args.output)
        print(f"Sample data created at {args.output}")
        return
    
    print(f"Processing {args.input_file}...")
    
    # Load and validate data
    df, errors = preprocessor.load_and_validate_csv(args.input_file)
    
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
        
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors")
        
        response = input("Continue with cleaning anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    if not df.empty:
        # Save cleaned data
        preprocessor.save_cleaned_data(df, args.output)
        print(f"Cleaned data saved to {args.output}")
        print(f"Processed {len(df)} rows")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        print("No valid data found")

if __name__ == "__main__":
    main() 