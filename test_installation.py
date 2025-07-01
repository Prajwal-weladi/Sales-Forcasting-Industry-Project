#!/usr/bin/env python3
"""
Test script to verify all dependencies for the Advanced Analytics Dashboard
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'pandas',
        'numpy', 
        'streamlit',
        'plotly',
        'sklearn',
        'statsmodels'
    ]
    
    print("Testing package imports...")
    print("=" * 50)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - FAILED: {e}")
            failed_imports.append(package)
    
    print("=" * 50)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements_advanced.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_data_loading():
    """Test if the data file can be loaded"""
    try:
        import pandas as pd
        print("\nTesting data loading...")
        print("=" * 50)
        
        # Try to load the data
        df = pd.read_csv('all_data.csv')
        print(f"✅ Data loaded successfully!")
        print(f"   - Rows: {len(df):,}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check required columns
        required_columns = [
            'order_purchase_timestamp',
            'customer_city', 
            'product_category_name_english',
            'total_price',
            'price',
            'product_weight_g',
            'order_id'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present!")
            return True
            
    except FileNotFoundError:
        print("❌ Data file 'all_data.csv' not found!")
        print("Please ensure the data file is in the backend directory.")
        return False
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_subcategory_creation():
    """Test the subcategory creation function"""
    try:
        import pandas as pd
        print("\nTesting subcategory creation...")
        print("=" * 50)
        
        # Load a small sample of data
        df = pd.read_csv('all_data.csv', nrows=1000)
        
        # Test the create_subcategory function
        def create_subcategory(row):
            category = row['product_category_name_english']
            weight = row['product_weight_g']
            price = row['price']
            
            if category == 'health_beauty':
                if weight < 500:
                    return 'Light Beauty Products'
                elif weight < 1000:
                    return 'Medium Beauty Products'
                else:
                    return 'Heavy Beauty Products'
            elif category == 'fashion_shoes':
                if price < 50:
                    return 'Budget Shoes'
                elif price < 150:
                    return 'Mid-range Shoes'
                else:
                    return 'Premium Shoes'
            elif category == 'sports_leisure':
                if weight < 1000:
                    return 'Light Sports Equipment'
                else:
                    return 'Heavy Sports Equipment'
            elif category == 'baby':
                if price < 30:
                    return 'Budget Baby Products'
                else:
                    return 'Premium Baby Products'
            else:
                if price < 25:
                    return 'Budget'
                elif price < 75:
                    return 'Mid-range'
                else:
                    return 'Premium'
        
        # Apply the function
        df['product_subcategory'] = df.apply(create_subcategory, axis=1)
        
        print(f"✅ Subcategories created successfully!")
        print(f"   - Unique subcategories: {df['product_subcategory'].nunique()}")
        print(f"   - Sample subcategories: {df['product_subcategory'].unique()[:5].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating subcategories: {e}")
        return False

def main():
    """Run all tests"""
    print("Advanced Analytics Dashboard - Installation Test")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data loading
    data_ok = test_data_loading()
    
    # Test subcategory creation
    subcategory_ok = test_subcategory_creation()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if imports_ok and data_ok and subcategory_ok:
        print("✅ ALL TESTS PASSED!")
        print("\n🎉 Your environment is ready to run the Advanced Analytics Dashboard!")
        print("\nTo start the dashboard, run:")
        print("streamlit run advanced_analytics_dashboard.py")
    else:
        print("❌ SOME TESTS FAILED!")
        print("\nPlease fix the issues above before running the dashboard.")
        
        if not imports_ok:
            print("\nTo install missing packages:")
            print("pip install -r requirements_advanced.txt")
        
        if not data_ok:
            print("\nTo fix data issues:")
            print("- Ensure all_data.csv is in the backend directory")
            print("- Check that the CSV file has all required columns")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 