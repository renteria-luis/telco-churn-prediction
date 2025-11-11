import numpy as np
import pandas as pd

def build_features(df):
    """
    Build engineered features for the Telco Customer Churn dataset.
    This version excludes the target and is intended for use in modeling pipelines.
    """

    # ======================
    # Feature Engineering
    # ======================

    # Contract length in months
    contract_map = {'Month-to-Month': 1, 'One Year': 12, 'Two Year': 24}
    if 'contract' in df.columns:
        df['contract_length_months'] = df['contract'].map(contract_map)

    # Average monthly spending (avoid division by zero)
    if {'total_charges', 'tenure_in_months'}.issubset(df.columns):
        df['avg_monthly_spending'] = (
            df['total_charges'] / df['tenure_in_months'].replace(0, np.nan)
        ).round(2)

    # Net revenue after refunds
    if {'total_revenue', 'total_refunds'}.issubset(df.columns):
        df['revenue_minus_refunds'] = (
            df['total_revenue'] - df['total_refunds']
        ).round(2)

    # Any streaming service active
    streaming_cols = ['streaming_tv', 'streaming_movies', 'streaming_music']
    if all(col in df.columns for col in streaming_cols):
        df['has_streaming'] = df[streaming_cols].eq('Yes').any(axis=1).astype(int)

    # Total number of active services
    binary_services = [
        'phone_service', 'internet_service', 'online_security', 'online_backup',
        'device_protection_plan', 'premium_tech_support', 'streaming_tv',
        'streaming_movies', 'streaming_music'
    ]
    available = [col for col in binary_services if col in df.columns]
    df['num_services'] = df[available].eq('Yes').sum(axis=1)

    # ======================
    # Final feature selection
    # ======================

    cat_features = [
        'gender', 'offer', 'internet_service', 'internet_type', 'payment_method'
    ]

    num_features = [
        'age', 'number_of_dependents', 'number_of_referrals', 'tenure_in_months',
        'contract_length_months', 'monthly_charge', 'total_charges', 'total_refunds',
        'total_extra_data_charges', 'total_long_distance_charges', 'total_revenue',
        'avg_monthly_gb_download', 'avg_monthly_spending', 'revenue_minus_refunds',
        'num_services'
    ]

    binary_features = [
        'has_phone_service', 'has_multiple_lines', 'has_online_security', 'has_online_backup',
        'has_device_protection_plan', 'has_premium_tech_support', 'has_streaming_tv',
        'has_streaming_movies', 'has_streaming_music', 'has_unlimited_data',
        'has_paperless_billing', 'has_streaming'
    ]

    # Combine all selected features
    selected_features = cat_features + num_features + binary_features

    return df[selected_features]
