"""
this file defines a table structure for a database using SQLAlchemy.
ig exists so that the app can store data in a database which in this case is the supabase 
"""

from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()

# these are all the 30 coumns
ALL_COLS = [
    "Datetime", "region", "MW",
    "hour", "day_of_week", "month", "quarter", "year", "is_weekend",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_lag_720h",
    "load_roll_mean_24h", "load_roll_std_24h", "load_roll_mean_168h", "load_roll_std_168h",
    "is_holiday", "is_month_start", "is_month_end",
    "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2",
]

# these are the features that the model actually uses
FEATURE_COLS = [
    "hour", "day_of_week", "month", "year", "is_weekend",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_lag_720h",
    "load_roll_mean_24h", "load_roll_std_24h", "load_roll_mean_168h",
    "is_holiday", "is_month_start", "is_month_end", "load_roll_std_168h",
    "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2",
]


class SmartGridLoad(Base):
    __tablename__ = "smart_grid_load"

    id = Column(Integer, primary_key = True, autoincrement=True)

    # identifiers
    Datetime = Column(DateTime, nullable = True)
    region = Column(String(20), nullable=True)

    MW = Column(Float, nullable = True)

    hour        = Column(Integer, nullable=True) 
    day_of_week = Column(Integer, nullable=True) 
    month       = Column(Integer, nullable=True) 
    quarter     = Column(Integer, nullable=True) 
    year        = Column(Integer, nullable=True)
    is_weekend  = Column(Integer, nullable=True) 


     # --- cyclical (sin/cos) encodings ---
    hour_sin        = Column(Float, nullable=True)
    hour_cos        = Column(Float, nullable=True)
    day_of_week_sin = Column(Float, nullable=True)
    day_of_week_cos = Column(Float, nullable=True)
    month_sin       = Column(Float, nullable=True)
    month_cos       = Column(Float, nullable=True)


    # --- lagged load (how load looked n hours ago) ---
    load_lag_1h   = Column(Float, nullable=True)
    load_lag_24h  = Column(Float, nullable=True)
    load_lag_168h = Column(Float, nullable=True)
    load_lag_720h = Column(Float, nullable=True)


    # --- rolling statistics of load ---
    load_roll_mean_24h  = Column(Float, nullable=True)
    load_roll_std_24h   = Column(Float, nullable=True)
    load_roll_mean_168h = Column(Float, nullable=True)
    load_roll_std_168h  = Column(Float, nullable=True)


     # --- calendar flags / weather-ish flags ---
    is_holiday     = Column(Integer, nullable=True) 
    is_month_start = Column(Integer, nullable=True) 
    is_month_end   = Column(Integer, nullable=True) 


    # --- Fourier terms (daily/weekly seasonality) ---
    fourier_sin_1 = Column(Float, nullable=True)
    fourier_cos_1 = Column(Float, nullable=True)
    fourier_sin_2 = Column(Float, nullable=True)
    fourier_cos_2 = Column(Float, nullable=True)

def create_tables(engine):
    """Create all declared tables in the database."""
    Base.metadata.create_all(engine)
    print("[db] Tables created")




