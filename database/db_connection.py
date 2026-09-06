import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# load environment variables from .env file
ENV_PATH = Path(__file__).resolve().parent.parent/ '.env'
load_dotenv(dotenv_path = ENV_PATH, override = True, encoding = 'utf-8-sig')

def get_engine():
    database_url = os.getenv('DATABASE_URL')
    if database_url:
        database_url = database_url.strip().strip('"').strip("'")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg//", 1)

    url = database_url
    connect_args = {}
    if url.startswith("postgresql+"):
        connect_args["options"] = "-c statement_timeout = 0"

    return create_engine(url, echo = False, connect_args=connect_args)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind = engine)
    return Session()